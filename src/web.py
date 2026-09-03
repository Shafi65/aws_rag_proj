"""A small web UI for demonstrating the pipeline.

    python src/web.py            # http://127.0.0.1:8000
    python src/web.py --port 9000

Standard library only -- http.server, no Flask, no FastAPI, no Streamlit. The
project's claim is three dependencies; a UI is not a good reason to add a
fourth. The page is a single self-contained HTML file with no CDN links, so it
also works with no internet, which matters when demoing on venue wifi.

This is a thin shell over the same functions the CLI uses. It adds no retrieval
logic of its own -- if it did, the demo would be showing something other than
the system.
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import answer as answering
import config
import db
import embed
import rerank as reranking

PAGE = Path(__file__).resolve().parent.parent / "web" / "index.html"


def _hit_to_dict(hit: db.Hit, rank: int) -> dict:
    return {
        "rank": rank,
        "chunk_id": hit.chunk_id,
        "citation": hit.citation(),
        "filename": hit.filename,
        "page_start": hit.page_start,
        "page_end": hit.page_end,
        "distance": round(hit.distance, 4),
        "score": round(hit.rerank_score, 4) if hit.rerank_score is not None else None,
        "preview": " ".join(hit.content.split()),
    }


def ask(question: str, use_rerank: bool) -> dict:
    """Run the pipeline, timing each stage so the UI can show where time goes."""
    bedrock_embed = embed.make_client()
    bedrock_gen = boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    started = time.time()
    query_vector = embed.embed_one(bedrock_embed, question)
    embed_ms = (time.time() - started) * 1000

    with db.connect() as conn:
        started = time.time()
        candidates = db.search(
            conn, query_vector, config.CANDIDATE_K, config.EMBEDDING_MODEL_ID
        )
        search_ms = (time.time() - started) * 1000
        stats = db.corpus_stats(conn)

    # Always compute stage one's own top-k, so the UI can show what reranking
    # changed rather than just showing the final order.
    baseline = candidates[: config.FINAL_K]
    baseline_rank = {hit.chunk_id: i for i, hit in enumerate(candidates, start=1)}

    rerank_ms = 0.0
    if use_rerank:
        started = time.time()
        final = reranking.rerank(
            reranking.make_client(), question, candidates, config.FINAL_K
        )
        rerank_ms = (time.time() - started) * 1000
    else:
        final = baseline

    started = time.time()
    try:
        generated = answering.generate(bedrock_gen, question, final)
    except ClientError as error:
        info = error.response["Error"]
        return {"error": f"{info['Code']}: {info['Message']}"}
    generate_ms = (time.time() - started) * 1000

    stage2 = []
    for rank, hit in enumerate(final, start=1):
        row = _hit_to_dict(hit, rank)
        # Where this chunk sat in stage one -- the number that makes the
        # reranker's effect visible instead of asserted.
        row["was_rank"] = baseline_rank.get(hit.chunk_id)
        stage2.append(row)

    text = generated["text"]
    return {
        "question": question,
        "answer": text,
        "refused": text.strip().startswith("INSUFFICIENT CONTEXT"),
        "rerank": use_rerank,
        "stage1": [_hit_to_dict(h, i) for i, h in enumerate(candidates, start=1)],
        "stage1_top": [_hit_to_dict(h, i) for i, h in enumerate(baseline, start=1)],
        "stage2": stage2,
        "timings": {
            "embed": round(embed_ms),
            "search": round(search_ms),
            "rerank": round(rerank_ms),
            "generate": round(generate_ms),
        },
        "usage": generated["usage"],
        "corpus": stats,
        "config": {
            "candidate_k": config.CANDIDATE_K,
            "final_k": config.FINAL_K,
            "chunk_size": config.CHUNK_SIZE_CHARS,
            "chunk_overlap": config.CHUNK_OVERLAP_CHARS,
            "embedding_model": config.EMBEDDING_MODEL_ID,
            "generation_model": config.GENERATION_MODEL_ID,
            "rerank_model": config.RERANK_MODEL_ID,
        },
    }


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif self.path == "/api/config":
            payload = {
                "candidate_k": config.CANDIDATE_K,
                "final_k": config.FINAL_K,
                "chunk_size": config.CHUNK_SIZE_CHARS,
                "chunk_overlap": config.CHUNK_OVERLAP_CHARS,
                "generation_model": config.GENERATION_MODEL_ID,
            }
            with db.connect() as conn:
                payload["corpus"] = db.corpus_stats(conn)
            self._send(200, json.dumps(payload).encode(), "application/json")
        else:
            self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self._send(404, b"not found", "text/plain")
            return
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        question = (request.get("question") or "").strip()
        if not question:
            self._send(400, b'{"error":"empty question"}', "application/json")
            return
        result = ask(question, bool(request.get("rerank", True)))
        self._send(200, json.dumps(result).encode(), "application/json")

    def log_message(self, fmt, *args):
        # One tidy line per request instead of the default noise.
        print(f"  {self.command} {self.path}")


def main() -> int:
    port = 8000
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])

    with db.connect() as conn:
        stats = db.corpus_stats(conn)

    print(f"corpus     : {stats['documents']} documents, {stats['chunks']} chunks")
    print(f"generation : {config.GENERATION_MODEL_ID}")
    print(f"retrieval  : {config.CANDIDATE_K} candidates -> {config.FINAL_K} final")
    print(f"\n  http://127.0.0.1:{port}\n")

    # 127.0.0.1, not 0.0.0.0: this binds to the loopback interface only, so it
    # is not reachable from the network. There is no authentication here.
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
