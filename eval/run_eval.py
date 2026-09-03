"""Measure retrieval quality, and whether reranking actually helps.

    python eval/run_eval.py              # retrieval metrics
    python eval/run_eval.py --refusal    # also test the refusal path (costs generation calls)
    python eval/run_eval.py --verbose    # per-question detail

Retrieval quality is the ceiling on answer quality: if the right chunk never
reaches the prompt, no amount of prompt engineering rescues the answer. So this
measures retrieval directly rather than judging final answers -- fewer moving
parts, deterministic, and no LLM-as-judge to validate first.

METRICS

  recall@20   Did stage one put a relevant chunk anywhere in the candidate pool?
              This is stage one's actual job. If this is low, reranking cannot
              help -- it only reorders what it is given, so recall@20 is the
              hard ceiling on everything downstream.

  hit-rate@5  Did a relevant chunk reach the prompt? Reported before and after
              reranking, which is the comparison that justifies stage two.

  MRR@5       Mean reciprocal rank: 1/position of the first relevant chunk,
              averaged, 0 when none is in the top 5. Hit-rate treats rank 1 and
              rank 5 as equal; MRR does not. Position matters because models
              attend unevenly across a long context.

GROUND TRUTH is (file, page range), not chunk id -- chunk ids change whenever
chunking parameters change, which would invalidate the eval set for the exact
experiment it exists to support.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402
import db  # noqa: E402
import embed  # noqa: E402
import rerank as reranking  # noqa: E402

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def is_relevant(hit: db.Hit, expected_file: str, expected_pages: list[int]) -> bool:
    """Relevant if the file matches and the page ranges overlap at all."""
    if hit.filename != expected_file:
        return False
    low, high = expected_pages
    return hit.page_start <= high and hit.page_end >= low


def first_relevant_rank(hits, expected_file, expected_pages) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if is_relevant(hit, expected_file, expected_pages):
            return rank
    return None


def reciprocal_rank(rank: int | None) -> float:
    return 1.0 / rank if rank else 0.0


def evaluate_retrieval(verbose: bool) -> dict:
    questions = json.loads(QUESTIONS_PATH.read_text())["answerable"]
    bedrock = embed.make_client()
    rerank_client = reranking.make_client()

    recall_at_k = 0
    hits_before = hits_after = 0
    rr_before_total = rr_after_total = 0.0
    rows = []

    with db.connect() as conn:
        for item in questions:
            vector = embed.embed_one(bedrock, item["question"])
            candidates = db.search(
                conn, vector, config.CANDIDATE_K, config.EMBEDDING_MODEL_ID
            )
            baseline = candidates[: config.FINAL_K]
            reranked = reranking.rerank(
                rerank_client, item["question"], candidates, config.FINAL_K
            )

            in_pool = first_relevant_rank(candidates, item["file"], item["pages"])
            rank_before = first_relevant_rank(baseline, item["file"], item["pages"])
            rank_after = first_relevant_rank(reranked, item["file"], item["pages"])

            recall_at_k += 1 if in_pool else 0
            hits_before += 1 if rank_before else 0
            hits_after += 1 if rank_after else 0
            rr_before_total += reciprocal_rank(rank_before)
            rr_after_total += reciprocal_rank(rank_after)

            rows.append(
                {
                    "id": item["id"],
                    "in_pool": in_pool,
                    "before": rank_before,
                    "after": rank_after,
                    "question": item["question"],
                }
            )

    total = len(questions)
    print(f"Retrieval evaluation -- {total} questions with known source pages")
    print(f"candidates={config.CANDIDATE_K}  final_k={config.FINAL_K}  "
          f"chunk={config.CHUNK_SIZE_CHARS}/{config.CHUNK_OVERLAP_CHARS}\n")

    print(f"  STAGE ONE (vector search)")
    print(f"    recall@{config.CANDIDATE_K:<3}        {recall_at_k}/{total}   "
          f"{recall_at_k / total:.2f}   <- ceiling for everything below\n")

    print(f"  TOP {config.FINAL_K} INTO THE PROMPT      before rerank -> after rerank")
    print(f"    hit-rate@{config.FINAL_K}          "
          f"{hits_before / total:.2f}  ->  {hits_after / total:.2f}    "
          f"({hits_before}/{total} -> {hits_after}/{total})")
    print(f"    MRR@{config.FINAL_K}               "
          f"{rr_before_total / total:.3f} ->  {rr_after_total / total:.3f}")

    if verbose:
        print(f"\n  {'id':<5} {'pool':>5} {'before':>7} {'after':>6}  question")
        for row in rows:
            fmt = lambda v: str(v) if v else "-"  # noqa: E731
            print(
                f"  {row['id']:<5} {fmt(row['in_pool']):>5} "
                f"{fmt(row['before']):>7} {fmt(row['after']):>6}  "
                f"{row['question'][:58]}"
            )
        print("  (numbers are the rank of the first relevant chunk; - means not found)")

    return {
        "recall": recall_at_k / total,
        "hit_before": hits_before / total,
        "hit_after": hits_after / total,
        "mrr_before": rr_before_total / total,
        "mrr_after": rr_after_total / total,
    }


def evaluate_refusal() -> None:
    """Does the system decline when the corpus does not support an answer?

    A RAG system that always answers is worst exactly where it is least
    trustworthy. The fixed INSUFFICIENT CONTEXT prefix is what makes this
    countable rather than a matter of impression.
    """
    import boto3
    import answer as answering

    questions = json.loads(QUESTIONS_PATH.read_text())["unanswerable"]
    bedrock_embed = embed.make_client()
    bedrock_gen = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

    refused = 0
    print(f"\nRefusal evaluation -- {len(questions)} questions the corpus cannot answer\n")

    with db.connect() as conn:
        for item in questions:
            hits = reranking.retrieve(conn, bedrock_embed, item["question"], True)
            result = answering.generate(bedrock_gen, item["question"], hits)
            declined = result["text"].strip().startswith("INSUFFICIENT CONTEXT")
            refused += 1 if declined else 0
            print(f"  {item['id']}  {'REFUSED ' if declined else 'ANSWERED'}  "
                  f"{item['question'][:60]}")
            if not declined:
                print(f"        -> {result['text'][:150]}")

    print(f"\n    refusal rate      {refused}/{len(questions)}  "
          f"{refused / len(questions):.2f}")


def main() -> int:
    flags = set(sys.argv[1:])
    evaluate_retrieval(verbose="--verbose" in flags)
    if "--refusal" in flags:
        evaluate_refusal()
    return 0


if __name__ == "__main__":
    sys.exit(main())
