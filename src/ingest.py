"""Ingestion: S3 -> extract -> chunk -> embed -> Postgres.

Run:
    python src/ingest.py           # skip documents whose bytes have not changed
    python src/ingest.py --force   # re-ingest everything
"""

import hashlib
import sys
import time

import chunk as chunking
import config
import db
import embed
import extract
import s3_source


def ingest_one(s3, bedrock, conn, key: str, force: bool) -> dict:
    """Ingest a single S3 object. Returns a summary of what happened."""
    started = time.time()
    pdf_bytes = s3_source.download(s3, key)

    # Fingerprint the raw bytes. If it matches what we stored, the file has not
    # changed and re-embedding it would cost money for an identical result.
    content_sha256 = hashlib.sha256(pdf_bytes).hexdigest()
    if not force and db.stored_hash(conn, key) == content_sha256:
        return {"key": key, "status": "unchanged", "chunks": 0, "seconds": 0.0}

    text, page_starts = extract.extract_pages(pdf_bytes)
    if not text.strip():
        # Almost always a scanned PDF with no text layer. Worth surfacing
        # rather than silently writing a document with zero chunks.
        return {"key": key, "status": "no text extracted", "chunks": 0, "seconds": 0.0}

    chunks = chunking.chunk_text(
        text, page_starts, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS
    )
    vectors = embed.embed_many(bedrock, [c.text for c in chunks])

    # Guard against the silent-corruption case: a vector of the wrong length
    # would fail at insert anyway, but this says why.
    for vector in vectors:
        if len(vector) != config.EMBEDDING_DIM:
            raise RuntimeError(
                f"{config.EMBEDDING_MODEL_ID} returned {len(vector)} dimensions, "
                f"schema expects {config.EMBEDDING_DIM}"
            )

    db.replace_document(
        conn,
        s3_key=key,
        filename=key.rsplit("/", 1)[-1],
        content_sha256=content_sha256,
        chunks=chunks,
        vectors=vectors,
        embedding_model=config.EMBEDDING_MODEL_ID,
    )

    return {
        "key": key,
        "status": "ingested",
        "chunks": len(chunks),
        "pages": len(page_starts),
        "seconds": time.time() - started,
    }


def main() -> int:
    force = "--force" in sys.argv

    s3 = s3_source.make_client()
    bedrock = embed.make_client()

    print(f"bucket : {config.S3_BUCKET}")
    print(f"model  : {config.EMBEDDING_MODEL_ID}")
    print(f"chunk  : size={config.CHUNK_SIZE_CHARS} overlap={config.CHUNK_OVERLAP_CHARS}\n")

    with db.connect() as conn:
        keys = list(s3_source.list_pdf_keys(s3))
        if not keys:
            print("No .pdf objects found. Upload documents first.")
            return 1

        for key in keys:
            print(f"  {key} ...", end=" ", flush=True)
            result = ingest_one(s3, bedrock, conn, key, force)
            if result["status"] == "ingested":
                print(
                    f"{result['chunks']} chunks from {result['pages']} pages "
                    f"in {result['seconds']:.1f}s"
                )
            else:
                print(result["status"])

        stats = db.corpus_stats(conn)

    print(f"\ncorpus: {stats['documents']} documents, {stats['chunks']} chunks")
    if stats["embedding_models"] > 1:
        # Vectors from different models occupy different spaces. Distances
        # between them still compute, still sort, and are meaningless.
        print(
            f"WARNING: {stats['embedding_models']} different embedding models present. "
            f"Re-ingest with --force so every vector comes from one model."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
