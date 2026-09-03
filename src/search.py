"""Retrieval, with stage two switchable so the difference can be measured.

    python src/search.py "what is the shared responsibility model"
    python src/search.py --rerank "what is the shared responsibility model"
    python src/search.py --compare "what is the shared responsibility model"

--compare runs the same query both ways and prints them side by side. That is
the honest way to justify a reranker: show what stage one alone returns, then
show what changed.
"""

import sys

import config
import db
import embed
import rerank as reranking


def _format(hits: list[db.Hit], label: str) -> None:
    print(f"--- {label} ---")
    for rank, hit in enumerate(hits, start=1):
        score = (
            f"rerank {hit.rerank_score:.4f}"
            if hit.rerank_score is not None
            else f"dist {hit.distance:.4f}"
        )
        preview = " ".join(hit.content.split())[:150]
        print(f"{rank:>2}. [{score}] {hit.citation()}")
        print(f"    {preview}...")
    print()


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if not args:
        print(__doc__)
        return 1

    question = " ".join(args)
    bedrock = embed.make_client()

    print(f'query: "{question}"')
    print(f"stage one: {config.CANDIDATE_K} candidates -> stage two: {config.FINAL_K}\n")

    with db.connect() as conn:
        if "--compare" in flags:
            baseline = reranking.retrieve(conn, bedrock, question, use_rerank=False)
            _format(baseline, "VECTOR SEARCH ONLY (stage one)")
            reranked = reranking.retrieve(conn, bedrock, question, use_rerank=True)
            _format(reranked, "AFTER RERANKING (stage two)")

            # The concrete win: did reranking pull something into the top-k
            # that stage one had ranked below the cutoff?
            before = [h.chunk_id for h in baseline]
            after = [h.chunk_id for h in reranked]
            promoted = [c for c in after if c not in before]
            dropped = [c for c in before if c not in after]
            print(f"promoted into top-{config.FINAL_K}: {len(promoted)} chunk(s)")
            print(f"dropped out of top-{config.FINAL_K}: {len(dropped)} chunk(s)")
        else:
            use_rerank = "--rerank" in flags
            hits = reranking.retrieve(conn, bedrock, question, use_rerank)
            _format(hits, "AFTER RERANKING" if use_rerank else "VECTOR SEARCH ONLY")

    return 0


if __name__ == "__main__":
    sys.exit(main())
