"""Stage two of retrieval: re-score candidates by reading them against the query.

Stage one (search.py) compares two vectors that were produced INDEPENDENTLY --
the question was embedded without ever seeing the chunk, and the chunk was
embedded months before the question existed. That is a bi-encoder, and it is
what makes scanning a whole corpus cheap: every chunk vector is precomputed.
The cost is precision. Each chunk is compressed to a single point, so a page
that merely mentions the right topic words can land as close as one that
actually answers the question.

A reranker is a cross-encoder. It takes the query and one chunk TOGETHER as a
single input and lets every token attend to every other token, so it can judge
"does this passage answer this question" rather than "are these two summaries
near each other". That is far more accurate and far more expensive -- which is
precisely why it runs on 20 candidates rather than 1,482 chunks.

    cheap + high recall  (find the neighbourhood)  ->  vector search, all chunks
    costly + high precision (pick the winner)      ->  reranker, top 20 only
"""

import boto3
from botocore.config import Config

import config
import db

_RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})


def make_client():
    # A THIRD Bedrock client. bedrock = control plane (listing models),
    # bedrock-runtime = invoke/converse, bedrock-agent-runtime = rerank.
    # Reaching for the wrong one gives "object has no attribute 'rerank'",
    # which reads like a boto3 version problem and is not.
    return boto3.client(
        "bedrock-agent-runtime", region_name=config.AWS_REGION, config=_RETRY_CONFIG
    )


def _model_arn() -> str:
    """Rerank takes a full model ARN, not the bare model ID the other APIs use.

    The empty account field is deliberate -- foundation models are AWS-owned,
    so the ARN has no account number in it.
    """
    return f"arn:aws:bedrock:{config.AWS_REGION}::foundation-model/{config.RERANK_MODEL_ID}"


def rerank(client, query: str, hits: list[db.Hit], top_n: int) -> list[db.Hit]:
    """Re-order hits by relevance to the query, keeping the best top_n."""
    if not hits:
        return []

    response = client.rerank(
        queries=[{"type": "TEXT", "textQuery": {"text": query}}],
        sources=[
            {
                "type": "INLINE",
                "inlineDocumentSource": {
                    "type": "TEXT",
                    "textDocument": {"text": hit.content},
                },
            }
            for hit in hits
        ],
        rerankingConfiguration={
            "type": "BEDROCK_RERANKING_MODEL",
            "bedrockRerankingConfiguration": {
                "numberOfResults": min(top_n, len(hits)),
                "modelConfiguration": {"modelArn": _model_arn()},
            },
        },
    )

    # The API returns positions into the list we sent, plus a relevance score --
    # not the documents themselves. We map back to our own Hit objects so the
    # citation metadata (filename, pages) survives the round trip. Losing that
    # mapping is the easiest way to end up citing the wrong page.
    reranked = []
    for result in response["results"]:
        hit = hits[result["index"]]
        hit.rerank_score = result["relevanceScore"]
        reranked.append(hit)
    return reranked


def retrieve(conn, bedrock_embed, question: str, use_rerank: bool) -> list[db.Hit]:
    """The full two-stage pipeline, with stage two switchable.

    The flag exists so the same question can be run both ways and the
    difference measured, rather than asserted.
    """
    import embed

    query_vector = embed.embed_one(bedrock_embed, question)
    candidates = db.search(
        conn, query_vector, config.CANDIDATE_K, config.EMBEDDING_MODEL_ID
    )

    if not use_rerank:
        return candidates[: config.FINAL_K]

    return rerank(make_client(), question, candidates, config.FINAL_K)
