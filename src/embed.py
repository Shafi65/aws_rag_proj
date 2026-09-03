"""Text -> vector, via Titan Text Embeddings v2 on Bedrock.

The same function embeds a chunk at ingest time and a question at query time.
That is not a convenience -- it is a correctness requirement. A question and
the chunks it should match have to be placed in the same coordinate space by
the same model, or the distances between them mean nothing.
"""

import json
from concurrent.futures import ThreadPoolExecutor

import boto3
from botocore.config import Config

import config

# Bedrock throttles on burst. Adaptive retry backs off and slows the client
# down when it sees ThrottlingException, rather than hammering and failing.
# Without this, a few hundred rapid embedding calls will drop some requests.
_RETRY_CONFIG = Config(retries={"max_attempts": 5, "mode": "adaptive"})

# Titan v2 accepts up to 8192 tokens (~32K characters). Our chunks are ~1200
# characters, so this is a guard against a pathological input rather than a
# limit we expect to reach.
MAX_INPUT_CHARS = 30000


def make_client():
    return boto3.client(
        "bedrock-runtime", region_name=config.AWS_REGION, config=_RETRY_CONFIG
    )


def embed_one(client, text: str) -> list[float]:
    """One text in, EMBEDDING_DIM floats out."""
    response = client.invoke_model(
        modelId=config.EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": text[:MAX_INPUT_CHARS]}),
    )
    return json.loads(response["body"].read())["embedding"]


def embed_many(client, texts: list[str], workers: int = 8) -> list[list[float]]:
    """Embed a list of texts, preserving order.

    Titan's API takes one text per call -- there is no batch endpoint -- so a
    thousand chunks means a thousand HTTP requests. Sequentially that is
    minutes of mostly waiting on the network. A small thread pool turns it into
    seconds. boto3 clients are safe to share across threads.

    Order matters: pool.map returns results in input order, so vectors[i]
    always corresponds to texts[i]. Anything that reordered these would pair
    chunks with the wrong vectors and corrupt retrieval silently.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda text: embed_one(client, text), texts))
