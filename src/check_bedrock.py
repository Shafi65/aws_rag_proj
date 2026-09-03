"""Phase 2 connectivity check: one embedding call, one generation call.

Deliberately the smallest thing that proves AWS works. Between them these two
calls confirm credentials, region, per-model access, request format, and
response parsing -- with no S3, no database, and no pipeline involved. Once
this passes, any later failure is our code rather than AWS.

Run:  python src/check_bedrock.py
"""

import json
import math
import sys

import boto3
from botocore.exceptions import ClientError

import config


def check_embedding(client) -> bool:
    """Embed a trivial string with Titan and inspect what comes back.

    The dimension check is the point: it is the first empirical confirmation
    that vector(1024) in schema.sql matches what this model actually returns.
    """
    response = client.invoke_model(
        modelId=config.EMBEDDING_MODEL_ID,
        body=json.dumps({"inputText": "hello world"}),
    )
    payload = json.loads(response["body"].read())
    vector = payload["embedding"]

    # Titan normalizes by default, so unit length is expected. That is what
    # makes cosine distance and dot product rank identically, and is why the
    # schema uses the <=> operator.
    magnitude = math.sqrt(sum(v * v for v in vector))

    print(f"  model       : {config.EMBEDDING_MODEL_ID}")
    print(f"  dimensions  : {len(vector)}  (schema declares {config.EMBEDDING_DIM})")
    print(f"  magnitude   : {magnitude:.4f}  (~1.0 = normalized)")
    print(f"  first four  : {[round(v, 4) for v in vector[:4]]}")
    print(f"  input tokens: {payload.get('inputTextTokenCount')}")

    if len(vector) != config.EMBEDDING_DIM:
        print(
            f"  MISMATCH: schema.sql declares vector({config.EMBEDDING_DIM}), "
            f"so every insert would fail. Fix before ingesting."
        )
        return False
    return True


def check_generation(client) -> bool:
    """One Converse call.

    Converse rather than invoke_model because it normalizes the message format
    across model families -- which is what lets GENERATION_MODEL_ID be a real
    config variable instead of a hardcoded assumption about one provider's
    request shape.
    """
    response = client.converse(
        modelId=config.GENERATION_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": "Reply with exactly: OK"}]}],
        inferenceConfig={"maxTokens": 20},
    )
    text = response["output"]["message"]["content"][0]["text"]
    usage = response["usage"]

    print(f"  model       : {config.GENERATION_MODEL_ID}")
    print(f"  reply       : {text.strip()!r}")
    print(f"  tokens      : in={usage['inputTokens']} out={usage['outputTokens']}")
    print(f"  stop reason : {response['stopReason']}")

    return bool(text.strip())


def main() -> int:
    print(f"region: {config.AWS_REGION}\n")

    # bedrock-runtime is the data plane (invoking models). The plain "bedrock"
    # client is the control plane (listing models); "bedrock-agent-runtime" is
    # a third client, used for reranking in Phase 5.
    runtime = boto3.client("bedrock-runtime", region_name=config.AWS_REGION)

    all_passed = True
    for label, check in (("EMBEDDING", check_embedding), ("GENERATION", check_generation)):
        print(f"[{label}]")
        try:
            all_passed = check(runtime) and all_passed
        except ClientError as error:
            # Surface the AWS error code rather than a stack trace.
            # AccessDeniedException here almost always means "this model is not
            # enabled for this account in this region" -- not bad credentials,
            # which sts get-caller-identity already ruled out.
            info = error.response["Error"]
            print(f"  FAILED {info['Code']}: {info['Message']}")
            all_passed = False
        print()

    print("PASS -- Bedrock is reachable and both calls work." if all_passed else "FAIL")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
