"""The CLI: ask a question, get an answer grounded in the documents.

    python src/answer.py "what is the shared responsibility model?"
    python src/answer.py --no-rerank "..."     # skip stage two
    python src/answer.py --show-context "..."  # print what was retrieved

Retrieval decides what the model is allowed to know; this file decides what it
is allowed to do with it. The system prompt is the part that turns a search
engine into something trustworthy: it forbids outside knowledge, requires a
citation on every claim, and defines an explicit way to say "I don't know".

That refusal path matters more than it looks. A RAG system that always answers
is worse than useless on the questions its corpus does not cover -- it produces
fluent, confident, unsupported text, and the citations make it look MORE
credible rather than less.
"""

import sys

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

import config
import db
import embed
import rerank as reranking

SYSTEM_PROMPT = """You answer questions using only the numbered sources provided in the user's message.

Rules:
1. Use only information present in the sources. Do not use prior knowledge, and do not infer beyond what is written.
2. Cite the source for every factual claim, inline, using its number in square brackets: [1], [2]. A sentence drawing on two sources cites both.
3. If the sources do not contain enough information to answer the question, reply with exactly:
   INSUFFICIENT CONTEXT: the provided documents do not answer this question.
   Then, in one sentence, say what they do cover. Do not attempt a partial answer from outside knowledge.
4. If sources disagree, say so and cite both rather than silently picking one.
5. Be concise. Answer the question asked; do not summarise the sources.
"""


def build_context(hits: list[db.Hit]) -> str:
    """Number each chunk so the model has something concrete to cite.

    The number, not the filename, is what the model cites -- short, unambiguous,
    and impossible to garble. We map numbers back to real citations ourselves
    when printing, so the model can never invent a page number that does not
    exist. Letting it write "page 12" directly would make hallucinated
    citations possible; this way the worst it can do is cite the wrong number.
    """
    blocks = []
    for number, hit in enumerate(hits, start=1):
        blocks.append(f"[{number}] {hit.citation()}\n{hit.content}")
    return "\n\n".join(blocks)


def generate(client, question: str, hits: list[db.Hit]) -> dict:
    response = client.converse(
        modelId=config.GENERATION_MODEL_ID,
        system=[{"text": SYSTEM_PROMPT}],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": (
                            f"Sources:\n\n{build_context(hits)}\n\n"
                            f"Question: {question}"
                        )
                    }
                ],
            }
        ],
        inferenceConfig={"maxTokens": 800, "temperature": 0.0},
    )
    return {
        "text": response["output"]["message"]["content"][0]["text"],
        "usage": response["usage"],
    }


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    if not args:
        print(__doc__)
        return 1

    question = " ".join(args)
    use_rerank = "--no-rerank" not in flags

    bedrock_embed = embed.make_client()
    bedrock_gen = boto3.client(
        "bedrock-runtime",
        region_name=config.AWS_REGION,
        config=Config(retries={"max_attempts": 5, "mode": "adaptive"}),
    )

    with db.connect() as conn:
        hits = reranking.retrieve(conn, bedrock_embed, question, use_rerank)

    if not hits:
        print("No chunks retrieved. Has ingestion run?")
        return 1

    if "--show-context" in flags:
        print("=== RETRIEVED CONTEXT ===")
        print(build_context(hits))
        print()

    try:
        result = generate(bedrock_gen, question, hits)
    except ClientError as error:
        # The common ones are account-level, not code-level, and the error code
        # is the only part that tells you which. A stack trace buries it.
        info = error.response["Error"]
        print(f"{info['Code']}: {info['Message']}")
        if "use case details" in info["Message"]:
            print(
                "\nAnthropic models on Bedrock need a one-time use-case form:\n"
                "  console.aws.amazon.com/bedrock -> Model access -> Anthropic\n"
                f"Meanwhile, any other model works without code changes:\n"
                f"  GENERATION_MODEL_ID=us.amazon.nova-lite-v1:0 python src/answer.py \"...\""
            )
        return 1

    print(f"Q: {question}\n")
    print(result["text"])
    print("\nSources:")
    for number, hit in enumerate(hits, start=1):
        print(f"  [{number}] {hit.citation()}")

    usage = result["usage"]
    print(
        f"\n({config.GENERATION_MODEL_ID.split('.')[-1]} | "
        f"rerank={'on' if use_rerank else 'off'} | "
        f"tokens in={usage['inputTokens']} out={usage['outputTokens']})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
