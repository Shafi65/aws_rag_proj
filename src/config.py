"""Configuration, read once from .env at import time.

Every tunable value in the system lives here. Nothing else in the codebase
reads os.environ directly, so when a setting misbehaves there is exactly one
place to look.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_env(path: Path) -> None:
    """Read KEY=value lines from a .env file into the environment.

    This is deliberately not python-dotenv. Eight lines is the whole feature,
    and a dependency we could read in eight lines is a dependency we do not
    need to explain.

    setdefault (not assignment) means a real environment variable always wins,
    so any setting can be overridden for a single run without editing the file:

        CHUNK_SIZE_CHARS=800 python src/ingest.py

    Known limitation, fine for our purposes: no quote stripping and no support
    for inline comments after a value.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator:
            os.environ.setdefault(key.strip(), value.strip())


_load_env(PROJECT_ROOT / ".env")


def _require(name: str) -> str:
    """Fail loudly at import time rather than with a confusing error later."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"{name} is not set. Copy .env.example to .env and fill it in."
        )
    return value


# ---------- AWS ----------
AWS_REGION = _require("AWS_REGION")
S3_BUCKET = _require("S3_BUCKET")
S3_PREFIX = os.environ.get("S3_PREFIX", "")  # optional: empty means whole bucket

# ---------- Postgres ----------
PGHOST = _require("PGHOST")
PGPORT = int(_require("PGPORT"))
PGDATABASE = _require("PGDATABASE")
PGUSER = _require("PGUSER")
PGPASSWORD = _require("PGPASSWORD")

# ---------- Models ----------
# Bare model ID -- embedding models are invoked directly.
EMBEDDING_MODEL_ID = _require("EMBEDDING_MODEL_ID")
EMBEDDING_DIM = int(_require("EMBEDDING_DIM"))
# Inference profile ID ("us." prefix) -- newer generation models pool
# on-demand capacity across regions and reject the bare model ID.
GENERATION_MODEL_ID = _require("GENERATION_MODEL_ID")
RERANK_MODEL_ID = _require("RERANK_MODEL_ID")

# ---------- Chunking ----------
CHUNK_SIZE_CHARS = int(_require("CHUNK_SIZE_CHARS"))
CHUNK_OVERLAP_CHARS = int(_require("CHUNK_OVERLAP_CHARS"))

# ---------- Retrieval ----------
CANDIDATE_K = int(_require("CANDIDATE_K"))  # candidates from vector search
FINAL_K = int(_require("FINAL_K"))  # survivors after reranking
