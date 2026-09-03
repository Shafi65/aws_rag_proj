# Document-agnostic RAG over AWS Bedrock

Point it at a set of documents, ask a question, get an answer grounded in those
documents with citations back to source file and page — and an explicit refusal
when the documents don't support an answer.

Plain Python. No LangChain, no LlamaIndex, no framework that hides the
mechanics. Three dependencies total: `boto3`, `pypdf`, `psycopg`.

```
$ python src/answer.py "what is the AWS shared responsibility model?"

Q: what is the AWS shared responsibility model?

The AWS shared responsibility model is a framework that defines the security
responsibilities shared between AWS and its customers [1][3][4]. AWS is
responsible for "security of the cloud", which includes the infrastructure,
hardware, software, networking, and facilities that run AWS Cloud services
[2][3]. Customers are responsible for "security in the cloud", which includes
management of the guest operating system, application software, security group
configuration, data management, and IAM tools [1][3][4].

Sources:
  [1] wellarchitected-security-pillar.pdf, p. 10
  [2] wellarchitected-reliability-pillar.pdf, p. 9
  [3] wellarchitected-security-pillar.pdf, p. 186
  [4] wellarchitected-security-pillar.pdf, pp. 11-12
  [5] wellarchitected-reliability-pillar.pdf, pp. 10-11
```

Ask something the corpus doesn't cover and it says so rather than improvising:

```
$ python src/answer.py "what is the exact monthly price of an m5.large in us-east-1?"

INSUFFICIENT CONTEXT: the provided documents do not answer this question.
The documents cover cost optimization strategies and best practices but do not
provide specific pricing details for AWS services.
```

---

## Architecture

```
INGESTION (run once per document set)
─────────────────────────────────────────────────────────────────────────
  S3 bucket                  pypdf                    sliding window
  ┌──────────┐          ┌───────────────┐          ┌────────────────┐
  │  *.pdf   │─────────▶│ text +        │─────────▶│ chunks +       │
  └──────────┘          │ page offsets  │          │ page ranges    │
                        └───────────────┘          └───────┬────────┘
                                                           │
                        Bedrock: Titan Text Embeddings v2  │ 8 parallel
                        ┌──────────────────────────────────▼────────┐
                        │  1,024-dimension vector per chunk         │
                        └──────────────────┬────────────────────────┘
                                           ▼
                        Postgres 17 + pgvector  (Docker, localhost:5433)
                        ┌───────────────────────────────────────────┐
                        │ documents │ chunks (text, pages, vector)  │
                        └───────────────────────────────────────────┘

QUERY (per question)
─────────────────────────────────────────────────────────────────────────
  question ──▶ Titan ──▶ vector
                          │
                          ▼  STAGE 1: cosine search, HNSW index
                   20 candidates          cheap, high recall, whole corpus
                          │
                          ▼  STAGE 2: Cohere Rerank 3.5 (cross-encoder)
                   top 5 chunks           costly, high precision, 20 only
                          │
                          ▼  Bedrock Converse API
                   answer + [n] citations ──▶ mapped back to file + page
```

**Why two retrieval stages.** Stage one compares vectors that were produced
independently — the question never saw the chunk, the chunk never saw the
question. That is what makes it cheap enough to scan every row, and also what
limits it: each chunk is compressed to a single point, so a page that merely
*mentions* the right words can score as well as one that answers the question.
Stage two feeds the query and one chunk through a model *together*, so every
token can attend to every other token. Far more accurate, far more expensive —
hence 20 candidates, not 1,482.

The effect, measured on this corpus:

| Rank | Vector search only | After reranking |
|---|---|---|
| 1 | `reliability-pillar p. 3` — table of contents | `reliability-pillar p. 9` — definitional content |
| 2 | `cost-optimization p. 87` — "Related videos" list | **`security-pillar p. 10`** — the actual answer |
| 3 | `security-pillar p. 10` | **`security-pillar pp. 11-12`** — also the answer |

Two junk chunks dropped out of the top 5; two substantive ones were promoted in.

---

## Quickstart

Requires Docker Desktop, Python 3.13, and AWS credentials with Bedrock access
in `us-east-1`.

```bash
docker compose up -d                      # Postgres 17 + pgvector, schema auto-applied
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                      # then set S3_BUCKET

python src/check_bedrock.py               # verify AWS before building on it
aws s3 sync ./your-docs s3://your-bucket/
python src/ingest.py                      # S3 -> chunks -> vectors -> Postgres

python src/answer.py "your question here"
```

Useful variants:

```bash
python src/search.py --compare "..."      # retrieval before vs after reranking
python src/answer.py --no-rerank "..."    # generation without stage two
python src/answer.py --show-context "..." # print exactly what the model was given
python src/chunk.py path/to.pdf           # preview chunking, no AWS or DB needed
python src/ingest.py --force              # re-ingest even if content hash matches
```

Any config value can be overridden for a single run without editing `.env`:

```bash
GENERATION_MODEL_ID=us.anthropic.claude-haiku-4-5-20251001-v1:0 python src/answer.py "..."
```

---

## Configuration

Everything tunable lives in `.env`; `src/config.py` is the only module that
reads the environment.

| Variable | Default | Notes |
|---|---|---|
| `EMBEDDING_MODEL_ID` | `amazon.titan-embed-text-v2:0` | Bare model ID — embedding models are invoked directly |
| `EMBEDDING_DIM` | `1024` | Baked into `vector(1024)`; changing it is a schema migration, not a config change |
| `GENERATION_MODEL_ID` | `us.amazon.nova-lite-v1:0` | An **inference profile** ID, not a model ID. Swap to Claude with no code change. |
| `RERANK_MODEL_ID` | `cohere.rerank-v3-5:0` | Called through `bedrock-agent-runtime` |
| `CHUNK_SIZE_CHARS` | `1200` | ≈ 300 tokens |
| `CHUNK_OVERLAP_CHARS` | `200` | ≈ 17% duplication |
| `CANDIDATE_K` | `20` | Stage-one candidate pool |
| `FINAL_K` | `5` | Chunks that reach the prompt |

---

## Design decisions

| Decision | Rationale |
|---|---|
| **Postgres + pgvector, not a dedicated vector DB** | Text, metadata, and vectors live in one table, so retrieval is one query with no syncing between a metadata store and a vector store. Transactions, joins, and SQL come free. |
| **Cosine distance (`<=>`)** | Titan normalises by default (measured: norm = 1.0000), so cosine, dot product, and Euclidean rank identically. Cosine wins on interpretability — a bounded 0–2 scale that is easy to threshold. |
| **HNSW over IVFFlat** | IVFFlat derives its clusters from rows present at build time and is useless on an empty table. HNSW builds incrementally, which fits start-container-then-ingest. |
| **`page_start` and `page_end`, not one page column** | A chunk can begin at the bottom of one page and end on the next. A single column would force a wrong citation on one of them. |
| **`embedding_model` column** | Vectors from different models occupy different spaces. Mixing them produces distances that still compute, still sort, and are meaningless — with no error. Same dimension count does not mean same space, so the type system cannot catch this. |
| **`content_sha256` on documents** | Re-ingestion compares the file's hash and skips unchanged documents rather than paying to re-embed them. |
| **`DELETE` then `INSERT`, never `UPDATE`** | `ON DELETE CASCADE` only fires on a delete. Updating a document row in place would leave the previous version's chunks in the table, competing in search results with nothing to signal it. |
| **Converse API for generation** | Every model family has its own native request shape. Converse normalises them, which is what makes `GENERATION_MODEL_ID` a real config variable rather than a hardcoded assumption about one provider. |
| **Chunk size in characters, not tokens** | There is no public tokenizer for Titan, so a token count would be an estimate. Characters are exact and reproducible (~4 chars/token). |
| **Snap chunk boundaries to sentence/paragraph breaks** | A chunk starting mid-sentence embeds poorly — the model is asked to represent a fragment. ~15 lines of code. |
| **Model cites `[n]`, not page numbers** | Numbers are mapped back to real citations in code, so the model cannot invent a page that does not exist. The worst failure is citing the wrong number, not a fabricated source. |
| **Refusal is an explicit instruction with a fixed prefix** | `INSUFFICIENT CONTEXT:` is greppable, which makes refusal rate a measurable quantity rather than a vibe. |
| **8-way parallel embedding** | Titan has no batch endpoint; 1,482 sequential HTTPS calls is minutes of waiting on the network. A small thread pool makes it ~30s. |
| **No `python-dotenv`** | Reading a `.env` is eight lines of standard library. One fewer dependency to justify. |

---

## Failure modes and limitations

Known and unfixed, listed deliberately.

| Limitation | Impact | Mitigation |
|---|---|---|
| **Scanned PDFs extract nothing.** pypdf reads text, it is not OCR. | A scanned document ingests as zero chunks. | Detected and reported per document at ingest. A real fix is a Textract stage in front. |
| **Front matter is chunked like content.** Tables of contents become chunks of dot leaders. | TOC chunks appear in stage-one results (observed). | Reranking demotes them in practice. A filter on text density at ingest would remove them properly. |
| **Multi-column layouts and tables lose structure.** | Columns can interleave; table cells arrive as a run of words. | Not addressed. Would need a layout-aware extractor. |
| **No query rewriting.** A question phrased very differently from the source text retrieves poorly. | Recall depends on phrasing. | HyDE or multi-query expansion would help. |
| **Fixed chunk size regardless of document structure.** | A 1,200-character window may split a table or a procedure. | Structure-aware chunking (split on headings) is the next improvement. |
| **The vector index may not be used at this scale.** With ~1,500 rows the planner may prefer a sequential scan — and be right. | None at demo scale. | Verify with `EXPLAIN ANALYZE`; the index earns its keep at 10⁵+ rows. |
| **Single-tenant, no access control.** Any question can retrieve any chunk. | Fine for a demo corpus of public documents. | Row-level security keyed to the caller's identity, filtered in the same query. |
| **Retrieval quality is not yet measured.** | Improvements are argued, not proven. | An eval harness (recall@k, MRR before/after rerank) is the highest-value next addition. |

---

## What I'd change to productionize this on AWS

| Area | Now | Production |
|---|---|---|
| **Database** | Postgres in Docker on a laptop | Aurora PostgreSQL Serverless v2 with pgvector, Multi-AZ, in private subnets |
| **Credentials** | Long-lived IAM user access keys | IAM roles with short-lived credentials; Secrets Manager for the DB password; no static keys anywhere |
| **Ingestion trigger** | Run `ingest.py` by hand | S3 `ObjectCreated` event → Lambda (or Step Functions for large documents) — documents become searchable on upload |
| **Ingestion scale** | Sequential per document, in-process | SQS queue with per-document workers; DLQ for failures; idempotent via the existing content hash |
| **Large documents** | Whole PDF loaded into memory | Stream to `/tmp` or process page-ranges; Step Functions Map for parallelism |
| **Interface** | CLI | API Gateway + Lambda, or ECS Fargate; the retrieval code is unchanged |
| **Observability** | `print()` | Structured logs to CloudWatch; per-query metrics for retrieval latency, rerank latency, token spend, and refusal rate |
| **Cost control** | None | Budget alarms; cache embeddings for repeated queries; consider `EMBEDDING_DIM=512` at scale (half the storage and index size) |
| **Evaluation** | Manual spot checks | The eval harness in CI, gating changes to chunk size, `k`, or prompts on measured recall@k and MRR rather than intuition |
| **Data residency** | `us.` inference profile (US-only routing) | Explicit per-tenant region policy; the `global.` profiles route worldwide and would be a compliance problem for regulated data |
| **Multi-tenancy** | None | A `tenant_id` column on both tables, row-level security, and the filter pushed into the vector query so tenants never share a candidate pool |

---

## Repository layout

```
docker-compose.yml     Postgres 17 + pgvector; schema applied on first boot
schema.sql             documents + chunks tables, HNSW cosine index
requirements.txt       three dependencies
.env.example           every tunable value

src/
  config.py            .env loader (stdlib) + typed constants
  check_bedrock.py     Phase-2 connectivity check: one embedding, one generation
  extract.py           PDF bytes -> text + per-page character offsets
  chunk.py             text -> overlapping chunks tagged with page ranges
  s3_source.py         list and download documents
  embed.py             text -> 1,024-dim vector (8-way parallel)
  db.py                every SQL statement in the project
  ingest.py            S3 -> extract -> chunk -> embed -> Postgres
  search.py            retrieval CLI, with --compare for before/after reranking
  rerank.py            stage two, cross-encoder
  answer.py            the full pipeline: retrieve -> prompt -> cited answer
```

Corpus used for development: three AWS Well-Architected Framework pillar
whitepapers (Security, Reliability, Cost Optimization) — 669 pages, 1,482
chunks, ~97 seconds and roughly one cent to ingest.
