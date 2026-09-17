# Document-grounded question answering over AWS whitepapers

Ask a question in plain English. Get an answer drawn only from the source documents,
cited to file and page — and an explicit refusal when the documents don't support one.

Two-stage retrieval over Postgres + pgvector, with AWS Bedrock for embeddings,
reranking, and generation. Plain Python, three dependencies, no RAG framework.

**3 documents · 669 pages · 1,482 chunks · recall@20 1.00 · hit-rate@5 0.93 → 1.00 with reranking · refuses 4/4 unanswerable questions**

---

```
$ make ask q="what is the AWS shared responsibility model?"

AWS is responsible for "security of the cloud" — infrastructure, hardware,
networking, facilities [2][3]. Customers are responsible for "security in the
cloud" — guest OS, application software, security groups, data, and IAM [1][3][4].

  [1] wellarchitected-security-pillar.pdf, p. 10
  [2] wellarchitected-reliability-pillar.pdf, p. 9
  [3] wellarchitected-security-pillar.pdf, p. 186
```

```
$ make ask q="what is the monthly price of an m5.large in us-east-1?"

INSUFFICIENT CONTEXT: the provided documents do not answer this question.
The documents cover cost optimization strategies but not specific pricing.
```

---

## Architecture

**Ingestion** runs once per document set:

```
S3 → pypdf extract → chunk (1,200 chars, 200 overlap) → Titan embed → Postgres
```

Pages are joined into one string before chunking so sentences survive page breaks;
per-page character offsets are recorded separately, which is what lets a chunk be
cited as `pp. 223-224`. Text, page range, and a 1,024-dimension vector land in a
single row.

**Query** runs two retrieval stages:

```
question → Titan embed → cosine search over 1,482 chunks (HNSW) → 20 candidates
        → Cohere rerank → 5 chunks → Nova Lite → cited answer or refusal
```

Stage one is a bi-encoder: chunk vectors are precomputed, so scanning the whole
corpus is cheap. Stage two is a cross-encoder: it reads the query and each candidate
together, which is far more accurate and cannot be precomputed — hence 20 candidates,
not 1,482.

| Layer | Choice |
|---|---|
| Store | Postgres 17 + pgvector, HNSW index, cosine distance |
| Embedding | Titan Text Embeddings v2, 1,024 dimensions |
| Rerank | Cohere Rerank 3.5 |
| Generation | Nova Lite via the Bedrock Converse API |
| Dependencies | `boto3`, `psycopg[binary]`, `pypdf` |

## Run it

Requires Docker, Python 3.13, and AWS credentials with Bedrock access in `us-east-1`.

```bash
make up                        # Postgres + pgvector, schema applied on first boot
make ingest                    # S3 → chunks → vectors → Postgres
make ask q="your question"
make web                       # browser UI at 127.0.0.1:8765
make eval                      # retrieval metrics + refusal rate
```

## Results

19 questions: 15 with known source pages, 4 the corpus cannot answer. Ground truth is
**(file, page range)**, not chunk ID — chunk IDs change whenever chunk size changes,
which would invalidate the eval set for the experiment it exists to support.

| Metric | Value | |
|---|---|---|
| `recall@20` | **1.00** | stage one found the answer every time — the ceiling on everything below |
| `hit-rate@5` | **0.93 → 1.00** | reranking got the last one into the prompt |
| `MRR@5` | **0.856 → 0.917** | relevant chunks moved closer to rank 1 |
| refusal rate | **1.00** (4/4) | including two questions that look in-domain |

On *"what is chaos engineering"*, stage one ranked the relevant chunk **6th** — outside
the top 5, so it never reached the model. Reranking moved it to 4th.

One regression: q11 went from rank 1 to 2. Reranking is better on average, not
uniformly. And 15 questions is a small sample — `recall@20 = 1.00` means "no failures
in 15 trials", not "recall is 100%".

`make eval` reproduces all of the above.

## Design decisions

| Decision | Why |
|---|---|
| Postgres + pgvector, not a dedicated vector DB | Text, metadata, and vectors in one table — retrieval is one query, nothing to keep in sync |
| HNSW, not IVFFlat | IVFFlat builds its clusters from rows present at build time, so it is useless on an empty table |
| `page_start` **and** `page_end` | A chunk can span a page break; one column would force a wrong citation |
| `embedding_model` column | Vectors from different models are different spaces. Mixing them still sorts, silently wrong — and equal dimension count does not mean equal space, so the type system cannot catch it |
| `DELETE` then `INSERT`, never `UPDATE` | `ON DELETE CASCADE` only fires on delete; an update would leave the old version's chunks competing in search |
| Converse API for generation | Normalises request shape across model families, which is what makes the model ID a real config variable |
| Model cites `[n]`, never page numbers | Numbers map back to file and page in code, so a fabricated page is structurally impossible |

## Limitations

- **Scanned PDFs extract nothing** — pypdf reads text, it is not OCR. Reported per document rather than silently ingested.
- **Front matter is indexed like content** — a heading-and-header chunk ranked 5th on a real query before reranking demoted it. A text-density filter at ingest would remove it properly.
- **Chunks begin mid-word** — boundaries are snapped to sentence breaks at the end of a chunk but not at the start. No measured impact on recall, but it is a defect.
- **Eval is 19 questions on one document family** — enough to catch a regression, not to trust a small difference.
- **Chunk size is a reasonable default, not a tuned one** — tuning it properly is what the eval harness exists for.
