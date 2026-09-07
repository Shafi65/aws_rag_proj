# Document-agnostic RAG over AWS Bedrock

Ask questions about a set of PDFs. Get answers grounded in those documents, cited to
file and page — and an explicit refusal when the documents don't support one.

Postgres + pgvector, AWS Bedrock, two-stage retrieval. Plain Python, three
dependencies, no RAG framework.

**1,482 chunks from 669 pages · recall@20 1.00 · hit-rate@5 0.93 → 1.00 with reranking · refuses 4/4 unanswerable questions**

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

## Run it

Needs Docker, Python 3.13, and AWS credentials with Bedrock access in `us-east-1`.

```bash
make up                        # Postgres + pgvector, schema applied on first boot
make ingest                    # S3 → chunks → vectors → Postgres
make web                       # browser UI at 127.0.0.1:8765
make ask q="your question"
make eval                      # retrieval metrics + refusal rate
```

## How it works

**Ingest**, once per document set: read PDFs from S3 → extract text, recording where
each page starts → slide a 1,200-character window with 200 of overlap → embed each
chunk with Titan → store text, page range, and a 1,024-dimension vector in one
Postgres row.

Pages are joined into a single string before chunking rather than handled separately,
so a sentence spanning a page break survives intact. The recorded page offsets are
what let a chunk cite `pp. 11-12`.

**Query**: embed the question with the same model → cosine search over all 1,482
chunks (~20ms) → rerank the top 20 → send the best 5 to the model → answer with
citations.

**Why two retrieval stages.** Stage one is a bi-encoder: the question and each chunk
were embedded *independently*, so comparing them is cheap geometry over precomputed
vectors — cheap enough to scan the whole corpus. It is also imprecise, because each
chunk is compressed to a single point, so a page that merely *mentions* the right
words can score as well as one that answers. On a real query it ranked a table of
contents first. Stage two is a cross-encoder: it reads the query and one chunk
*together*, so every token attends to every other. Far more accurate, far more
expensive — hence 20 candidates, not 1,482.

Retrieval is local. All three model calls (embed, rerank, generate) go to Bedrock, so
the system needs network access.

## Results

15 questions with known source pages, plus 4 the corpus cannot answer. Ground truth is
**(file, page range)**, not chunk ID — chunk IDs change whenever chunk size changes,
which would invalidate the eval set for the experiment it exists to support.

| Metric | Value | |
|---|---|---|
| `recall@20` | **1.00** | stage one found the answer every time — the ceiling on everything below |
| `hit-rate@5` | **0.93 → 1.00** | reranking got the last one into the prompt |
| `MRR@5` | **0.856 → 0.917** | relevant chunks moved closer to rank 1 |
| refusal rate | **1.00** (4/4) | including two questions that look in-domain |

The win: on *"what is chaos engineering"*, stage one ranked the relevant chunk **6th** —
outside the top 5, so it never reached the model. Reranking moved it to 4th.

One regression: q11 went from rank 1 to 2. Reranking is better on average, not
uniformly. And 15 questions is a small sample — `recall@20 = 1.00` means "no failures
in 15 trials", not "recall is 100%".

## Decisions

| Decision | Why |
|---|---|
| Postgres + pgvector, not a dedicated vector DB | Text, metadata, and vectors in one table — retrieval is one query, nothing to keep in sync |
| Cosine distance (`<=>`) | Titan normalises by default (measured: norm 1.0000), so the choice is interpretability: a bounded 0–2 scale |
| HNSW, not IVFFlat | IVFFlat builds its clusters from rows present at build time, so it is useless on an empty table |
| `page_start` **and** `page_end` | A chunk can span a page break; one column would force a wrong citation |
| `embedding_model` column | Vectors from different models are different spaces. Mixing them still sorts, silently wrong — and equal dimension count does not mean equal space, so the type system cannot catch it |
| `DELETE` then `INSERT`, never `UPDATE` | `ON DELETE CASCADE` only fires on delete; an update would leave the old version's chunks competing in search |
| Converse API for generation | Normalises request shape across model families, which is what makes the model ID a real config variable |
| Model cites `[n]`, never page numbers | Numbers map back to file and page in code, so a fabricated page is structurally impossible |

## Limitations

- **Scanned PDFs extract nothing** — pypdf reads text, it is not OCR. Reported per document rather than silently ingested; a real fix is Textract in front.
- **Front matter is indexed like content** — a table-of-contents chunk ranked first on a real query. Reranking demoted it; a text-density filter at ingest would remove it properly.
- **Eval is 15 questions on one document family** — enough to catch a regression, not to trust a small difference.
- **Chunk size is a reasonable default, not a tuned one** — tuning it properly is what the eval harness exists for.

## Productionizing on AWS

- **Aurora PostgreSQL Serverless v2** with pgvector, Multi-AZ, private subnets — instead of Docker on a laptop
- **IAM roles** with short-lived credentials and Secrets Manager, instead of long-lived access keys
- **S3 `ObjectCreated` → Lambda** so documents become searchable on upload; SQS with a DLQ at scale, idempotent via the existing content hash
- **API Gateway + Lambda** behind Cognito instead of a local server — retrieval code unchanged, only the transport
- **`tenant_id` with row-level security**, pushed into the vector query so tenants never share a candidate pool
- **This eval harness in CI**, gating changes to chunk size, `k`, or prompts on measured recall rather than intuition
