-- RAG system schema.
-- Applied automatically the first time the Postgres container initialises
-- (see docker-compose.yml). To re-apply after editing: docker compose down -v && docker compose up -d

-- pgvector ships in the image but is not active until enabled per-database.
CREATE EXTENSION IF NOT EXISTS vector;


-- One row per source file.
CREATE TABLE IF NOT EXISTS documents (
    id             BIGSERIAL   PRIMARY KEY,

    -- Provenance: where this file came from. Unique so re-ingesting the same
    -- object updates one row rather than creating a duplicate.
    s3_key         TEXT        NOT NULL UNIQUE,

    -- Basename of the key. Denormalised on purpose: this is the string that
    -- appears in a citation, and we don't want to recompute it at display time.
    filename       TEXT        NOT NULL,

    -- SHA-256 of the raw bytes. Change detection: if this matches what we
    -- already stored, the file hasn't changed and we skip re-embedding it.
    content_sha256 TEXT        NOT NULL,

    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);


-- One row per retrievable chunk of text.
CREATE TABLE IF NOT EXISTS chunks (
    id              BIGSERIAL PRIMARY KEY,

    -- The link that makes citations possible. CASCADE means deleting a
    -- document deletes its chunks, so re-ingesting never leaves orphan
    -- vectors behind to pollute search results.
    document_id     BIGINT    NOT NULL REFERENCES documents(id) ON DELETE CASCADE,

    -- Position within the document, 0-based. Used for stable ordering, for
    -- debugging ("show me this doc's chunks in order"), and later for pulling
    -- neighbouring chunks when one chunk alone lacks context.
    chunk_index     INT       NOT NULL,

    -- The chunk text itself. This is what gets pasted into the prompt.
    content         TEXT      NOT NULL,

    -- Citation. Two columns because a chunk can begin near the bottom of one
    -- page and end on the next; a single column would force us to be wrong
    -- about one of them. NULL for formats without pages (.txt, .md).
    page_start      INT,
    page_end        INT,

    -- The 1024 numbers from Titan Text Embeddings v2. The dimension is part
    -- of the column type, so changing it is a migration, not a config change.
    embedding       vector(1024) NOT NULL,

    -- Which model produced the vector above. Correctness guard: vectors from
    -- two different models are coordinates in different spaces. Mixing them
    -- returns numbers that still sort but are meaningless. Nothing crashes --
    -- retrieval quality just silently degrades. We assert on this at query time.
    embedding_model TEXT      NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),

    UNIQUE (document_id, chunk_index)
);


-- Postgres does not index foreign keys automatically. Needed by the cascade
-- delete above and by any "all chunks from this document" query.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

-- Vector index. HNSW over IVFFlat because IVFFlat builds its clusters from
-- the rows present at build time and is useless on an empty table; HNSW
-- builds incrementally, which fits "start the container, then ingest".
--
-- vector_cosine_ops must match the <=> operator used in queries. Mismatch it
-- and Postgres silently sequential-scans instead of erroring.
--
-- Honest caveat: at a few thousand chunks the planner may choose a sequential
-- scan anyway, and be right to -- brute-force cosine over 5k vectors is fast.
-- The index is the correct choice at the scale this design targets, not a
-- performance win on a small demo corpus. We verify with EXPLAIN.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks USING hnsw (embedding vector_cosine_ops);
