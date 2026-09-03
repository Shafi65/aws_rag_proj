"""All database access. Every SQL statement in the project lives in this file.

Keeping SQL in one place means there is exactly one file to read when asking
"what does this system actually do to the database", and one place to change
if the schema moves.
"""

from dataclasses import dataclass

import psycopg

import config


def connect():
    return psycopg.connect(
        host=config.PGHOST,
        port=config.PGPORT,
        dbname=config.PGDATABASE,
        user=config.PGUSER,
        password=config.PGPASSWORD,
    )


def to_vector_literal(values: list[float]) -> str:
    """Format a Python list as pgvector's text representation: [1.0,2.0,3.0].

    This is why the project does not need pgvector's Python helper package.
    The column is already typed vector(1024), so Postgres parses this string
    into a vector on insert. One less dependency to install and explain.
    """
    return "[" + ",".join(str(v) for v in values) + "]"


def stored_hash(conn, s3_key: str) -> str | None:
    """The content hash we recorded for this key, or None if never ingested."""
    with conn.cursor() as cur:
        cur.execute("SELECT content_sha256 FROM documents WHERE s3_key = %s", (s3_key,))
        row = cur.fetchone()
        return row[0] if row else None


def replace_document(
    conn,
    s3_key: str,
    filename: str,
    content_sha256: str,
    chunks,
    vectors: list[list[float]],
    embedding_model: str,
) -> int:
    """Write one document and all of its chunks. Returns the document id.

    DELETE then INSERT, not UPDATE. This matters: ON DELETE CASCADE only fires
    on an actual DELETE. Updating the documents row in place would leave the
    previous version's chunks behind, and the old and new chunks would then
    compete in search results -- with no error to tell you it happened.

    The whole thing is one transaction. If embedding or insertion fails
    halfway, the rollback leaves the document exactly as it was rather than
    half-replaced.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM documents WHERE s3_key = %s", (s3_key,))

        cur.execute(
            """
            INSERT INTO documents (s3_key, filename, content_sha256)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            (s3_key, filename, content_sha256),
        )
        document_id = cur.fetchone()[0]

        cur.executemany(
            """
            INSERT INTO chunks (
                document_id, chunk_index, content,
                page_start, page_end, embedding, embedding_model
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    document_id,
                    chunk.index,
                    chunk.text,
                    chunk.page_start,
                    chunk.page_end,
                    to_vector_literal(vector),
                    embedding_model,
                )
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
        )

    conn.commit()
    return document_id


@dataclass
class Hit:
    """One retrieved chunk, with everything a citation needs."""

    chunk_id: int
    filename: str
    chunk_index: int
    page_start: int
    page_end: int
    content: str
    distance: float  # cosine distance: 0 = identical, 2 = opposite
    rerank_score: float | None = None  # set by stage two; higher is better

    def citation(self) -> str:
        pages = (
            f"p. {self.page_start}"
            if self.page_start == self.page_end
            else f"pp. {self.page_start}-{self.page_end}"
        )
        return f"{self.filename}, {pages}"


def search(conn, query_vector: list[float], limit: int, embedding_model: str) -> list[Hit]:
    """Nearest chunks to the query vector, closest first.

    The <=> operator is cosine distance, and it must match the vector_cosine_ops
    the HNSW index was built with -- use <-> here instead and Postgres silently
    ignores the index and scans every row. Correct answers, terrible latency,
    no warning.

    ORDER BY <=> ... LIMIT is the shape the index recognises. The query vector
    appears twice because it is needed in both the SELECT list (to report the
    distance) and the ORDER BY.

    The embedding_model filter is the guard against mixing vector spaces: a
    query embedded by one model must only be compared against chunks embedded
    by the same one. Today every row matches, so it costs nothing; the day a
    second model appears in the table it is the difference between correct
    results and confidently wrong ones.
    """
    literal = to_vector_literal(query_vector)
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id, d.filename, c.chunk_index, c.page_start, c.page_end,
                   c.content, c.embedding <=> %(q)s::vector AS distance
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.embedding_model = %(model)s
            ORDER BY c.embedding <=> %(q)s::vector
            LIMIT %(k)s
            """,
            {"q": literal, "model": embedding_model, "k": limit},
        )
        return [Hit(*row) for row in cur.fetchall()]


def corpus_stats(conn) -> dict:
    """Row counts, for confirming an ingestion run did what we expected."""
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM documents")
        documents = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM chunks")
        chunks = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT embedding_model) FROM chunks")
        models = cur.fetchone()[0]
    return {"documents": documents, "chunks": chunks, "embedding_models": models}
