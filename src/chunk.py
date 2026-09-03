"""Split a document's text into overlapping chunks, each tagged with its pages.

A chunk is the unit of retrieval: one chunk becomes one embedding vector and
one row in the `chunks` table. Size is a real tradeoff --

  small chunks -> precise, but a fact needing surrounding context gets orphaned
  large chunks -> more context, but the single vector has to represent several
                  different ideas at once and lands somewhere vague between
                  them, matching nothing sharply

Overlap exists so a fact stated across a boundary survives intact in at least
one chunk. It costs duplicated text: 200/1200 is roughly 17% more chunks, so
17% more embedding calls, storage, and index size.

Run directly to preview chunks from a local PDF:

    python src/chunk.py ~/rag-docs/wellarchitected-security-pillar.pdf
"""

import sys
from bisect import bisect_right
from dataclasses import dataclass

# How far back from the hard cut we are willing to look for a clean break.
SNAP_WINDOW = 250

# Tried in order of preference: paragraph break, then sentence end, then any
# line break. A chunk that begins mid-sentence embeds poorly, because the model
# is being asked to represent a fragment.
BREAK_MARKERS = ("\n\n", ". ", ".\n", "\n")


@dataclass
class Chunk:
    index: int  # position within the document, 0-based
    text: str
    char_start: int  # offsets into the document's full text, for auditing
    char_end: int
    page_start: int  # 1-based, for citations
    page_end: int


def _page_for_offset(page_starts: list[int], offset: int) -> int:
    """Which 1-based page does this character offset fall on?

    page_starts is ascending, so bisect_right gives the number of pages that
    begin at or before this offset -- which is exactly the 1-based page number.
    Binary search rather than a scan because this runs once per chunk boundary.
    """
    return bisect_right(page_starts, offset)


def _snap_to_break(text: str, hard_end: int, earliest: int) -> int:
    """Back up from hard_end to the nearest clean break, but not past earliest."""
    window = text[earliest:hard_end]
    for marker in BREAK_MARKERS:
        position = window.rfind(marker)
        if position != -1:
            return earliest + position + len(marker)
    return hard_end  # no break found; cut at the hard limit


def chunk_text(
    text: str,
    page_starts: list[int],
    size: int,
    overlap: int,
) -> list[Chunk]:
    """Slide a window of `size` characters through text, stepping `size - overlap`."""
    if overlap >= size:
        raise ValueError("CHUNK_OVERLAP_CHARS must be smaller than CHUNK_SIZE_CHARS")

    chunks: list[Chunk] = []
    start = 0
    length = len(text)

    while start < length:
        hard_end = min(start + size, length)

        if hard_end < length:
            end = _snap_to_break(text, hard_end, max(hard_end - SNAP_WINDOW, start + 1))
        else:
            end = hard_end  # last chunk: take whatever is left

        # Trim surrounding whitespace but keep the offsets exact, so char_start
        # and char_end still describe a verbatim span of the source text.
        raw = text[start:end]
        leading = len(raw) - len(raw.lstrip())
        body = raw.strip()

        if body:
            body_start = start + leading
            body_end = body_start + len(body)
            chunks.append(
                Chunk(
                    index=len(chunks),
                    text=body,
                    char_start=body_start,
                    char_end=body_end,
                    page_start=_page_for_offset(page_starts, body_start),
                    page_end=_page_for_offset(page_starts, body_end - 1),
                )
            )

        if end >= length:
            break
        # max(..., start + 1) guarantees forward progress even if snapping
        # returned a very short chunk -- otherwise this could loop forever.
        start = max(end - overlap, start + 1)

    return chunks


if __name__ == "__main__":
    from pathlib import Path

    import config
    from extract import extract_pages, page_count

    if len(sys.argv) != 2:
        sys.exit("usage: python src/chunk.py <path-to-pdf>")

    pdf_bytes = Path(sys.argv[1]).read_bytes()
    text, page_starts = extract_pages(pdf_bytes)
    chunks = chunk_text(
        text, page_starts, config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS
    )

    print(f"pages      : {page_count(pdf_bytes)}")
    print(f"characters : {len(text):,}")
    print(f"chunks     : {len(chunks)}")
    print(f"settings   : size={config.CHUNK_SIZE_CHARS} overlap={config.CHUNK_OVERLAP_CHARS}")

    for chunk in chunks[:3]:
        pages = (
            f"p. {chunk.page_start}"
            if chunk.page_start == chunk.page_end
            else f"pp. {chunk.page_start}-{chunk.page_end}"
        )
        print(f"\n--- chunk {chunk.index} | {pages} | chars {chunk.char_start}-{chunk.char_end} ---")
        print(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

    # Sanity check: consecutive chunks should share `overlap` characters.
    if len(chunks) >= 2:
        shared = chunks[0].char_end - chunks[1].char_start
        print(f"\noverlap between chunks 0 and 1: {shared} chars")
