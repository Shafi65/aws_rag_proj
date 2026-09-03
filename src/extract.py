"""PDF bytes -> one text string, plus a record of where each page began.

pypdf is not OCR. A PDF normally stores text as drawing instructions ("put
this character, in this font, at these coordinates"), and pypdf reconstructs a
string from them. A scanned PDF contains an image rather than text
instructions, so extraction returns an empty string -- that case needs an OCR
stage (Textract, Tesseract) in front of this pipeline.

Why one joined string instead of chunking each page separately: a sentence
that starts near the bottom of page 4 and finishes on page 5 would become two
truncated fragments, and neither fragment answers a question properly. So we
join the pages and separately remember the character offset each page began
at, which is what lets a chunk spanning the boundary be cited as "pp. 4-5".
"""

import io

from pypdf import PdfReader

# Inserted between pages. Two newlines because it also reads as a paragraph
# break to the chunker, which prefers to split there.
PAGE_SEPARATOR = "\n\n"


def extract_pages(pdf_bytes: bytes) -> tuple[str, list[int]]:
    """Return (full_text, page_starts).

    page_starts[i] is the character offset in full_text where page i+1 begins,
    so the list is 0-indexed but describes 1-indexed page numbers. It is
    ascending, which is what lets chunk.py map a character range back to pages
    with a binary search.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))

    pages: list[str] = []
    page_starts: list[int] = []
    cursor = 0

    for page in reader.pages:
        # extract_text() returns None on a page with no text layer (blank page,
        # or a scanned image). Treat that as empty rather than crashing -- one
        # unreadable page should not fail a 200-page document.
        text = (page.extract_text() or "").strip()

        page_starts.append(cursor)
        pages.append(text)
        cursor += len(text) + len(PAGE_SEPARATOR)

    return PAGE_SEPARATOR.join(pages), page_starts


def page_count(pdf_bytes: bytes) -> int:
    return len(PdfReader(io.BytesIO(pdf_bytes)).pages)
