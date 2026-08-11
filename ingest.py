# ingest.py
# from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# A statement page prints its units and year columns once, at the top of the
# table, then many rows beneath. Splitting at 800 characters puts most of those
# rows in chunks that no longer contain the header: on page 91 the header sits
# at offset ~1100 and the Turnover row at ~1874, so the row-bearing chunk
# arrives with six bare numbers and nothing to say which year each belongs to.
UNITS_LINE = re.compile(r"(?:Thousands|Millions|Billions) of EUR", re.I)

# Lines above the units line that are worth carrying with it: a bare year row
# ("2024 2025") and a short table title. Longer lines are body prose, not
# header, and would only dilute the chunk.
MAX_HEADER_LINE = 90
MAX_HEADER = 220

# The running header repeated at the top of every page. Excluded because it
# carries a stray year ("... Annual Report 2025 62") that reads like a column.
RUNNING_HEADER = re.compile(r"Umicore Annual Report\s+\d{4}", re.I)


def _is_header_line(line: str) -> bool:
    """Whether a line above the units line belongs to the header.

    Wanted: a bare year row ("2024 2025") or a short table title
    ("Adjustments included in the result"). Not wanted: body prose, which is
    what a trailing full stop reliably indicates on these pages.
    """
    return (
        bool(line)
        and len(line) <= MAX_HEADER_LINE
        and not line.endswith(".")
        and not RUNNING_HEADER.search(line)
    )


def table_headers(page_text: str) -> list[tuple[int, str]]:
    """Every table header on the page, as (character offset, header text).

    A page often carries several tables - page 91 has R&D expenditure above the
    adjustments reconciliation - with different column layouts. Returning all of
    them with their offsets lets each chunk be matched to the header that
    actually governs it. Taking merely the first would caption a six-column row
    with a two-column header, which is worse than no caption at all.
    """
    found = []
    offset = 0
    lines = page_text.splitlines()
    stripped = [line.strip() for line in lines]

    for i, line in enumerate(lines):
        if UNITS_LINE.search(stripped[i]):
            parts = [stripped[i]]
            for previous in reversed(stripped[max(0, i - 2) : i]):
                if _is_header_line(previous):
                    parts.insert(0, previous)
            # Trimmed from the front: the units line is last and must survive.
            found.append((offset, " | ".join(parts)[-MAX_HEADER:]))
        offset += len(line) + 1  # +1 for the newline splitlines() removed

    return found


def add_table_headers(
    chunks: list[Document], headers: dict[int, list[tuple[int, str]]]
) -> int:
    """Prefix each chunk with the table header that governs it.

    Applied after splitting, so a header reaches the chunks the split severed
    it from. It joins the embedded text deliberately: it makes a bare row of
    figures retrievable by the year it reports, not just by its row label.
    """
    tagged = 0
    for chunk in chunks:
        start = chunk.metadata.get("start_index", 0)

        # The governing header is the last one at or before this chunk's start.
        header = ""
        for offset, text in headers.get(chunk.metadata.get("page"), []):
            if offset > start:
                break
            header = text

        if not header or header in chunk.page_content:
            continue  # no table above this chunk, or it kept its own header

        label = chunk.metadata.get("page_label", "?")
        chunk.page_content = f"[page {label} table header: {header}]\n{chunk.page_content}"
        tagged += 1

    return tagged


def load_pdf_pages(pdf_file: Path) -> list[Document]:
    """One Document per page, read straight from pypdf.

    Replaces langchain_community's PyPDFLoader (that package is being sunset)
    while keeping the same metadata keys that ask.py reads: source, page
    (0-based), page_label and total_pages.
    """
    reader = PdfReader(str(pdf_file))
    total_pages = len(reader.pages)

    try:
        labels = reader.page_labels
    except Exception:  # not all PDFs declare a page-label tree
        labels = None

    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        if not text:
            continue  # image-only / blank page: nothing to embed

        pages.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_file),
                    "page": i,
                    "page_label": labels[i] if labels else str(i + 1),
                    "total_pages": total_pages,
                },
            )
        )
    return pages


def ingest_pdf_to_chroma(
    pdf_path: str = "Umicore Annual Report 2025.pdf",
    persist_dir: str = "chroma_db",
    collection_name: str = "umicore-annual-report",
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> None:
    # 1) Load environment variables (OPENAI_API_KEY)
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not found. Create a .env file with:\n"
            "OPENAI_API_KEY=your_key_here\n"
            "or set it in your OS environment variables."
        )

    # 2) Validate paths
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_file.resolve()}")

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    print("1) Loading PDF...")
    docs = load_pdf_pages(pdf_file)
    print(f"   Loaded {len(docs)} pages with extractable text")

    print("2) Splitting into chunks (for better retrieval)...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # keeps start position for traceability
    )
    chunks = splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks")

    headers = {doc.metadata["page"]: table_headers(doc.page_content) for doc in docs}
    tagged = add_table_headers(chunks, headers)
    pages_with_tables = sum(1 for found in headers.values() if found)
    print(
        f"   Carried table headers onto {tagged} chunks "
        f"from {pages_with_tables} pages with tables"
    )

    print("3) Creating embeddings and building vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Drop any previous run first - Chroma.from_documents appends, so
    # re-ingesting without this would store every chunk twice.
    Chroma(
        embedding_function=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    ).delete_collection()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    )

    print(
        f"   ✅ Vector store ready at ./{persist_dir} (collection: {collection_name})"
    )

    # (Optional) sanity check
    print("4) Sanity check (one match)...")
    sample = vectordb.similarity_search("What were the key financial results?", k=1)
    if sample:
        preview = sample[0].page_content.replace("\n", " ")[:200]
        print("   Text:", preview, "...")
        print("   Meta:", sample[0].metadata)
    else:
        print(
            "   No results returned. (This can happen if the PDF is empty or unreadable.)"
        )


def main() -> None:
    # The Windows console is UTF-8, but a redirected or piped stdout falls back
    # to cp1252, which cannot encode the checkmark below (or much of what a PDF
    # may contain). Without this, 'python ingest.py > log.txt' dies on the
    # success message after the store has already been built.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    ingest_pdf_to_chroma(
        pdf_path="Umicore Annual Report 2025.pdf",
        persist_dir="chroma_db",
        collection_name="umicore-annual-report",
    )


if __name__ == "__main__":
    main()
