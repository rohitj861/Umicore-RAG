# ingest.py
# from __future__ import annotations

import os
import re
import shutil
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
UNITS_LINE = re.compile(
    r"(?:Thousands|Millions|Billions) of EUR"  # statement tables
    r"|\(in (?:million|thousand|billion)s? ?€\)"  # segment key figures
    r"|\bin %",
    re.I,
)

# A run of two or more years separated by nothing but whitespace is a column
# header: "2024 2025", "(in million €) 2024 2025 2024 2025", "Company
# performance 2021 2022 2023 2024 2025". Requiring whitespace-only separators
# is what keeps prose out - "between 2024 and 2025" does not match.
YEAR_COLUMNS = re.compile(r"\b(?:19|20)\d{2}(?:\s+(?:19|20)\d{2})+\b")

# Thousands-separated numbers mark a data row, not a header. Without this,
# "Baseline value (t) (baseline year) 791,816 (2019) 6,816,941 (2019)" reads as
# a year header because it mentions 2019 twice.
DATA_NUMBER = re.compile(r"\d{1,3},\d{3}")

# Used to tell a header band apart from a data row: years are fine in a header,
# three or more other figures are not.
NUMERIC_TOKEN = re.compile(r"-?\d[\d,.]*%?")
YEAR = re.compile(r"(?:19|20)\d{2}")

# Lines above the units line that are worth carrying with it: a bare year row
# ("2024 2025") and a short table title. Longer lines are body prose, not
# header, and would only dilute the chunk.
MAX_HEADER_LINE = 90
MAX_HEADER = 220

# The running header repeated at the top of every page. Excluded because it
# carries a stray year ("... Annual Report 2025 62") that reads like a column.
RUNNING_HEADER = re.compile(r"Umicore Annual Report\s+\d{4}", re.I)


def _is_header_line(line: str) -> bool:
    """Whether a line above the header line belongs to the header too.

    Wanted: a bare year row ("2024 2025"), a column band ("% INTEREST IN
    % INTEREST IN", "Key figures H2 H2") or a short table title. Not wanted:
    body prose - a trailing full stop marks that reliably here - and not a data
    row, which is what several non-year numbers mean. Without the numeric test,
    page 52 captions its table with the row "% change versus previous year
    7.5% 5.6% 10.2% -0.9% 4.8%".
    """
    if not line or len(line) > MAX_HEADER_LINE:
        return False
    if line.endswith(".") or RUNNING_HEADER.search(line):
        return False

    figures = [n for n in NUMERIC_TOKEN.findall(line) if not YEAR.fullmatch(n)]
    return len(figures) < 3


def _is_units_or_year_header(line: str) -> bool:
    """Whether this line is the header row of a table.

    Two ways to qualify: it names the units, or it is a run of year columns.
    The year route needs the data-row guard; the units route deliberately does
    not, because extraction sometimes merges a units header with its first row
    and that line is still the header.
    """
    if UNITS_LINE.search(line):
        return True
    return (
        bool(YEAR_COLUMNS.search(line))
        and not DATA_NUMBER.search(line)
        and len(line) <= 130
        and not line.endswith(".")
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
        if _is_units_or_year_header(stripped[i]):
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


def reset_store(persist_path: Path) -> None:
    """Delete the whole store directory so a rebuild starts from nothing.

    delete_collection() alone is not enough: it drops the collection but leaves
    its UUID-named segment folder on disk and leaves the freed sqlite pages
    unreclaimed. Three ingests had grown a 28 MB store to 51 MB, two thirds of
    the segment folders dead - and since the store is committed, every byte of
    that ships to anyone who clones or deploys it.

    Guarded on chroma.sqlite3 being present, so a mistyped persist_dir removes
    nothing that isn't a Chroma store.
    """
    if not persist_path.exists():
        return

    if not (persist_path / "chroma.sqlite3").exists():
        raise RuntimeError(
            f"'{persist_path}' exists but is not a Chroma store "
            "(no chroma.sqlite3). Refusing to delete it - check persist_dir."
        )

    shutil.rmtree(persist_path)
    print(f"   Removed the previous store at ./{persist_path}")


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
    reset_store(persist_path)
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
