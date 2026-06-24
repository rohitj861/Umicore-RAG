# ingest.py
# from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv, get_key

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


def ingest_pdf_to_chroma(
    pdf_path: str = r"C:\Rohit Joshi\employee_handbook.pdf",
    persist_dir: str = "chroma_db",
    collection_name: str = "employee-handbook",
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
    loader = PyPDFLoader(str(pdf_file))
    docs = loader.load()  # one Document per page (includes metadata like page number)
    print(f"   Loaded {len(docs)} pages")

    print("2) Splitting into chunks (for better retrieval)...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,  # keeps start position for traceability
    )
    chunks = splitter.split_documents(docs)
    print(f"   Created {len(chunks)} chunks")

    print("3) Creating embeddings and building vector store...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(persist_path),
        collection_name=collection_name,
    )

    # Some versions auto-persist; calling persist() is still fine in many setups.
    if hasattr(vectordb, "persist"):
        vectordb.persist()

    print(
        f"   ✅ Vector store ready at ./{persist_dir} (collection: {collection_name})"
    )

    # (Optional) sanity check
    print("4) Sanity check (one match)...")
    sample = vectordb.similarity_search("What are the office timings?", k=1)
    if sample:
        preview = sample[0].page_content.replace("\n", " ")[:200]
        print("   Text:", preview, "...")
        print("   Meta:", sample[0].metadata)
    else:
        print(
            "   No results returned. (This can happen if the PDF is empty or unreadable.)"
        )


def main() -> None:
    ingest_pdf_to_chroma(
        pdf_path=r"C:\Rohit Joshi\employee_handbook.pdf",
        persist_dir="chroma_db",
        collection_name="employee-handbook",
    )


if __name__ == "__main__":
    main()
