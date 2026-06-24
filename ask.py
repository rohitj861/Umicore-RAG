# ask.py
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain


messages = [
    (
        "system",
        "You are a helpful assistant. Answer ONLY using the information in the context. "
        "you can look at the context and give me answers if they are already present in context",
    ),
    (
        "human",
        "Question: {input}\n\n"
        "Context:\n{context}\n\n"
        "Answer concisely in bullet points where helpful.",
    ),
]


def build_rag(
    persist_dir: str = "chroma_db",
    collection: str = "employee-handbook",
    k: int = 4,
):
    # Load env vars from .env (OPENAI_API_KEY)
    load_dotenv()

    # 1) Use same embeddings model as ingest
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2) Reconnect to persisted Chroma DB
    vectordb = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
        collection_name=collection,
    )

    # 3) Retriever: get top-k relevant chunks
    retriever = vectordb.as_retriever(search_kwargs={"k": k})

    # 4) LLM that will answer using retrieved context
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 5) Prompt: instruct model to only use context
    print(messages)
    prompt = ChatPromptTemplate.from_messages(messages)

    # 6) Chains: stuff docs into prompt -> LLM, and connect retrieval
    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    return rag_chain


def ask(question: str) -> None:
    rag = build_rag()
    response = rag.invoke({"input": question})

    answer = response.get("answer", "")
    contexts = response.get("context", [])

    print("\nQUESTION:")
    print(question)

    messages.append(
        (
            "human",
            f"{question}",
        ),
    )

    print("\nANSWER:")
    print("| ASSISTANT | -> ", end="")
    print(answer)

    messages.append(
        (
            "ai",
            f"{answer}",
        ),
    )
    print()

    # Show sources (file + page)
    if contexts:
        print("\nSOURCES:")
        for i, doc in enumerate(contexts, start=1):
            page = doc.metadata.get("page", "N/A")
            src = doc.metadata.get("source", "N/A")
            print(f"  ({i}) {os.path.basename(src)} - page {page}")


if __name__ == "__main__":
    # Demo questions:

    while True:
        question = input("| HUMAN | -> ")

        if "exit" in question:
            print("Exited from Chat")
            break

        print()
        ask(question)

    # ask("What are the official office timings and lunch break?")
    # ask("How many paid leaves do we get per year and how carry-forward works?")
    # ask("How many Work From Home (WFH) days can an employee take per month?")
    # ask("Whom to contact during emergencies?")
