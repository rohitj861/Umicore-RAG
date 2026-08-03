# ask.py
"""Ask any question about the ingested PDF (Umicore Annual Report 2025).

Answers are grounded strictly in the chunks stored in the Chroma vector store
built by ingest.py. If the PDF does not contain the answer, the bot replies
with "I don't know about this." instead of guessing.

Usage:
    python ask.py                  # interactive chat
    python ask.py "your question"  # one-shot question
"""

import os
import re
import sys
from itertools import zip_longest
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION = "umicore-annual-report"
EMBED_MODEL = "text-embedding-3-small"  # must match ingest.py
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 6  # chunks fetched per search query
MAX_SUBQUERIES = 3  # a multi-part question is split into at most this many searches
MAX_CONTEXT_CHUNKS = 10  # total chunks handed to the LLM
MAX_HISTORY_TURNS = 6  # remember the last N question/answer pairs
UNKNOWN_ANSWER = "I don't know about this."

# Cheap signal that a question may ask for more than one thing. Deliberately
# over-eager: a false positive costs one small LLM call, a false negative can
# cost a correct answer.
MULTIPART_HINT = re.compile(r"\b(and|also|as well as|plus|besides|versus|vs)\b", re.I)

SYSTEM_PROMPT = (
    "You answer questions about the Umicore Annual Report 2025 using ONLY the "
    "context extracted from that PDF.\n"
    "Rules:\n"
    "1. If the context answers NONE of the question, reply with exactly: "
    f'"{UNKNOWN_ANSWER}" and nothing else.\n'
    "2. Never use outside knowledge, never guess, and never fill gaps with "
    "plausible-sounding numbers.\n"
    "3. If only part of the question is covered, answer that part and write "
    '"The report does not cover ..." for the rest. Do not use the exact '
    f'sentence "{UNKNOWN_ANSWER}" in a partial answer - that sentence is '
    "reserved for questions the report cannot answer at all.\n"
    "4. Quote dates and names exactly as they appear in the context.\n"
    "5. UNITS. If the context already states a figure with its unit in prose "
    "(e.g. '€ 847 million'), quote it exactly as written and add nothing.\n"
    "6. UNITS IN TABLES. A bare figure in a table row is governed by a units "
    "header such as 'Thousands of EUR', usually several lines above the row. "
    "Find that header and convert the number before writing it:\n"
    "   - under 'Thousands of EUR': divide by 1,000 for € million, by "
    "1,000,000 for € billion;\n"
    "   - under 'Millions of EUR': divide by 1,000 for € billion.\n"
    "   Write the converted value, then the original in brackets. The row "
    "'Turnover 19,374,073' under 'Thousands of EUR' must be written as "
    "'€ 19.37 billion (19,374,073 thousand EUR)'.\n"
    "   NEVER write the raw digits straight after a euro sign: "
    "'€ 19,374,073' is WRONG because it understates the amount 1000-fold. "
    "Any euro amount you write must carry the word million or billion unless "
    "it is genuinely under a million.\n"
    "   If a table figure has no units header in the context, say its unit "
    "is unclear rather than assuming.\n"
    "7. Cite the page number(s) you used, e.g. (page 12).\n"
    "8. Be concise; use bullet points when listing several facts."
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        (
            "human",
            "Context from the PDF:\n"
            "-----\n"
            "{context}\n"
            "-----\n\n"
            "Question: {input}",
        ),
    ]
)

# Turns the user's message into standalone search queries. This does two jobs:
# resolves follow-ups like "and in 2024?" against the history, and splits a
# multi-part question so each part gets its own search (a single embedding for
# "who is the CEO and how much leave do staff get?" lands between both topics
# and retrieves neither well).
QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user's latest message as search queries for a "
            "document database. Rules:\n"
            "- Each query must stand alone: resolve pronouns and references "
            "using the chat history.\n"
            "- If the message asks several distinct things, output ONE query "
            f"per line, at most {MAX_SUBQUERIES}. Otherwise output a single line.\n"
            "- Keep the original wording where possible. Do NOT answer the "
            "question. Output only the queries, no numbering or bullets.",
        ),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _page_label(doc: Document) -> str:
    """Human-friendly 1-based page number from PyPDFLoader metadata."""
    page = doc.metadata.get("page")
    try:
        return str(int(page) + 1)
    except (TypeError, ValueError):
        return str(doc.metadata.get("page_label", "N/A"))


def _format_context(docs: list[Document]) -> str:
    blocks = []
    for i, doc in enumerate(docs, start=1):
        blocks.append(f"[chunk {i} | page {_page_label(doc)}]\n{doc.page_content}")
    return "\n\n".join(blocks)


def _is_unknown(answer: str) -> bool:
    """True only when the whole answer is the fallback.

    Deliberately not a substring test: a partial answer may mention what the
    report doesn't cover while still citing real, useful sources.
    """
    cleaned = answer.strip().lstrip("-*• ").strip().strip('"').rstrip(".").lower()
    return cleaned == UNKNOWN_ANSWER.rstrip(".").lower()


def _open_vectorstore() -> Chroma:
    """Reconnect to the persisted Chroma DB, failing with a clear message."""
    if not Path(PERSIST_DIR).is_dir():
        sys.exit(
            f"Vector store './{PERSIST_DIR}' not found.\n"
            "Build it first with:  python ingest.py"
        )

    vectordb = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )

    if not vectordb.get(limit=1).get("ids"):
        sys.exit(
            f"Collection '{COLLECTION}' in './{PERSIST_DIR}' is empty.\n"
            "Re-run:  python ingest.py"
        )
    return vectordb


class PdfChatbot:
    """Retrieval-augmented chatbot over a single ingested PDF."""

    def __init__(self, k: int = TOP_K):
        load_dotenv()
        if not os.getenv("OPENAI_API_KEY"):
            sys.exit(
                "OPENAI_API_KEY not found.\n"
                "Copy .env.example to .env and put your key in it."
            )

        self.k = k
        self.vectordb = _open_vectorstore()
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.chat_history: list = []

    def _needs_rewrite(self, question: str) -> bool:
        """Whether the extra query-rewrite LLM call is worth making.

        A first, single-topic question is already its own best search query,
        so we skip the call. We only pay for it when there is history to
        resolve or the question plausibly asks for more than one thing.
        """
        if self.chat_history:
            return True
        return bool(MULTIPART_HINT.search(question)) or question.count("?") > 1

    def _search_queries(self, question: str) -> list[str]:
        """One standalone search query per distinct thing the user asked."""
        if not self._needs_rewrite(question):
            return [question]

        response = (QUERY_PROMPT | self.llm).invoke(
            {"chat_history": self.chat_history, "input": question}
        )
        queries = [
            line.strip().lstrip("-*•0123456789. ").strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
        return [q for q in queries if q][:MAX_SUBQUERIES] or [question]

    def _retrieve(self, queries: list[str]) -> list[Document]:
        """Search each query, then interleave the hits.

        Interleaving (rather than concatenating) matters: with a flat cap, the
        first sub-question's hits would fill every slot and the second would
        get nothing - which is how a covered fact gets reported as missing.
        """
        per_query = [self.vectordb.similarity_search(q, k=self.k) for q in queries]

        seen, merged = set(), []
        for rank in zip_longest(*per_query):
            for doc in rank:
                if doc is None:
                    continue
                key = (
                    doc.metadata.get("source"),
                    doc.metadata.get("page"),
                    doc.metadata.get("start_index"),
                )
                if key in seen:
                    continue
                seen.add(key)
                merged.append(doc)
                if len(merged) >= MAX_CONTEXT_CHUNKS:
                    return merged
        return merged

    def _remember(self, question: str, answer: str) -> None:
        self.chat_history.extend([HumanMessage(question), AIMessage(answer)])
        del self.chat_history[: -2 * MAX_HISTORY_TURNS]

    def ask(self, question: str) -> tuple[str, list[Document]]:
        """Answer a question from the PDF. Returns (answer, source documents)."""
        docs = self._retrieve(self._search_queries(question))

        if not docs:
            # Nothing relevant in the PDF at all - do not even call the LLM.
            answer, sources = UNKNOWN_ANSWER, []
        else:
            response = (ANSWER_PROMPT | self.llm).invoke(
                {
                    "chat_history": self.chat_history,
                    "context": _format_context(docs),
                    "input": question,
                }
            )
            answer = response.content.strip() or UNKNOWN_ANSWER
            # Don't show sources for an answer that isn't in the PDF.
            sources = [] if _is_unknown(answer) else docs

        self._remember(question, answer)
        return answer, sources


def print_answer(answer: str, sources: list[Document]) -> None:
    print("\n| ASSISTANT | -> " + answer + "\n")

    if sources:
        print("SOURCES:")
        seen = set()
        for doc in sources:
            src = os.path.basename(doc.metadata.get("source", "N/A"))
            key = (src, _page_label(doc))
            if key in seen:
                continue
            seen.add(key)
            print(f"  - {src} (page {_page_label(doc)})")
        print()


def main() -> None:
    bot = PdfChatbot()

    # One-shot mode:  python ask.py "What were the 2025 revenues?"
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\n| HUMAN | -> {question}")
        print_answer(*bot.ask(question))
        return

    print("Ask anything about the Umicore Annual Report 2025.")
    print("Type 'exit' or press Ctrl+C to quit.\n")

    while True:
        try:
            question = input("| HUMAN | -> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExited from Chat")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q", ":q"}:
            print("Exited from Chat")
            break

        try:
            print_answer(*bot.ask(question))
        except Exception as exc:  # keep the chat alive on API/network hiccups
            print(f"\n[error] Could not answer that: {exc}\n")


if __name__ == "__main__":
    main()
