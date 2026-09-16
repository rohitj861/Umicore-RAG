# ask.py
"""Ask any question about the ingested PDF (Umicore Annual Report 2025).

Answers are grounded strictly in the chunks stored by ingest.py. Those chunks
are found by two searches at once - dense vector similarity over Chroma and
BM25 keyword matching - whose rankings are combined by reciprocal rank fusion;
retrieval.py holds that machinery. If the PDF does not contain the answer, the
bot replies with "I don't know about this." instead of guessing.

Usage:
    python ask.py                  # interactive chat
    python ask.py "your question"  # one-shot question
    streamlit run app.py           # browser UI (see app.py)
"""

import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from retrieval import HybridRetriever

PERSIST_DIR = "chroma_db"
COLLECTION = "umicore-annual-report"
EMBED_MODEL = "text-embedding-3-small"  # must match ingest.py
CHAT_MODEL = "gpt-4o-mini"
# Depth is safe to run this deep only because ingest.py carries each table's
# column header onto the rows beneath it. On a store built without that, k=10
# measurably produced wrong figures - more context meant more lookalike tables
# to misread - and k=6 was the safest setting available. With headers attached,
# k=10 answered every graded case correctly and k=6 missed facts that sit just
# outside it.
#
# Both numbers were measured against fixed-size chunks and a vector-only
# search. Semantic chunks are larger and vary in size, and fusion changes which
# chunks arrive rather than how many, so the pair is worth re-measuring - see
# README. TOP_K now lives in retrieval.py, where it applies to each retriever.
MAX_SUBQUERIES = 4  # search queries per question: two forms x up to two topics
MAX_CONTEXT_CHUNKS = 16  # total chunks handed to the LLM
MAX_HISTORY_TURNS = 6  # remember the last N question/answer pairs
UNKNOWN_ANSWER = "I don't know about this."

# Words that end the chat rather than get answered. Without this the bot would
# search the report for "exit" and honestly report that it isn't in there.
EXIT_COMMANDS = {"exit", "quit", "q", ":q", "bye", "goodbye", "stop", "close"}
EXIT_MESSAGE = "Exited from Chat"

# Substrings that mark a key as never having been filled in. Checked instead of
# requiring an "sk-" prefix: rejecting an unfamiliar but valid key format would
# break a working setup, which is worse than the vague error this replaces.
PLACEHOLDER_HINTS = ("your-openai-api-key", "your_key_here", "api-key-here", "xxx")

# Cheap signal that a question may ask for more than one thing. Deliberately
# over-eager: a false positive costs one small LLM call, a false negative can
# cost a correct answer.
MULTIPART_HINT = re.compile(r"\b(and|also|as well as|plus|besides|versus|vs)\b", re.I)

# Words this report uses in its own way, where a question phrased in ordinary
# business English searches for the wrong thing. "Revenue" is the one measured
# so far: the report's revenue row is labelled "Revenues (excluding metal)" and
# is about a fifth of turnover, and a search for "total revenue in 2025" ranks
# that row 40th - it never reaches the context, so the model answered with
# turnover instead. Rewritten into the report's own words the same row ranks
# 10th-12th in both retrievers.
#
# Matching one of these is enough on its own to pay for the rewrite call, which
# is otherwise skipped for a simple first question. Add a term here only with a
# measurement behind it; the rewriter is told what each one means.
HOUSE_TERMS = re.compile(r"\brevenues?\b", re.I)

# Removes list markers the rewriter may prefix to each query ("1. ", "- ").
# Matching the marker shape matters: lstrip() over a digit set also eats a
# query that legitimately begins with a number, turning "2024 turnover" into
# "turnover" and silently dropping the year the follow-up was about.
LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s+")

# How much of each past message the rewriter sees. Enough to resolve "that
# year" or "the same segment"; short enough that a long answer can't crowd out
# the instruction or be copied back wholesale.
HISTORY_EXCERPT = 400

SYSTEM_PROMPT = (
    "You answer questions about the Umicore Annual Report 2025 using ONLY the "
    "context extracted from that PDF.\n\n"
    # Stated before the numbered rules, and repeated in rule 9, because as a
    # sub-bullet of rule 9 alone it was not applied. The note that discusses a
    # figure at length outranks the statement that reports it in every search,
    # so the note tends to arrive first and more often, and answering from
    # whatever came first is exactly the habit this has to override. Asked for
    # 2024 profit before income tax, the model took note F13's 1,375,542 -
    # consolidated companies only - over the income statement's 1,424,122,
    # even with the income statement sitting in the context.
    "READ THIS FIRST. Some lines appear twice in the context: once in a "
    "PRIMARY STATEMENT - a chunk headed 'Consolidated income statement', "
    "'Consolidated balance sheet' or 'Consolidated statement of cash flows' - "
    "and again in a NOTE, headed by F and a number ('F13 Income taxes') or by "
    "'RELATIONSHIP BETWEEN'. The note's version is a differently-scoped "
    "subtotal, not the same number. Before answering with any figure, scan the "
    "whole context for a primary statement carrying that line; if one is "
    "there, use ITS figure and say which statement it came from. Never answer "
    "from the note because it appeared first, or more often.\n\n"
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
    "   BRACKETS MEAN NEGATIVE, AND CONVERT EXACTLY AS A POSITIVE DOES. "
    "'(1,424,122)' under 'Thousands of EUR' is a LOSS of € 1.42 billion, and "
    "must be written 'a loss of € 1.42 billion (1,424,122 thousand EUR)' or "
    "'€ -1.42 billion'. Writing '€ (1,424,122) million' is WRONG twice over: "
    "it keeps the raw thousands and labels them millions. Carry the sign into "
    "words - say 'loss' or use a minus sign - and never leave brackets round "
    "an unconverted figure.\n"
    "7. TABLE HEADER LINES. A chunk may begin with '[page N table header: "
    "...]'. That is the units and column layout of the table the rows below it "
    "came from, restored because the extraction separated it from those rows. "
    "Treat it as the header for those rows and nothing else.\n"
    "8. COLUMNS AND YEARS. Map a figure to a year using the header before "
    "reporting it.\n"
    "   THE KEY FIGURES TABLES ARE HALF-YEAR THEN FULL-YEAR. A header reading "
    "'Key figures H2 H2' over '(in million €) 2024 2025 2024 2025' - used by "
    "'Group key figures' and by every business group's table - has FOUR "
    "columns in this order: H2 2024, H2 2025, FULL-YEAR 2024, FULL-YEAR 2025. "
    "The last two are the full year. So the row 'EBITDA 244 781 (1,025) 1,212' "
    "gives full-year 2024 = (1,025) and full-year 2025 = 1,212; 244 and 781 "
    "are half-year figures. Unless the question says 'H2', 'second half' or "
    "'half year', it is asking for the FULL YEAR - the third and fourth "
    "figures, not the first and second.\n"
    "   A full-year figure in brackets is a loss, and it is still the answer. "
    "Never take an earlier, positive-looking column because the full-year "
    "figure is negative.\n"
    "   Where the header names years and then repeats sub-labels, "
    "the figures in a row are grouped in the same order: with header "
    "'2024 2025 | Total Adjusted Adjustments Total Adjusted Adjustments', the "
    "row 'Turnover 14,853,681 14,859,584 (5,903) 19,374,073 18,849,795 "
    "524,279' gives 2024 Total = 14,853,681 and 2025 Total = 19,374,073 - the "
    "first three figures are 2024, the next three 2025.\n"
    "   Prefer a plain statement table over an adjustments or reconciliation "
    "table when both are present, and say which you used.\n"
    "   If you cannot tell which column belongs to the year asked, say the "
    "figure is ambiguous and name the candidates. NEVER pick a column because "
    "it looks plausible - a figure reported against the wrong year is worse "
    "than no figure.\n"
    "9. SCOPE: WHOSE FIGURE IS IT? The report states the same line at "
    "several scopes, with different figures. Group adjusted EBITDA for 2025 "
    "is 847; the Catalysis figure on the same row label is 450, and "
    "Recycling's is 371. Reporting one as the other is a large error.\n"
    "   The scopes, and how to recognise each from the table header:\n"
    "   - the GROUP total - 'Group key figures', and the consolidated "
    "statements ('Consolidated income statement', 'Consolidated balance "
    "sheet');\n"
    "   - one business group - 'Battery Materials Solutions key figures', "
    "'Catalysis key figures', 'Recycling key figures', 'Specialty Materials "
    "key figures';\n"
    "   - 'of consolidated companies', which EXCLUDES associates and joint "
    "ventures and so differs from the Group figure;\n"
    "   - the parent company Umicore SA's own statutory accounts.\n"
    "   Rules:\n"
    "   - If the question names a scope, answer at that scope.\n"
    "   - If it does not name one, give the GROUP figure and say it is the "
    "Group figure.\n"
    "   - ALWAYS name the scope of any figure you report - 'Group adjusted "
    "EBITDA', 'Catalysis turnover', 'profit before income tax of "
    "consolidated companies'. A bare figure with no scope is not an "
    "acceptable answer for any line that appears at more than one scope.\n"
    "   - If the context holds only a narrower scope, give that figure, say "
    "plainly which scope it is, and say the Group figure is not in the "
    "context. NEVER pass a business group's or a consolidated-companies "
    "figure off as the Group total.\n"
    "   - THE PRIMARY STATEMENTS OUTRANK THE NOTES. When the same line "
    "appears both in a primary statement ('Consolidated income statement', "
    "'Consolidated balance sheet', 'Consolidated statement of cash flows') "
    "and in a note - a header starting with F and a number, such as 'F13 "
    "Income taxes' - or under a heading containing 'RELATIONSHIP BETWEEN' or "
    "'reconciliation', report the PRIMARY STATEMENT's figure and name that "
    "statement. The note's variant is usually a differently-scoped subtotal "
    "on the way to it. For 2025, the consolidated income statement gives "
    "profit before income tax of 771,739; note F13 gives 845,345 for "
    "consolidated companies only. The first is the answer to an unscoped "
    "question; mention the second only if you explain what it is.\n"
    "10. REVENUE AND TURNOVER ARE DIFFERENT FIGURES. In this report:\n"
    "   - 'Turnover' is the total of outgoing sales invoices and INCLUDES the "
    "value of the purchased metals;\n"
    "   - the revenue line is labelled 'Revenues (excluding metal)' - the "
    "same sales minus the value of the purchased metals.\n"
    "   They differ by roughly a factor of five, so giving one when asked for "
    "the other is a large error, not a rounding difference.\n"
    "   'Revenues (excluding metal)' IS this report's revenue figure. When "
    "the question asks about revenue - including 'total revenue' or just "
    "'revenue' - answer with that row, note that it excludes metal, and give "
    "the group total rather than one business group's. Do NOT reply that the "
    "report does not cover revenue when that row is in the context; it is "
    "the answer. Equally, never offer turnover as a stand-in for revenue, or "
    "revenue as a stand-in for turnover - if the one asked for is genuinely "
    "absent, say so without substituting the other.\n"
    "11. Cite the page number(s) you used, e.g. (page 12).\n"
    "12. Be concise; use bullet points when listing several facts."
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
            "You rewrite the user's latest message into search queries for a "
            "document database. You never answer it. Rules:\n"
            "- Each query must stand alone: resolve pronouns and references "
            "using the conversation.\n"
            "- For each distinct thing asked, output TWO lines: a terse "
            "keyword phrase, then the same thing as a full standalone "
            'question. For "and in 2024?" following a question about adjusted '
            'EBITDA, that is:\n'
            "    2024 adjusted EBITDA\n"
            "    What was the adjusted EBITDA in 2024?\n"
            "  Both are needed: the terse phrase matches figures in table "
            "rows, the full question matches figures stated in prose. Emitting "
            "only one form loses whichever the answer happens to live in.\n"
            f"- At most {MAX_SUBQUERIES} lines. If more than two distinct "
            "things are asked, give each one a terse phrase instead.\n"
            "- Keep the user's wording, but add the report's where they "
            "differ. This report calls its revenue line 'Revenues (excluding "
            "metal)', a different and much smaller figure than 'turnover'. So "
            'for "total revenue in 2025" emit BOTH the user\'s phrasing and '
            "the report's:\n"
            "    revenues excluding metal 2025\n"
            "    What were the revenues excluding metal in 2025?\n"
            "  Never swap turnover in for revenue or the other way round - "
            "they are different figures, and searching for the wrong one "
            "returns the wrong answer.\n"
            "- Never state a fact, figure or page number, even if you believe "
            "you know it. A query containing an answer you invented sends the "
            "search to the wrong part of the document.\n"
            "- Output only the queries: no numbering, bullets or commentary.",
        ),
        # The conversation goes in as *text*, not as replayed messages. With a
        # MessagesPlaceholder here the model saw its own earlier answers in the
        # assistant role, matched that pattern and answered the follow-up
        # instead of rewriting it - putting a figure it had invented into the
        # search query. As quoted text there is no turn-taking pattern to
        # continue, and the instruction stays next to the input.
        (
            "human",
            "Conversation so far, for resolving references only:\n"
            "-----\n"
            "{history}\n"
            "-----\n\n"
            "Latest message: {input}\n\n"
            "Search queries:",
        ),
    ]
)


class SetupError(RuntimeError):
    """Environment isn't ready to answer questions (no key, no vector store).

    Raised rather than sys.exit-ed so a caller with a UI - app.py - can render
    the message. The CLI turns it back into an exit in main().
    """


def _page_label(doc: Document) -> str:
    """Human-friendly 1-based page number from the metadata ingest.py writes."""
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


def _format_history(messages: list) -> str:
    """The conversation as plain text for the query rewriter.

    Each message is truncated: the rewriter only needs enough to resolve a
    reference, and a full financial answer is mostly figures it must not reuse.
    """
    if not messages:
        return "(no earlier messages)"

    speakers = {"human": "User", "ai": "Assistant"}
    lines = []
    for message in messages:
        content = str(message.content).strip().replace("\n", " ")
        if len(content) > HISTORY_EXCERPT:
            content = content[:HISTORY_EXCERPT] + " [...]"
        lines.append(f"{speakers.get(message.type, message.type)}: {content}")
    return "\n".join(lines)


def is_exit_command(text: str) -> bool:
    """Whether the user is asking to leave rather than asking a question."""
    return text.strip().rstrip("!.").lower() in EXIT_COMMANDS


def _is_unknown(answer: str) -> bool:
    """True only when the whole answer is the fallback.

    Deliberately not a substring test: a partial answer may mention what the
    report doesn't cover while still citing real, useful sources.
    """
    cleaned = answer.strip().lstrip("-*• ").strip().strip('"').rstrip(".").lower()
    return cleaned == UNKNOWN_ANSWER.rstrip(".").lower()


def require_api_key() -> None:
    """Load .env and fail early if there is still no usable key.

    Called by every entry point that builds an OpenAI client - embeddings as
    well as chat - because the client raises a much vaguer error of its own if
    the key is missing.

    Only obvious non-keys are caught here. A wrong-but-plausible key can only
    be judged by OpenAI, and nothing in setup calls the API, so that case
    surfaces at the first question - see explain_api_error.
    """
    load_dotenv()
    key = (os.getenv("OPENAI_API_KEY") or "").strip()

    if not key:
        raise SetupError(
            "OPENAI_API_KEY not found.\n"
            "Copy .env.example to .env and put your key in it."
        )

    if any(hint in key.lower() for hint in PLACEHOLDER_HINTS):
        raise SetupError(
            f"OPENAI_API_KEY is still the placeholder ('{key[:28]}...').\n"
            "Replace it in .env with your real key."
        )


def explain_api_error(exc: Exception) -> str:
    """Turn an OpenAI client error into something the user can act on.

    Matched on class name rather than by importing openai: the SDK is only a
    transitive dependency here, and its exception paths have moved between
    majors. An unrecognised error is passed through verbatim.
    """
    name = type(exc).__name__
    detail = str(exc)

    if name == "AuthenticationError" or "invalid_api_key" in detail:
        return (
            "OpenAI rejected the API key in .env. Check it is complete, "
            "current, and has no quotes or trailing spaces around it."
        )
    if name == "PermissionDeniedError":
        return "That key is not permitted to use this model."
    if name == "RateLimitError" or "insufficient_quota" in detail:
        return (
            "OpenAI refused the request - rate limit or exhausted quota. "
            "Check the billing and limits on your OpenAI account."
        )
    if name in {"APIConnectionError", "APITimeoutError"}:
        return "Could not reach OpenAI. Check your network connection."
    return f"Could not answer that: {exc}"


def open_vectorstore() -> Chroma:
    """Reconnect to the persisted Chroma DB, failing with a clear message."""
    require_api_key()  # OpenAIEmbeddings below needs it

    if not Path(PERSIST_DIR).is_dir():
        raise SetupError(
            f"Vector store './{PERSIST_DIR}' not found.\n"
            "Build it first with:  python ingest.py"
        )

    vectordb = Chroma(
        embedding_function=OpenAIEmbeddings(model=EMBED_MODEL),
        persist_directory=PERSIST_DIR,
        collection_name=COLLECTION,
    )

    if not vectordb.get(limit=1).get("ids"):
        raise SetupError(
            f"Collection '{COLLECTION}' in './{PERSIST_DIR}' is empty.\n"
            "Re-run:  python ingest.py"
        )
    return vectordb


def open_retriever() -> HybridRetriever:
    """Open the store and build the keyword index that searches beside it.

    Both halves of the hybrid come from the one Chroma collection, so there is
    nothing to keep in sync and no second artefact to ship: re-running
    ingest.py changes what BM25 indexes as well, automatically. Building that
    index reads every chunk's text out of the store, which takes a second or
    two at start-up - hence one retriever per process, held by the caller.
    """
    return HybridRetriever(open_vectorstore())


class PdfChatbot:
    """Retrieval-augmented chatbot over a single ingested PDF."""

    def __init__(
        self, retriever: HybridRetriever | None = None, use_bm25: bool = True
    ):
        """Pass an already-open `retriever` to skip rebuilding the indexes.

        app.py opens the store and builds the BM25 index once per server and
        hands the same retriever to every browser session; the CLI leaves it
        None and opens its own.

        `use_bm25` lives on the bot, not on the retriever, because the bot is
        already per-session while the retriever is shared - so one visitor
        switching to vector-only search cannot change what another visitor
        gets. Flip it between questions at any time; it affects the next
        search only, and nothing about the conversation so far.
        """
        require_api_key()

        self.retriever = retriever if retriever is not None else open_retriever()
        self.use_bm25 = use_bm25
        self.llm = ChatOpenAI(model=CHAT_MODEL, temperature=0)
        self.chat_history: list = []

    def reset(self) -> None:
        """Forget the conversation so far."""
        self.chat_history.clear()

    def _needs_rewrite(self, question: str) -> bool:
        """Whether the extra query-rewrite LLM call is worth making.

        A first, single-topic question is already its own best search query,
        so we skip the call. We only pay for it when there is history to
        resolve, the question plausibly asks for more than one thing, or it
        uses a word this report defines its own way - where the question is
        precisely NOT its own best search query, because the wording the user
        reaches for is not the wording the report indexes under.
        """
        if self.chat_history:
            return True
        return (
            bool(MULTIPART_HINT.search(question))
            or bool(HOUSE_TERMS.search(question))
            or question.count("?") > 1
        )

    def _search_queries(self, question: str) -> list[str]:
        """One standalone search query per distinct thing the user asked."""
        if not self._needs_rewrite(question):
            return [question]

        response = (QUERY_PROMPT | self.llm).invoke(
            {"history": _format_history(self.chat_history), "input": question}
        )
        queries = [
            LIST_MARKER.sub("", line).strip()
            for line in response.content.splitlines()
            if line.strip()
        ]
        return [q for q in queries if q][:MAX_SUBQUERIES] or [question]

    def _retrieve(self, queries: list[str]) -> list[Document]:
        """Search every query with both retrievers, and fuse the rankings.

        Each query is run twice - once as an embedding against Chroma, once as
        keywords against BM25 - and every ranking that comes back is merged by
        reciprocal rank fusion, which is where the order the model finally sees
        is decided. retrieval.py explains why ranks rather than scores are what
        get added up, and how the same arithmetic stops one sub-question of a
        multi-part question filling every slot.

        With `use_bm25` off the keyword half is skipped and this is a plain
        vector search - the behaviour this project shipped before fusion.
        """
        return self.retriever.search(
            queries, limit=MAX_CONTEXT_CHUNKS, use_bm25=self.use_bm25
        )

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
    # Answers quote the report verbatim, so they carry euro signs and whatever
    # else the PDF uses. A redirected stdout is cp1252 on Windows and would
    # raise UnicodeEncodeError on the first character it cannot map.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    try:
        bot = PdfChatbot()
    except SetupError as exc:
        sys.exit(str(exc))

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
            print(f"\n{EXIT_MESSAGE}")
            break

        if not question:
            continue
        if is_exit_command(question):
            print(EXIT_MESSAGE)
            break

        try:
            print_answer(*bot.ask(question))
        except Exception as exc:  # keep the chat alive on API/network hiccups
            print(f"\n[error] {explain_api_error(exc)}\n")


if __name__ == "__main__":
    main()
