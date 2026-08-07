# app.py
"""Browser UI for the Umicore Annual Report Q&A bot.

Same retrieval and answering as ask.py - this only puts a chat window around
it and makes the sources behind each answer inspectable, which is awkward to
do in a terminal.

Usage:
    streamlit run app.py
"""

import os

import streamlit as st

from ask import (
    CHAT_MODEL,
    EMBED_MODEL,
    EXIT_MESSAGE,
    MAX_CONTEXT_CHUNKS,
    TOP_K,
    PdfChatbot,
    SetupError,
    _page_label,
    is_exit_command,
    open_vectorstore,
)

PDF_NAME = "Umicore Annual Report 2025"

EXAMPLE_QUESTIONS = [
    "What was Umicore's adjusted EBITDA in 2025?",
    "Who is the CEO and who chairs the board?",
    "What is the turnover for the year?",
    "What are the main sustainability targets?",
]

st.set_page_config(
    page_title=f"{PDF_NAME} — Q&A",
    page_icon="📄",
    layout="centered",
)


@st.cache_resource(show_spinner="Opening the vector store...")
def get_vectorstore():
    """One Chroma handle per server process, shared by all browser sessions.

    Reopening it per rerun would re-read the store on every keystroke-driven
    script run, which is the single slowest thing this app does.
    """
    return open_vectorstore()


def get_bot() -> PdfChatbot:
    """The chatbot for this browser session.

    Per session, not cached globally: the bot carries conversation memory, so
    two users sharing one would resolve each other's follow-up questions.
    """
    if "bot" not in st.session_state:
        st.session_state.bot = PdfChatbot(vectordb=get_vectorstore())
    return st.session_state.bot


def render_sources(sources: list) -> None:
    """Page citations for one answer, with the retrieved text behind them."""
    if not sources:
        return

    pages, seen = [], set()
    for doc in sources:
        page = _page_label(doc)
        if page not in seen:
            seen.add(page)
            pages.append(page)

    with st.expander(f"Sources — pages {', '.join(pages)}"):
        st.caption(
            "The exact chunks the answer was written from. Page numbers in the "
            "answer text can drift between similar tables; these are reliable."
        )
        for doc in sources:
            src = os.path.basename(doc.metadata.get("source", "N/A"))
            st.markdown(f"**{src} — page {_page_label(doc)}**")
            st.text(doc.page_content)
            st.divider()


def start_new_chat() -> None:
    """Empty the transcript and the bot's memory, and reopen the input."""
    st.session_state.messages = []
    st.session_state.ended = False
    if "bot" in st.session_state:
        st.session_state.bot.reset()


def end_chat() -> None:
    """Close the chat in response to 'exit'.

    There is no process to quit in a browser, so ending means: stop taking
    questions, keep the transcript readable, and offer a fresh start.
    """
    st.session_state.ended = True
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": f"**{EXIT_MESSAGE}.** Start a new chat below to ask "
            "something else.",
        }
    )
    if "bot" in st.session_state:
        st.session_state.bot.reset()


def answer(question: str) -> None:
    """Run one question through the bot and append the exchange to the log."""
    st.session_state.messages.append({"role": "user", "content": question})

    if is_exit_command(question):
        end_chat()
        return

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the report..."):
            try:
                text, sources = get_bot().ask(question)
            except Exception as exc:  # API/network hiccup - keep the chat alive
                text, sources = f"Could not answer that: {exc}", []
        st.markdown(text)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": text, "sources": sources}
    )


# --- Sidebar -----------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.write(
        f"Answers come only from **{PDF_NAME}.pdf**. If the report doesn't "
        'contain the answer, the bot says *"I don\'t know about this."* '
        "rather than guessing."
    )

    st.subheader("Settings")
    st.caption(
        f"Chat model `{CHAT_MODEL}` · embeddings `{EMBED_MODEL}` · "
        f"{TOP_K} chunks per search, {MAX_CONTEXT_CHUNKS} max in context"
    )

    if st.button("Clear conversation", use_container_width=True):
        start_new_chat()
        st.rerun()

    st.caption(
        "Follow-ups work — the last 6 exchanges are remembered, so "
        '"and the year before that?" resolves correctly.'
    )
    st.caption("Type **exit** in the chat to end the conversation.")


# --- Main --------------------------------------------------------------------

st.title("📄 Umicore Annual Report 2025")
st.caption(
    "Ask anything about the report. Every answer cites its pages. "
    "Type **exit** when you're done."
)

try:
    get_bot()
except SetupError as exc:
    st.error(str(exc))
    st.info("Run the setup steps in README.md, then reload this page.")
    st.stop()

if "messages" not in st.session_state:
    start_new_chat()

# Replay the conversation so far (Streamlit reruns the script on every input).
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

# Starter questions, shown only on an empty chat.
if not st.session_state.messages:
    st.write("**Try one of these:**")
    for i, example in enumerate(EXAMPLE_QUESTIONS):
        if st.button(example, key=f"example_{i}", use_container_width=True):
            answer(example)
            st.rerun()

if st.session_state.ended:
    # The chat is closed: no more questions until the user asks for a new one.
    st.chat_input("Chat ended", disabled=True)
    if st.button("Start a new chat", type="primary", use_container_width=True):
        start_new_chat()
        st.rerun()
elif question := st.chat_input("Ask a question about the report..."):
    answer(question)
    if st.session_state.ended:
        # end_chat() only appended to the log - rerun to draw the closed state.
        st.rerun()
