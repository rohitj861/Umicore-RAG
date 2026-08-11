# Umicore Annual Report 2025 — Q&A (RAG)

Ask questions in plain English about `Umicore Annual Report 2025.pdf`.
Answers come only from the PDF. If the report doesn't contain the answer, the
bot replies **"I don't know about this."** rather than guessing.

```
| HUMAN | -> What was Umicore's adjusted EBITDA in 2025?

| ASSISTANT | -> Umicore's adjusted EBITDA in 2025 was € 847 million (page 18).

SOURCES:
  - Umicore Annual Report 2025.pdf (page 18)
```

## Setup (once)

```powershell
cd "C:\Umicore RAG\rag-umicore"

# 1. Virtual environment + dependencies
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. API key: create a file named .env containing one line
#    OPENAI_API_KEY=sk-...
Copy-Item .env.example .env
notepad .env

# 3. The report is not in this repo. Download it from Umicore and save it in
#    this folder as exactly: Umicore Annual Report 2025.pdf
#    (Any PDF works if you change pdf_path in ingest.py.)

# 4. Read the PDF into the vector database (~1-2 min, costs a few cents)
.\.venv\Scripts\python.exe ingest.py
```

## Asking questions

### Browser UI (easiest)

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Opens a chat window at <http://localhost:8501>. Each answer carries a
**Sources** panel you can expand to read the exact chunks of the report it was
written from — the quickest way to check an answer is really in the PDF.

Type **exit** (or `quit`, `bye`, `q`) to end the conversation; a *Start a new
chat* button appears. Closing the browser tab leaves the server running — stop
it with `Ctrl+C` in the terminal.

### Terminal

```powershell
# Interactive chat (type 'exit' to quit)
.\.venv\Scripts\python.exe ask.py

# One-shot
.\.venv\Scripts\python.exe ask.py "Who is the CEO?"
```

Follow-up questions work — the bot remembers the last 6 exchanges, so
"and what about the year before that?" resolves correctly.

You only re-run `ingest.py` if the PDF changes. It wipes and rebuilds the
collection each time, so re-running never duplicates data.

## How it works

`ingest.py` — reads the PDF page by page with `pypdf`, splits it into ~800
character chunks (150 char overlap), embeds them with `text-embedding-3-small`,
and stores them in a local Chroma database under `chroma_db/`.

`ask.py` — for each question:

1. **Query planning.** Multi-part questions ("who is the CEO *and* what is the
   revenue?") are split into separate searches, and follow-ups are rewritten
   into standalone questions using the chat history. Skipped for simple first
   questions, which are already good search queries.
2. **Retrieval.** Each query fetches its top 6 chunks. Results are
   *interleaved*, not concatenated, so one sub-question can't consume the whole
   context budget and starve the other. Capped at 10 chunks.
3. **Answering.** `gpt-4o-mini` (temperature 0) answers from those chunks only,
   citing page numbers. If nothing relevant is retrieved, the LLM isn't called
   at all — the fallback is returned directly.

`app.py` — a Streamlit wrapper around the same `PdfChatbot`. No retrieval or
prompting logic of its own; it opens the Chroma store once per server (shared
by all browsers) and keeps one bot per browser session, so two people's
follow-up questions don't resolve against each other's history.

### Units in financial tables

The statement tables (pages ~85–200) print bare numbers under a `Thousands of
EUR` header, so the turnover row reads `19,374,073`. The prompt makes the model
find that header and convert, giving *"€ 19.37 billion (19,374,073 thousand
EUR)"* rather than the 1000-fold understatement *"€ 19,374,073"*. Figures the
report already states in prose (*"€ 847 million"*) are quoted as written.

Page citations are usually right but can drift to a neighbouring table when
several similar tables are retrieved at once. The `SOURCES` list under each
answer is the reliable place to check.

## Configuration

Constants at the top of `ask.py`:

| Name | Default | Purpose |
| --- | --- | --- |
| `TOP_K` | 6 | Chunks fetched per search query |
| `MAX_SUBQUERIES` | 3 | Max searches for one multi-part question |
| `MAX_CONTEXT_CHUNKS` | 10 | Total chunks sent to the LLM |
| `MAX_HISTORY_TURNS` | 6 | Q&A pairs kept as conversation memory |
| `CHAT_MODEL` | `gpt-4o-mini` | Answering model |

If you change `EMBED_MODEL`, re-run `ingest.py` — the question and the stored
chunks must be embedded by the same model.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `OPENAI_API_KEY not found.` | Create `.env` with your key. Watch for Notepad saving it as `.env.txt`. |
| `PDF not found at: ...` | The report is not bundled. Save your copy in this folder as `Umicore Annual Report 2025.pdf`. |
| `Vector store './chroma_db' not found.` | Run `ingest.py` first. |
| `Collection ... is empty.` | Re-run `ingest.py`. |
| Answers are "I don't know" too often | Raise `TOP_K` / `MAX_CONTEXT_CHUNKS` in `ask.py`. |
| `chroma_db/` grows by ~9 MB per ingest | Expected. Chroma drops the old collection but leaves its UUID-named folder on disk. To reclaim: delete the whole `chroma_db/` folder and re-run `ingest.py`. |

Note: `.env` holds a real API key. It's already in `.gitignore` — keep it that
way, and don't commit `chroma_db/` either (it's ignored too).

## License

Source code is MIT licensed — see [LICENSE](LICENSE).

The report itself is not included in this repository — it is published by
Umicore and is theirs to distribute.
