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

# That is all. The vector store under chroma_db/ is committed, so there is
# nothing to ingest and no PDF to download - go straight to asking questions.
```

### Rebuilding the store

Only needed to point the project at a different PDF, or to change
`EMBED_MODEL`. The report is not in this repo — it is Umicore's to distribute —
so supply your own copy first:

```powershell
# Save the report in this folder as exactly: Umicore Annual Report 2025.pdf
# (Any PDF works if you change pdf_path in ingest.py.)
.\.venv\Scripts\python.exe ingest.py   # ~1-2 min, costs a few cents
```

`ingest.py` wipes and rebuilds the collection each time, so re-running never
duplicates data. Committing the result replaces roughly 28 MB.

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

## Deploying it publicly

The vector store under `chroma_db/` is committed, so a hosted deployment has
everything it needs: it never runs `ingest.py`, never needs the PDF, and spends
nothing on embeddings at boot. Note what that means — the store holds the
report's extracted text alongside the vectors, so this repository carries that
text even though the PDF itself is not here.

On Streamlit Community Cloud:

1. **New app** → point it at this repo, branch `main`, file `app.py`.
2. **Advanced settings → Python version.** This project is developed on 3.14;
   if that isn't offered, pick 3.13. Every pin in `requirements.txt` has a
   cp313 wheel.
3. **Secrets.** Paste the contents of `.streamlit/secrets.toml.example`, with
   your real key. `app.py` copies these into the environment at startup, which
   is where `ask.py` looks for them — there is no `.env` in a deployment.
4. Deploy. First boot takes a few minutes while dependencies install.

### Before you make the URL public

Every answer spends **your** OpenAI credits, and the app has no rate limiting.

- **Set a hard monthly spend cap** in the OpenAI billing dashboard. It is the
  only limit that cannot be bypassed, and the only one worth relying on.
- **Set `APP_PASSWORD`** in the secrets to put a password in front of the app.
  The gate runs before the vector store is opened, so an unauthenticated
  visitor cannot trigger any API call. Leave it unset and the app is open to
  anyone with the link.

### If chromadb fails to import on first deploy

Some hosting images ship an sqlite older than the 3.35 chromadb requires.
`requirements.txt` already installs `pysqlite3-binary` on Linux and `app.py`
swaps it in for the stdlib module before chromadb loads, so this should be
handled — but that swap is what the traceback will be about if it isn't.

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
| `OPENAI_API_KEY is still the placeholder` | `.env` still holds the example value. Put your real key in it. |
| `OpenAI rejected the API key in .env.` | The key reached OpenAI and was refused — check it is complete and current, with no quotes or trailing spaces. |
| `PDF not found at: ...` | Only `ingest.py` needs the PDF. The report is not bundled; save your copy in this folder as `Umicore Annual Report 2025.pdf`. |
| `Vector store './chroma_db' not found.` | The store is committed, so this means it was deleted — restore it with `git checkout chroma_db`, or rebuild via `ingest.py`. |
| `Collection ... is empty.` | Re-run `ingest.py`. |
| Answers are "I don't know" too often | Raise `TOP_K` / `MAX_CONTEXT_CHUNKS` in `ask.py`. |
| `chroma_db/` grows by ~9 MB per ingest | Expected. Chroma drops the old collection but leaves its UUID-named folder on disk. To reclaim: delete the whole `chroma_db/` folder and re-run `ingest.py`. |

Note: `.env` holds a real API key. It's already in `.gitignore` — keep it that
way, and don't commit `chroma_db/` either (it's ignored too).

## License

Source code is MIT licensed — see [LICENSE](LICENSE).

The report itself is not included in this repository — it is published by
Umicore and is theirs to distribute.
