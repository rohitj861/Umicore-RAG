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

`ingest.py` deletes the store directory and rebuilds it each time, so
re-running never duplicates data and never accumulates dead segment folders.
Committing the result replaces roughly 28 MB.

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

### Table headers are carried onto their rows

A statement page prints its units and year columns once, then many rows
beneath. Splitting at 800 characters puts most rows in chunks that no longer
contain the header. On page 91 the header sits at offset ~1100 and the Turnover
row at ~1874, so the row arrived as six bare numbers:

```
Turnover a 14,853,681 14,859,584 (5,903) 19,374,073 18,849,795 524,279
```

Asked for 2024 turnover, the model answered `€ 18.85 billion (18,849,795)` —
the *2025 adjusted* column, labelled as 2024. Confidently, and every time.

`ingest.py` now finds every table header on a page, records its offset, and
prefixes each chunk with the header that *precedes* it:

```
[page 91 table header: Adjustments included in the result | 2024 2025 |
 Thousands of EUR Notes Total Adjusted Adjustments Total Adjusted Adjustments]
Turnover a 14,853,681 14,859,584 (5,903) ...
```

Nearest-preceding matters: page 91 carries R&D expenditure *above* the
reconciliation, and captioning a six-column row with that two-column header
would be worse than no caption. Prompt rules 7 and 8 tell the model to read
that line as the header, how to map grouped columns to years, and to declare a
figure ambiguous rather than pick a plausible column.

Measured over three runs of a nine-turn conversation at the shipped settings:
**27 of 27 correct, none wrong** — the comparative figures, the prose figure,
the reconciliation column, both four-column segment cases, the percentage
series, and the unanswerable question all held.

Measured after the change, three repeats per case:

| `TOP_K`, `MAX_CONTEXT_CHUNKS` | correct | refused | **wrong** |
| --- | --- | --- | --- |
| 6, 10 | 21 | 3 | **0** |
| **10, 16 (current)** | **24** | **0** | **0** |
| 14, 20 | 24 | 0 | **0** |

This reverses an earlier finding recorded here. On the store built *without*
header propagation, raising `TOP_K` to 10 produced wrong figures and 6 was the
safest setting. With headers attached, the wrong-column failure is gone at
every depth, and 6 is merely shallow — it missed free cash flow, whose chunk
ranks 11th. Hence 10/16. Re-measure before changing either number.

A header line qualifies two ways: it names units (`Thousands of EUR`,
`(in million €)`, `in %`) or it is a run of years separated by nothing but
whitespace (`2024 2025`, `Company performance 2021 2022 2023 2024 2025`).
Whitespace-only separators are what keep prose out — *"between 2024 and 2025"*
does not match. Two guards stop data rows being mistaken for headers: a
thousands-separated number disqualifies the year route, and a candidate line
above the header is dropped if it carries three or more non-year figures.
Without the second, page 52 captioned its table with the row
`% change versus previous year 7.5% 5.6% 10.2% -0.9% 4.8%`.

Coverage: **80 of 219 pages** carry a header onto 320 chunks. Three pages
(16, 53, 54) mention a year pair without being tables.

The riskiest layout is the segment key figures on pages 18–24, which print four
columns per row — `Key figures H2 H2` over `(in million €) 2024 2025 2024 2025`,
a half-year pair then a full-year pair. Graded on Catalysis full-year 2025
turnover (`4,482`, against the traps `2,178` for H2 and `4,346` for 2024) and on
ROCE 2025 (`15.7%`, trap `12.3%`), three repeats each, all correct.

### What this does not cover

Verified question shapes are the ones above: comparative statement figures,
prose figures, a reconciliation column, four-column segment tables, a
percentage series, and a question the report cannot answer. That is not
coverage of a 220-page report, and the model is not deterministic — treat the
`SOURCES` panel as the check on any figure that matters.

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
| `TOP_K` | 10 | Chunks fetched per search query |
| `MAX_SUBQUERIES` | 3 | Max searches for one multi-part question |
| `MAX_CONTEXT_CHUNKS` | 16 | Total chunks sent to the LLM |
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
| `git status` shows `chroma_db/chroma.sqlite3` modified, and you changed nothing | Opening the store writes to internal sqlite pages, so simply running the app dirties the tracked file. The data is unchanged (same size, same contents). Discard it with `git checkout chroma_db`. |
| Answers are "I don't know" too often | Raise `TOP_K` / `MAX_CONTEXT_CHUNKS` in `ask.py` — but re-measure afterwards. Depth is only safe because table headers travel with their rows; on a store built without that, raising it produced confidently wrong figures. See *Table headers are carried onto their rows*. |
| `chroma_db/` seems to be growing | It no longer should. `delete_collection()` used to leave each run's UUID-named segment folder on disk and its sqlite pages unreclaimed, which grew a 28 MB store to 51 MB over three ingests. `ingest.py` now deletes the store directory before rebuilding, so a rebuild starts from nothing. It refuses to delete a directory without a `chroma.sqlite3` in it, in case `persist_dir` is mistyped. |

Note: `.env` holds a real API key. It's already in `.gitignore` — keep it that
way, and don't commit `chroma_db/` either (it's ignored too).

## License

Source code is MIT licensed — see [LICENSE](LICENSE).

The report itself is not included in this repository — it is published by
Umicore and is theirs to distribute.
