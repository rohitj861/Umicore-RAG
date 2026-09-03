# Umicore Annual Report 2025 — Q&A (RAG)

**Live app → <https://umicore-rag.streamlit.app/>**

Ask questions in plain English about the Umicore Annual Report 2025. Answers
come only from the report; if it does not contain the answer, the bot replies
**"I don't know about this."** rather than guessing, and every answer cites the
pages it was written from.

> **The app is public but password-protected.** No Streamlit account is needed
> — anyone can open the link, and a password field is the first thing they
> meet. Ask the owner for the phrase, or run your own copy (below). The gate
> exists because every answer spends the owner's OpenAI credits.

Nothing to install to use it. Open the link, ask a question, and expand
**Sources** under the answer to read the exact chunks of the report it was
written from — the quickest way to check that a figure really is in the PDF.

Two searches run behind every question — embeddings for meaning, BM25 for exact
wording — and the sidebar lets you switch to vector-only search to compare.
*How it works* explains why.

Everything below this line is for running or changing the project yourself.

## Running your own copy

Needed only to change the code, point it at a different PDF, or use it without
the hosted app. The vector store is committed, so there is nothing to ingest
and no PDF to download — a virtual environment and an OpenAI key is the whole
setup.

```powershell
cd "C:\RAG-Umicore\Umicore-RAG"

# 1. Virtual environment + dependencies
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

# 2. API key: copy the template, then put your real key in the copy
Copy-Item .env.example .env
notepad .env
```

`.env` is gitignored. `.env.example` also carries a commented-out
`APP_PASSWORD` line — leave it commented for a local run and the app is open;
uncomment it to put the same password gate in front of your local copy.

### Rebuilding the store

Needed to point the project at a different PDF, to change `EMBED_MODEL`, or to
change how the PDF is split. The committed store was built with semantic
chunking (1,324 chunks) and is ready to use as shipped.

The report is not in this repo — it is Umicore's to distribute — so supply your
own copy first:

```powershell
# Save the report in this folder as exactly: Umicore Annual Report 2025.pdf
# (Any PDF works if you change pdf_path in ingest.py.)
.\.venv\Scripts\python.exe ingest.py   # ~1-2 min, costs a few cents
```

`ingest.py` deletes the store directory and rebuilds it each time, so
re-running never duplicates data and never accumulates dead segment folders.
Committing the result replaces roughly 28 MB.

## Asking questions

The chat UI is the same whether you open the hosted app or run it locally —
`app.py` is the one entry point, and the deployment runs exactly this file.

| Where | How |
| --- | --- |
| **Hosted** | <https://umicore-rag.streamlit.app/> — nothing to install |
| **Local** | `.\.venv\Scripts\python.exe -m streamlit run app.py` → <http://localhost:8501> |

Each answer carries a **Sources** panel you can expand to read the exact chunks
of the report it was written from — the quickest way to check an answer is
really in the PDF.

The sidebar has a **Search** switch with two modes:

| Mode | What it does |
| --- | --- |
| **Hybrid — vector + BM25** (default) | Both searches per query, merged by reciprocal rank fusion |
| **Vector only** | Embedding similarity alone — how this project worked before fusion |

Switch between questions to compare them; it affects the next search only and
leaves the conversation intact. Each answer's Sources panel records the mode
that produced it, so a transcript stays readable after you switch.

The difference shows up in **which pages reach the model**, so compare the
Sources panels rather than only the answers. A question pinned to an exact
figure is the clearest case — *"What does the turnover figure 19,374,073 refer
to?"* cites page 90 (total segment turnover) in hybrid mode and the IFRS 15
prose in vector-only. Expect many well-phrased questions to answer identically
in both modes; that is the honest result, not a broken switch.

Note that a bare figure typed on its own (`19,374,073`) is refused in *both*
modes — the prompt answers questions, and a number alone is not one.

Type **exit** (or `quit`, `bye`, `q`) to end the conversation; a *Start a new
chat* button appears. On a local run, closing the browser tab leaves the server
running — stop it with `Ctrl+C` in the terminal.

Follow-up questions work in either place — the bot remembers the last 6
exchanges, so "and what about the year before that?" resolves correctly. Memory
is per browser session, so two people using the hosted app at once do not
resolve follow-ups against each other's history.

### Terminal (local only)

```powershell
# Interactive chat (type 'exit' to quit)
.\.venv\Scripts\python.exe ask.py

# One-shot
.\.venv\Scripts\python.exe ask.py "Who is the CEO?"
```

Same retrieval and the same prompt as the browser UI — `app.py` is a wrapper
around this. The hosted app has no terminal, so this is for local use.

## How it works

`ingest.py` — reads the PDF page by page with `pypdf`, splits each page
*semantically* (`chunking.py`), embeds the chunks with `text-embedding-3-small`,
and stores them in a local Chroma database under `chroma_db/`.

`ask.py` — for each question:

1. **Query planning.** Multi-part questions ("who is the CEO *and* what is the
   revenue?") are split into separate searches, and follow-ups are rewritten
   into standalone questions using the chat history. Skipped for simple first
   questions, which are already good search queries.
2. **Retrieval.** Every query is searched twice — dense vectors over Chroma and
   BM25 keywords — and all the resulting rankings are merged by reciprocal rank
   fusion (`retrieval.py`). 16 chunks per retriever per query, capped at 16
   after fusion.
3. **Answering.** `gpt-4o-mini` (temperature 0) answers from those chunks only,
   citing page numbers. If nothing relevant is retrieved, the LLM isn't called
   at all — the fallback is returned directly.

`app.py` — a Streamlit wrapper around the same `PdfChatbot`. No retrieval or
prompting logic of its own; it opens the Chroma store and builds the BM25 index
once per server (shared by all browsers) and keeps one bot per browser session,
so two people's follow-up questions don't resolve against each other's history.

`evaluate.py` — grades the answers against figures read out of the report, so a
change to the splitter, the retrieval settings or the prompt can be checked
rather than eyeballed. See *Checking it* below.

The two modules the retrieval rests on are `chunking.py` (how the PDF is cut
up) and `retrieval.py` (how chunks are found and ranked); each is described in
its own section.

### Semantic chunking

A fixed-size splitter cuts at a character count and only prefers a line break
near it, which on this report means cutting a sentence mid-clause and cutting
tables between rows that belong to the same block of figures. A chunk then
holds the tail of one topic and the head of the next, and matches neither well.

`chunking.py` cuts where the *meaning* changes instead:

1. The page is broken into units — one per extracted line, split further where
   a line holds several sentences. Lines are the unit because `pypdf` emits one
   line per visual line, so a table row arrives whole and is never cut in half.
2. Each unit is embedded together with its immediate neighbours, so a bare row
   like `Turnover 19,374,073` is measured in the context it appears in rather
   than as six digits on their own.
3. The cosine distance between consecutive units is measured, and a cut is made
   wherever that distance exceeds the **92nd percentile of the page's own
   distances**. The threshold is relative on purpose: a page of dense table
   rows drifts far less between units than a page of prose, and one fixed
   cut-off would leave the first whole and shred the second.
4. Chunks longer than 1200 characters are split again at their own widest
   remaining seam; chunks shorter than 250 characters are folded back into a
   neighbour.

Chunks are exact substrings of the page, never re-joined text, so the character
offsets that table-header propagation depends on stay truthful.

The cost is one extra embedding pass at ingest time over short units — and it
is genuinely small: 552k tokens over 17,894 units, about **1.1 cents** on top of
the 0.4 cents to embed the chunks themselves. Nothing at query time.

Measured on the 2025 report: **219 pages → 17,894 units → 1,324 chunks**, median
643 characters. 1.4% are at the 1200-character cap (split by size rather than
meaning) and 1.5% fall under the 250-character floor (short pages, and the tail
end of a cap-split). Median 6 chunks per page.

The splitter earns its keep most visibly on tables. Page 18's key-figures table
keeps its column header attached to the rows it governs, in one chunk:

```
Group key figures
Key figures H2 H2
(in million €) 2024 2025 2024 2025
Turnover 7,407 10,680 14,854 19,374
Adjusted EBITDA 370 414 763 847
```

The running page header (`Financial statements – Umicore Annual Report 2025 91`)
is a semantic outlier from the body text below it, so a breakpoint fires right
after it on almost every page. That would leave 219 one-line chunks; the
minimum-size rule folds them forward, and only 3 survive as their own chunk.

### Two searches, fused

Vector search matches meaning, and is weakest where meaning is thinnest. A bare
figure is the clearest case: searching this store for `19,374,073` returns the
employee-numbers table first, because six digits give an embedding almost
nothing to match on. BM25 finds the pages that literally print it. BM25 fails
the other way — it needs the question to share wording with the report, and
returns nothing for a paraphrase.

Both handle a well-phrased question well, and usually agree. Asked *"Which note
covers IFRS 16 leases?"*, both modes answer *"Note F2.8 (page 71)"*. What fusion
buys is the tail, where one retriever has nothing useful to offer.

`retrieval.py` runs both over the same Chroma collection (the BM25 index is
built from the store at start-up, so there is nothing to keep in sync and no
second artefact to deploy) and merges the rankings with reciprocal rank fusion:

```
score(chunk) = sum over rankings r of  weight_r / (60 + rank_r(chunk))
```

Ranks are added, not scores, and that is the point: a cosine similarity of 0.83
and a BM25 score of 11.4 are not comparable quantities, and normalising them
needs a calibration that changes with every query. Positions in a list need
none. A chunk both retrievers rank well finishes above one that a single
retriever put first and the other never returned — agreement between two
different notions of relevance is what gets rewarded.

Fusion also replaced the hand-written interleaving that stopped one sub-question
of a multi-part question filling the whole context: each ranking's first hit is
worth the same, whichever query produced it, so the balance is now a property of
the arithmetic.

Tokenisation is tuned for a financial report. Figures keep their separators, so
`19,374,073` is one token rather than three, and are indexed bare as well so a
question typed without commas still matches. The stopword list is deliberately
short — `report`, `year` and `group` all discriminate between pages here.

**Does it earn its place?** Over twelve representative queries, BM25 supplied
**5.4 of the 16 context slots per query** that vector search alone would not have
returned — a third of what reaches the model. And for 11 of those 12 queries the
chunk fusion ranks first is one *both* retrievers returned, which is the
agreement effect RRF exists to produce.

The exception is instructive. On the bare string `19,374,073` the two retrievers
overlap on nothing at all — a lone figure carries almost no meaning for an
embedding to match — so every rank-1 candidate ties, and at equal weights the
tie broke by list order in favour of a vector hit for the employee-numbers
table. `BM25_WEIGHT` is therefore 1.2 rather than 1.0: enough to settle that
tie toward the retriever that actually matched the literal string, which
returns pages 87, 91 and 88 instead. It is the smallest value that does so, it
changed no graded answer for the worse, and raising it further changes nothing.

### The same label, different figures

The single largest source of wrong numbers here is not retrieval quality. It is
that **the report states the same row label many times at different scopes**.
An audit of the store found **74 labels that appear on more than one page with
different figures**. `Adjusted EBITDA` alone appears five times:

| Page | Scope | FY2025 |
| --- | --- | --- |
| 18 | **Group** | **847** |
| 19 | Battery Materials Solutions | (21) |
| 21 | Catalysis | 450 |
| 23 | Recycling | 371 |
| 24 | Specialty Materials | 108 |

Whichever chunk retrieval happened to surface decided the answer, and nothing
required the model to say which scope it had used — so a segment figure could
be reported as the Group's with no visible sign anything was wrong. The same
shape caused a reported bug: profit before income tax came back as 845,345
(note F13, *of consolidated companies*) instead of 771,739 (the consolidated
income statement).

Three fixes, none of which works alone:

**The scope is already in the chunk.** Table-header propagation means each
chunk carries `Group key figures` or `Catalysis key figures` at the top, so the
model can tell them apart — it simply was not told the distinction mattered.
Prompt rule 9 now lists the scopes, requires an unscoped question to be
answered with the Group figure, and requires *every* figure to be reported with
its scope named.

**One page could monopolise the context.** Asked for profit before income tax,
note F13 (page 95) took three of the sixteen slots — the note discusses the
line at length, while the income statement states it once among thirty rows —
and pushed the statement to ninth, or out entirely. Both retrievers agreed on
the note, so this is not something fusion can repair: it is what both were
asked for. `MAX_PER_PAGE = 2` caps how much of the context any one page may
occupy; chunks over the quota are set aside and fill any shortfall at the end,
so the context never shrinks. 2 measured better than 1 (which costs a table
its own rows) and better than no cap.

**Notes outrank statements in every search, so the prompt has to say they
don't.** That instruction sits *above* the numbered rules, not inside them: as
a sub-bullet of rule 9 it was ignored, and the model kept answering from
whichever chunk came first.

### Revenue is not turnover

Asked *"total revenue in 2025"*, the bot answered **€ 19.37 billion** — the
turnover. Umicore's revenue line is `Revenues (excluding metal)`, defined on
page 11 as all revenue elements minus the value of the purchased metals, and
the group figure for 2025 is **3,562** (€ 3.56 billion) on page 18. The two
differ by a factor of five, so this was a large error, not a rounding one.

It had two independent causes, and needed both fixed:

**The question searched for the wrong words.** For the query *"total revenue in
2025"* the page 18 row ranks **40th** in vector search and outside the top 60
in BM25 — it never reached the model, which then answered with the closest
thing it *had* been given. Rewritten in the report's own vocabulary
(*"revenues excluding metal 2025"*) the same row ranks 10th–12th in both. The
query rewriter already existed for follow-ups and multi-part questions; it now
also knows this report's house vocabulary, and `HOUSE_TERMS` makes a question
containing "revenue" worth the rewrite call even when it is a simple first
question. Add a term there only with a measurement behind it.

**`TOP_K` was 10, and the row ranks 10th–12th.** A near miss in *both*
retrievers is the one thing fusion cannot repair — it reranks what the
retrievers return, and neither returned it. `TOP_K` is now 16, which brings in
page 18 and page 87 (the same figure in thousands). This costs no prompt
tokens: it widens the candidate pool, and `MAX_CONTEXT_CHUNKS` still caps the
context at 16. Higher is not better — at 24, page 87 gets crowded out again.

Prompt rule 10 then states the distinction outright, because retrieval alone
did not settle it: with both figures in context the model still has to be told
which one the question asked for. The rule is phrased affirmatively —
`Revenues (excluding metal)` *is* the report's revenue figure — after a first
attempt phrased as a prohibition made the model refuse a question the report
answers perfectly well.

The general lesson is worth keeping: **a domain term the source defines its own
way will not be caught by better chunking or better fusion.** The retrieval was
working correctly and returning good matches for the words it was given; the
words were wrong.

### Checking it: `evaluate.py`

```powershell
.\.venv\Scripts\python.exe evaluate.py            # both modes, 28 cases = 56 questions
.\.venv\Scripts\python.exe evaluate.py --hybrid   # hybrid only
.\.venv\Scripts\python.exe evaluate.py --quick    # group key figures only
```

Run it after changing the splitter, the retrieval settings or the prompt. None
of those changes announce themselves, and a wrong figure looks exactly like a
right one.

Each case names the figure the report gives **and the lookalikes that would be
wrong** — the segment figure, the prior-year column, the note's subtotal. A
case passes only if the answer contains the right figure and none of the wrong
ones, so an answer taken from the wrong column or the wrong business group
fails instead of passing on a keyword.

Matching is on the figure, not its spelling: 771,739 counts whether it is
written `771,739`, `771.74 million` or `771.7`, because the prompt asks for
unit conversion. Roundings are only accepted while they stay faithful to within
0.5% — without that, 4,346 rounded to millions is `4`, which matches inside
"4.48 billion" and marks a correct answer wrong.

**The unit is checked separately from the figure**, because the two fail
independently — and getting the row right while reporting it a thousandfold too
large is the more dangerous of the two, since the digits still match the PDF
when spot-checked. `unit_errors()` fails any answer that writes a statement
figure as `€ (1,424,122) million` (raw thousands labelled millions) or as
`€ 19,374,073` with no scale word at all.

That check was added after finding exactly one such answer in 22: every
positive figure converted correctly, but the **parenthesised negative** did
not — `(1,424,122)` under *Thousands of EUR* came back as
`€ (1,424,122) million` instead of a loss of € 1.42 billion. Rule 6 said
nothing about brackets, so it now spells out that brackets mean negative, that
the conversion applies unchanged, and that the sign must be carried into words.

### Half-year before full-year

Fixing the brackets surfaced a second one, in the same place. The key figures
tables — `Group key figures` and every business group's — are laid out

```
Key figures            H2      H2
(in million €)       2024    2025    2024    2025
EBITDA                244     781  (1,025)   1,212
```

so the **full-year columns are third and fourth**, and the first two are half-
year. Asked for 2024 EBITDA the bot answered **€ 244 million** — the H2 figure.
The full year was `(1,025)`, a loss.

Five other rows on the same table answered 2024 correctly, so this was not a
general column failure: it was the one row whose full-year figure is negative,
and the model preferred the positive-looking column. Rule 8 now sets out this
table's four-column layout explicitly, says an unqualified year means the full
year, and says a bracketed full-year figure is still the answer.

`FULL_YEAR_VS_HALF_YEAR` in `evaluate.py` guards all six rows, with each row's
H2 column as the wrong answer. Asking about **2024** is what exposes this —
for 2025 the wanted column is last and hard to get wrong, which is why a
suite written only around 2025 missed it.

Current state, 28 cases:

| Mode | Score |
| --- | --- |
| **Hybrid — vector + BM25** | **28 / 28** |
| Vector only | not re-measured since the suite grew |

Vector-only last scored 17 of the 22 cases that existed before the half-year
section was added, and every one of those failures was a page it never
retrieves — the Group key figures for adjusted EBITDA, the consolidated income
statement for profit before income tax. That gap is the clearest measured
argument for the hybrid default, and the reason hybrid is the default.

### What was measured after the change

The eight cases below were run against the rebuilt store, each with a fresh
conversation, one run apiece:

| Question | Answer | Verdict |
| --- | --- | --- |
| Adjusted EBITDA 2025 | € 847 million | correct |
| Turnover in 2024 | € 14.85 billion (14,853,681 thousand) | correct — avoided the 2025-adjusted column |
| ROCE 2025 | 15.67% (page 121, note F32) | correct — avoided the 12.31% 2024 column |
| Catalysis full-year turnover 2025 | € 4.48 billion (4,482) | correct — avoided H2 (2,178) and 2024 (4,346) |
| Free cash flow | € 524 million, vs € 384 million prior year | correct |
| Total R&D expenditure 2025 | € 205.7 million | correct |
| Employee headcount | 11,230 (2025), 11,581 (2024) | correct |
| Cryptocurrency mining policy | *"I don't know about this."* | correct refusal |
| Total revenue 2025 | € 3.56 billion (3,562 million), noted as excluding metal | correct — see *Revenue is not turnover* |

**9 of 9.** All three column traps that the earlier grading run was built around
were avoided, and the revenue case above re-checked after its fix. One run per case is not the three-repeat protocol used for the
table in the next section, so treat this as a smoke test that the rebuild and
the new retrieval hold up, not as a re-run of that measurement.

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
beneath. Any split puts most rows in chunks that no longer contain the header.
On page 91 the header sits at offset ~1100 and the Turnover row at ~1874, so
the row arrived as six bare numbers:

```
Turnover a 14,853,681 14,859,584 (5,903) 19,374,073 18,849,795 524,279
```

Asked for 2024 turnover, the model answered `€ 18.85 billion (18,849,795)` —
the *2025 adjusted* column, labelled as 2024. Confidently, and every time.

Semantic chunking narrows this — rows that belong together now tend to stay
together — but does not remove it. A long table still exceeds the size cap, and
the header band and its first data row are exactly the pair of neighbours whose
embeddings differ most, so a cut there is the one the splitter most wants to
make. `ingest.py` therefore still finds every table header on a page, records
its offset, and prefixes each chunk with the header that *precedes* it:

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
| **10, 16** | **24** | **0** | **0** |
| 14, 20 | 24 | 0 | **0** |

This reverses an earlier finding recorded here. On the store built *without*
header propagation, raising `TOP_K` to 10 produced wrong figures and 6 was the
safest setting. With headers attached, the wrong-column failure is gone at
every depth, and 6 is merely shallow — it missed free cash flow, whose chunk
ranks 11th. Hence 10/16.

> **These figures predate semantic chunking and hybrid retrieval.** They were
> measured on fixed-size chunks under a vector-only search. Chunks are now
> larger and vary in size, `TOP_K` now applies per retriever rather than per
> query, and fusion changes which chunks arrive rather than how many. The
> grading run above has not been repeated since — treat 10/16 as a carried-over
> default, and re-measure before relying on either number or changing it.

A header line qualifies two ways: it names units (`Thousands of EUR`,
`(in million €)`, `in %`) or it is a run of years separated by nothing but
whitespace (`2024 2025`, `Company performance 2021 2022 2023 2024 2025`).
Whitespace-only separators are what keep prose out — *"between 2024 and 2025"*
does not match. Two guards stop data rows being mistaken for headers: a
thousands-separated number disqualifies the year route, and a candidate line
above the header is dropped if it carries three or more non-year figures.
Without the second, page 52 captioned its table with the row
`% change versus previous year 7.5% 5.6% 10.2% -0.9% 4.8%`.

Coverage on the semantic store: **80 of 219 pages** carry 137 headers onto
**298 chunks**. (Header detection runs over page text, not chunks, so the page
count is unchanged from the fixed-size store; the chunk count fell from 320
because semantic chunks are larger and fewer rows get separated from their
header in the first place.) Three pages (16, 53, 54) mention a year pair
without being tables.

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

## The deployment

This repo is already deployed on Streamlit Community Cloud:

| | |
| --- | --- |
| URL | <https://umicore-rag.streamlit.app/> |
| Tracks | branch `main`, entry point `app.py` |
| Python | 3.11 |
| Visibility | **Settings → Sharing**: *public and searchable* — no Streamlit account needed to open it |
| Secrets | `OPENAI_API_KEY`, `APP_PASSWORD` — set in the app's **Settings → Secrets**, never in the repo |

Those last two work together, and it is worth being clear which does what.
Streamlit's sharing setting decides who may **load** the app; `APP_PASSWORD`
decides who may **use** it. Public sharing plus a password means anyone can
reach the URL and nobody gets an answer without the phrase — which is the
intended arrangement here. Set sharing to private instead and viewers would
need a Streamlit account *as well as* the password, which locks out anyone you
just want to hand a link to.

**It redeploys itself on every push to `main`.** There is no deploy step to
run: commit, push, and the app rebuilds. Use **Reboot** (app menu → Reboot) to
force it, which is also what picks up a changed `requirements.txt`.

Python 3.11 is older than the 3.14 this was developed on, and deliberately left
alone — every pin resolves on it (all require ≥3.9 or ≥3.10, `pysqlite3-binary`
ships a cp311 wheel, `chromadb` an abi3 wheel from cp39), and changing it costs
a full rebuild for no gain.

### Why it needs so little to run

The vector store under `chroma_db/` is committed, so a hosted deployment has
everything it needs: it never runs `ingest.py`, never needs the PDF, and spends
nothing on embeddings at boot. The BM25 index is built from that same store at
start-up rather than shipped, which costs a second or two on the first request
and nothing after — it is cached for the life of the server process, shared
across browser sessions. Note what committing the store means — it holds the
report's extracted text alongside the vectors, so this repository carries that
text even though the PDF itself is not here.

**Community Cloud builds from GitHub, not from your working copy.** Anything
uncommitted simply is not there, and the two ways that bites are both silent
until the app boots:

```powershell
git status --short          # nothing untracked that app.py imports
git add -A                  # source, requirements, AND chroma_db/
git commit -m "..."
git push
```

- `app.py` imports `ask.py`, which imports `retrieval.py`, which is imported
  alongside `chunking.py` — miss one and the deploy dies with `ModuleNotFound`.
- **The store is a directory, not a file.** Every rebuild writes a new
  UUID-named segment folder under `chroma_db/`; `git add -A` picks it up,
  `git add *.py` does not. A store missing its segment folder opens without
  error and then answers nothing.

### Deploying a fresh copy

To stand up your own instead of using the one above:

1. **New app** → point it at your fork, branch `main`, file `app.py`.
2. **Advanced settings → Python version.** Community Cloud offers only
   released, security-supported versions, and the version is chosen here —
   there is no `runtime.txt`. Anything from 3.10 up works: `pysqlite3-binary`
   ships cp38–cp314 wheels, `chromadb` an abi3 wheel from cp39, and the rest
   are pure-Python wheels.
3. **Secrets.** Paste the contents of `.streamlit/secrets.toml.example` with
   your real key. `app.py` copies these into the environment at startup, which
   is where `ask.py` looks for them — there is no `.env` in a deployment, and
   secrets never belong in the repo.
4. Deploy. First boot takes a few minutes.

Only `requirements.txt` is present, which is what Community Cloud wants — it
picks the *first* dependency file it finds (`uv.lock`, `Pipfile`,
`environment.yml`, `requirements.txt`, `pyproject.toml`) and ignores the rest,
so do not add a second one.

### What the first boot spends

Installing takes a few minutes because `chromadb` pulls `onnxruntime` (~45 MB),
`grpcio`, `kubernetes` and OpenTelemetry, and `streamlit` pulls `pandas` and
`pyarrow` — roughly 200 MB of wheels, of which this project uses very little.
They are hard dependencies of Chroma, so they cannot be trimmed without
replacing the vector store.

After that, a cold start opens the 27 MB store and builds the BM25 index over
1,324 chunks — a second or two, once per server process, then cached for every
session. No embeddings are computed at boot and no API call is made until
someone asks a question.

### Guarding the URL

A public Streamlit URL is reachable by anyone who has it, every answer spends
the owner's OpenAI credits, and the app has no rate limiting. Two defences,
both already in place on the deployment above:

- **A hard monthly spend cap** in the OpenAI billing dashboard. It is the only
  limit that cannot be bypassed, and the only one worth relying on — a password
  protects against strangers, not against a shared password.
- **`APP_PASSWORD`**, which puts a password in front of the app. **You invent
  this value** — it is not issued by OpenAI, GitHub or Streamlit, and there is
  nowhere to go and fetch it. Pick a long phrase and give it to whoever should
  have access.

  Where it goes depends on where the app runs:

  | Running | Put it in | Looks like |
  | --- | --- | --- |
  | Streamlit Community Cloud | app → **Settings → Secrets** | `APP_PASSWORD = "your long phrase"` |
  | Locally | `.env`, uncommented | `APP_PASSWORD=your long phrase` |

  Both files ship with the line present but commented out, so the app is open
  until you deliberately switch the gate on. Unset means no gate — that is the
  sensible default locally and the wrong one on a public URL.

  The gate runs before the vector store is opened, so an unauthenticated
  visitor cannot trigger any API call. Changing it later means editing the
  secret and letting the app reboot; there is no password reset flow, because
  there are no accounts — just the one shared phrase.

### If chromadb fails to import on a deploy

Community Cloud runs Debian 11, which ships sqlite 3.34 - below the 3.35
chromadb requires.
`requirements.txt` already installs `pysqlite3-binary` on Linux and `app.py`
swaps it in for the stdlib module before chromadb loads, so this should be
handled — but that swap is what the traceback will be about if it isn't.

## Configuration

Constants at the top of `ask.py`:

| Name | Default | Purpose |
| --- | --- | --- |
| `MAX_SUBQUERIES` | 4 | Max searches for one multi-part question |
| `MAX_CONTEXT_CHUNKS` | 16 | Total chunks sent to the LLM after fusion |
| `MAX_HISTORY_TURNS` | 6 | Q&A pairs kept as conversation memory |
| `CHAT_MODEL` | `gpt-4o-mini` | Answering model |

Retrieval, at the top of `retrieval.py`:

| Name | Default | Purpose |
| --- | --- | --- |
| `TOP_K` | 16 | Chunks fetched **per retriever, per query**. The candidate pool, not the context — `MAX_CONTEXT_CHUNKS` still caps what reaches the model, so raising this costs no prompt tokens |
| `RRF_K` | 60 | Fusion constant: the rank at which a hit is worth half of first place. Lower it to let a single retriever's top hit dominate |
| `VECTOR_WEIGHT` | 1.0 | Weight of the vector ranking in the fusion |
| `BM25_WEIGHT` | 1.2 | Weight of the BM25 ranking. Above 1.0 only to settle rank-1 ties on exact-string queries — see above |
| `MAX_PER_PAGE` | 2 | Most context slots any one page may occupy, so a note that discusses a figure at length cannot crowd out the statement that reports it. Chunks over quota are set aside and refill any shortfall, so the context never shrinks |

The two weights are the lever to reach for when answers miss in a way that is
characteristically one retriever's fault: keyword misses argue for more BM25,
paraphrase misses for more vector.

Chunking, at the top of `chunking.py` — these only take effect on a rebuild:

| Name | Default | Purpose |
| --- | --- | --- |
| `BREAKPOINT_PERCENTILE` | 92.0 | Cut where a distance exceeds this percentile of the page's own distances. Lower cuts more often |
| `BUFFER_SIZE` | 1 | Neighbouring units mixed into a unit before embedding it |
| `MIN_CHUNK_CHARS` | 250 | Below this, a chunk is folded into a neighbour |
| `MAX_CHUNK_CHARS` | 1200 | Above this, a chunk is split again at its widest seam |

If you change `EMBED_MODEL`, re-run `ingest.py` — the question and the stored
chunks must be embedded by the same model. `ingest.py` measures its breakpoints
with that same model too, so the setting governs both.

## Troubleshooting

| Message | Fix |
| --- | --- |
| `OPENAI_API_KEY not found.` | Locally: create `.env` with your key - watch for Notepad saving it as `.env.txt`. On the hosted app: the secret is missing or misspelt in **Settings > Secrets**, and note it is TOML there (`OPENAI_API_KEY = "sk-..."`, quoted) rather than `.env` syntax. |
| `OPENAI_API_KEY is still the placeholder` | `.env` still holds the example value. Put your real key in it. |
| `OpenAI rejected the API key in .env.` | The key reached OpenAI and was refused — check it is complete and current, with no quotes or trailing spaces. |
| `PDF not found at: ...` | Only `ingest.py` needs the PDF. The report is not bundled; save your copy in this folder as `Umicore Annual Report 2025.pdf`. |
| `Vector store './chroma_db' not found.` | The store is committed, so locally this means it was deleted - restore it with `git checkout chroma_db`, or rebuild via `ingest.py`. On the hosted app it means the store was never pushed: check `git ls-files chroma_db` lists the segment folder as well as `chroma.sqlite3`. |
| `Collection ... is empty.` | Re-run `ingest.py`. |
| `git status` shows `chroma_db/chroma.sqlite3` modified, and you changed nothing | Opening the store writes to internal sqlite pages, so simply running the app dirties the tracked file. The data is unchanged (same size, same contents). Discard it with `git checkout chroma_db`. |
| Answers are "I don't know" too often | Raise `TOP_K` in `retrieval.py` / `MAX_CONTEXT_CHUNKS` in `ask.py` — but re-measure afterwards. Depth is only safe because table headers travel with their rows; on a store built without that, raising it produced confidently wrong figures. See *Table headers are carried onto their rows*. |
| `chroma_db/` seems to be growing | It no longer should. `delete_collection()` used to leave each run's UUID-named segment folder on disk and its sqlite pages unreclaimed, which grew a 28 MB store to 51 MB over three ingests. `ingest.py` now deletes the store directory before rebuilding, so a rebuild starts from nothing. It refuses to delete a directory without a `chroma.sqlite3` in it, in case `persist_dir` is mistyped. |

Two notes on what is and isn't tracked, because they point opposite ways:

- **`.env` holds a real API key and must never be committed.** It is in
  `.gitignore`, along with `.streamlit/secrets.toml`. Keep both there. On the
  hosted app the key lives in **Settings → Secrets** instead, and the two are
  separate keys unless you deliberately use the same one.
- **`chroma_db/` *is* committed, on purpose.** The line for it in `.gitignore`
  is commented out. The deployment never runs `ingest.py`, so the store has to
  ship with the repo — ignoring it would leave the hosted app with nothing to
  search.

## License

Source code is MIT licensed — see [LICENSE](LICENSE).

The report itself is not included in this repository — it is published by
Umicore and is theirs to distribute.
