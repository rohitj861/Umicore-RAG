# RAGAS evaluation

An independent testing layer around the RAG pipeline. The app never imports
anything in this folder, and ragas is installed only in `evaluation/.venv`, so
`streamlit run app.py` behaves exactly the same whether or not this folder, its
environment or its API key exist.

It sits beside `evaluate.py`; it does not replace it. `evaluate.py` checks
figures and page citations against known values with exact rules. This folder
asks an LLM judge broader questions about grounding, retrieval and coverage.

## Setup (once)

```powershell
C:\Python314\python.exe -m venv evaluation\.venv
evaluation\.venv\Scripts\python.exe -m pip install -r evaluation\requirements.txt
```

Do not install ragas into the app's `.venv`: it would upgrade `langchain-core`
past the version `requirements.txt` pins.

## Run

From the project root:

```powershell
evaluation\.venv\Scripts\python.exe evaluation\run_ragas.py                  # hybrid search
evaluation\.venv\Scripts\python.exe evaluation\run_ragas.py --mode semantic  # semantic only
```

A run of 5 questions takes a few minutes. Most of that is waiting out the
OpenAI rate limit, and the `API call failed ... 429` lines it prints are those
waits being retried. They are not failures unless a row's `status` says so.

The run opens the vector store, which marks `chroma_db/chroma.sqlite3` as
modified even though its data is unchanged. `git checkout chroma_db` clears it.

## Configuration (environment variables / `.env`)

| Variable | Default | |
|---|---|---|
| `RAGAS_OPENAI_API_KEY` | falls back to `OPENAI_API_KEY` | Key for the evaluator |
| `RAGAS_EVALUATOR_MODEL` | `gpt-4o` | Judge model. Deliberately not the app's `gpt-4o-mini` |
| `RAGAS_EMBEDDING_MODEL` | `text-embedding-3-small` | Used by answer relevancy |
| `RAGAS_MAX_RETRIES` | `10` | Retries on a 429 rate-limit response |

## Adding questions

Add an entry to `eval_dataset.json`:

```json
{
  "user_input": "What was the capital expenditure in 2024?",
  "reference": "Capital expenditure in 2024 was € 555 million."
}
```

The reference should state the facts a complete answer must contain, and
nothing it may leave out. The completeness judge penalises every point in it
that the answer is missing.

The script fills in `response` and `retrieved_contexts` by asking the real
pipeline. A record that already has both is scored as it stands, without
calling the pipeline.

## Results

`results/ragas_results.csv` holds the latest run. Each run also saves a copy,
`results/ragas_results_YYYYMMDD_HHMM.csv`, for comparing over time. There is one
row per question, and every score is on a 0–1 scale where higher is better.

| Column | What it checks | A low score means |
|---|---|---|
| `faithfulness` | Is every claim in the answer supported by the retrieved chunks? | The model added something the chunks do not say (a hallucination, or a figure taken from the wrong row) |
| `context_precision` | Are the chunks that contain the answer ranked near the top? | Relevant chunks rank low, or the answer is buried among unrelated ones |
| `context_recall` | Do the retrieved chunks contain everything in the reference? | Retrieval missed the chunk holding the answer. The model cannot fix this |
| `answer_relevancy` | Does the answer address the question that was asked? | The answer is off-topic, evasive, or says a different thing from what was asked |
| `completeness` | Does the answer cover every point in the reference? The 0–5 rubric score divided by 5 | Part of the question went unanswered. `completeness_reason` lists what is missing |

Other columns: `completeness_raw_0_5` (the rubric score before dividing),
`retrieved_context_count`, `search_mode`, `status` (`ok`, `partial` when some
metrics failed, `failed`, or `pipeline_error`), and `errors`.

A failed metric is left blank and named in `errors`. A blank score is not a
zero, so read `status` before averaging a column.

### Reading this pipeline's scores

`context_precision` is roughly 1 divided by the rank of the first chunk
holding the answer. Irrelevant chunks ranked *below* it cost nothing. In hybrid
mode it averages about 0.28, because the chunk with the figure lands at rank
5–14 of 16. Definitions, targets and narrative prose that mention the term
outrank the table that prints the number. "Turnover" also matches employee
turnover (pages 186–187) and EU-taxonomy turnover (pages 147–150). Answers are
still right (`context_recall` is 1.0, because the chunk does make it into the
16), but it is the same weakness that forced `TOP_K` up to 16: the answer only
just gets in. Semantic-only mode, reranked by Cohere, scores about 0.72 on the
same questions.

The scores are LLM judgements, so they vary slightly between runs and can be
too strict. For example, the judge may mark "€ 19.37 billion" as incomplete
against a reference of "€ 19,374 million". Before acting on a low score, read
the row's `response` and `completeness_reason`.
