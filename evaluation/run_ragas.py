"""RAGAS evaluation of the RAG pipeline - run by hand, never by the app.

    evaluation\\.venv\\Scripts\\python.exe evaluation\\run_ragas.py
    evaluation\\.venv\\Scripts\\python.exe evaluation\\run_ragas.py --mode semantic

For each question in eval_dataset.json this asks the real pipeline (ask.py's
PdfChatbot, imported, not copied), records the chunks it retrieved and the
answer it gave, and scores them with four ragas metrics plus a custom
Completeness judge (completeness.py). Scores go to evaluation/results/ as CSV.

Nothing in the app imports this directory, and ragas lives in its own virtual
environment (evaluation/.venv), so the app runs the same whether ragas is
installed, broken or missing. See evaluation/README.md.

This is not evaluate.py. That one checks figures and page citations against
known values with exact rules; this one asks an LLM judge broader questions
about grounding, retrieval and coverage. They answer different things.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import sys
import types
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EVAL_DIR.parent
DATASET = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

# ragas sends usage telemetry by default. An evaluation of a private report
# has no reason to, so it is off unless the environment already says otherwise.
os.environ.setdefault("RAGAS_DO_NOT_TRACK", "true")

METRICS = [
    "faithfulness",
    "context_precision",
    "context_recall",
    "answer_relevancy",
    "completeness",
]
COLUMNS = [
    "user_input",
    "response",
    "reference",
    "retrieved_context_count",
    *METRICS,
    "completeness_raw_0_5",
    "completeness_reason",
    "search_mode",
    "status",
    "errors",
]


def _patch_ragas_import() -> None:
    """Let ragas 0.4.3 import against langchain-community 0.4.

    ragas imports ChatVertexAI from a langchain_community module that 0.4
    removed, and uses it only in an isinstance check that decides whether an
    LLM can return several completions at once. A stand-in class that nothing
    is ever an instance of gives that check its correct answer for an OpenAI
    model, and lets ragas import. Only this process is affected - no installed
    file is changed. Drop this when ragas stops importing the module.
    """
    name = "langchain_community.chat_models.vertexai"
    try:
        __import__(name)
    except ImportError:
        stub = types.ModuleType(name)
        stub.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules[name] = stub


def load_dataset(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    for i, record in enumerate(records, 1):
        for key in ("user_input", "reference"):
            if not str(record.get(key, "")).strip():
                raise ValueError(f"{path.name}, record {i}: '{key}' is missing or empty")
    return records


def run_pipeline(records: list[dict], use_bm25: bool) -> None:
    """Fill in `response` and `retrieved_contexts` from the real RAG pipeline.

    A record that already has both is left alone, so a dataset can also carry
    answers captured elsewhere and be scored without calling the pipeline.
    """
    # The app's modules resolve chroma_db and .env relative to the working
    # directory, so the pipeline has to run from the project root.
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from ask import PdfChatbot, _page_label, explain_api_error, open_retriever

    class RecordingChatbot(PdfChatbot):
        """PdfChatbot that keeps what it retrieved.

        ask() returns no sources for an answer it refuses, but the judges need
        the chunks the model actually saw either way - a refusal with the
        answer sitting in the context is a different failure from one without.
        """

        def _retrieve(self, queries):
            docs = super()._retrieve(queries)
            self.retrieved = docs
            return docs

    pending = [r for r in records if not ("response" in r and "retrieved_contexts" in r)]
    if not pending:
        return

    retriever = open_retriever()
    # Same pacing as evaluate.py: a trial Cohere key allows 10 calls a minute,
    # and only semantic mode calls Cohere at all.
    if retriever.reranker is not None and retriever.reranker.calls_per_minute is None:
        retriever.reranker.calls_per_minute = 10.0

    for i, record in enumerate(pending, 1):
        print(f"  [{i}/{len(pending)}] asking: {record['user_input']}")
        # A fresh bot per question, so no answer leans on the one before it.
        bot = RecordingChatbot(retriever=retriever, use_bm25=use_bm25)
        bot.retrieved = []
        try:
            answer, _ = bot.ask(record["user_input"])
        except Exception as exc:
            record["pipeline_error"] = explain_api_error(exc)
            record["response"], record["retrieved_contexts"] = "", []
            continue
        record["response"] = answer
        # Labelled with the page, as the model saw them - its answers cite
        # pages, and a judge that cannot see them would call the citation
        # unsupported.
        record["retrieved_contexts"] = [
            f"[page {_page_label(doc)}] {doc.page_content}" for doc in bot.retrieved
        ]


def build_evaluator():
    """The evaluator LLM and embeddings, configured apart from the app's."""
    from openai import AsyncOpenAI
    from ragas.embeddings.base import embedding_factory
    from ragas.llms import llm_factory

    key = (os.getenv("RAGAS_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "No evaluator API key: set RAGAS_OPENAI_API_KEY, or OPENAI_API_KEY in .env"
        )
    # Context precision makes one judge call per retrieved chunk, so a run
    # easily passes a low tokens-per-minute limit (30k on gpt-4o for a new
    # account). The OpenAI client retries a 429 after the wait OpenAI asks for;
    # its default of 2 retries is not enough to ride out a full minute.
    client = AsyncOpenAI(api_key=key, max_retries=int(os.getenv("RAGAS_MAX_RETRIES", "10")))
    # A stronger model than the app's gpt-4o-mini by default, so the pipeline
    # is not being graded by the same model that produced its answers.
    model = os.getenv("RAGAS_EVALUATOR_MODEL", "gpt-4o")
    llm = llm_factory(model, client=client, temperature=0, max_tokens=4096)
    embeddings = embedding_factory(
        "openai",
        model=os.getenv("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small"),
        client=client,
    )
    return model, llm, embeddings


async def score_record(record: dict, llm, embeddings) -> dict:
    """Every metric for one record. A metric that fails is recorded, not raised."""
    from completeness import score_completeness
    from ragas.metrics.collections import (
        AnswerRelevancy,
        ContextPrecisionWithReference,
        ContextRecall,
        Faithfulness,
    )

    q, a, ref = record["user_input"], record["response"], record["reference"]
    ctx = record["retrieved_contexts"]

    async def completeness():
        value, raw, reason = await score_completeness(llm, q, a, ref, ctx)
        record["completeness_raw_0_5"] = raw
        record["completeness_reason"] = reason
        return value

    jobs = {
        "faithfulness": Faithfulness(llm=llm).ascore(
            user_input=q, response=a, retrieved_contexts=ctx
        ),
        "context_precision": ContextPrecisionWithReference(llm=llm).ascore(
            user_input=q, reference=ref, retrieved_contexts=ctx
        ),
        "context_recall": ContextRecall(llm=llm).ascore(
            user_input=q, retrieved_contexts=ctx, reference=ref
        ),
        "answer_relevancy": AnswerRelevancy(llm=llm, embeddings=embeddings).ascore(
            user_input=q, response=a
        ),
        "completeness": completeness(),
    }
    scores, errors = {}, []
    # One metric at a time: all five at once is a burst that trips the rate
    # limit, and a run of five questions is not worth the complexity of pacing.
    for name, job in jobs.items():
        try:
            result = await job
        except Exception as exc:
            scores[name] = math.nan
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        value = getattr(result, "value", result)
        scores[name] = round(float(value), 4) if value is not None else math.nan
    return scores, errors


async def score_all(records: list[dict], llm, embeddings) -> list[tuple[dict, list]]:
    """Score records one after another, in one event loop.

    One loop for the whole run because the async OpenAI client belongs to the
    loop it first ran in; a loop per record would leave it pointing at a closed
    one.
    """
    outcomes = []
    for i, record in enumerate(records, 1):
        if record.get("pipeline_error"):
            outcomes.append(({m: math.nan for m in METRICS}, [record["pipeline_error"]]))
            continue
        print(f"  [{i}/{len(records)}] scoring: {record['user_input']}")
        try:
            outcomes.append(await score_record(record, llm, embeddings))
        except Exception as exc:
            outcomes.append(({m: math.nan for m in METRICS}, [f"{type(exc).__name__}: {exc}"]))
    return outcomes


def write_results(rows: list[dict]) -> tuple[Path, Path]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest = RESULTS_DIR / "ragas_results.csv"
    stamped = RESULTS_DIR / f"ragas_results_{datetime.now():%Y%m%d_%H%M}.csv"
    for path in (latest, stamped):
        # utf-8-sig so Excel shows the euro signs rather than mojibake.
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    return latest, stamped


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="RAGAS evaluation of the RAG pipeline")
    parser.add_argument(
        "--mode", choices=["hybrid", "semantic"], default="hybrid",
        help="search mode to evaluate (the app's two modes); default hybrid",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    args = parser.parse_args()

    try:
        _patch_ragas_import()
        import ragas  # noqa: F401 - fail here, clearly, if it is not installed
    except ImportError as exc:
        print(
            f"ragas is not importable ({exc}).\n"
            "Run this with evaluation\\.venv\\Scripts\\python.exe, set up as in "
            "evaluation/README.md."
        )
        return 2

    try:
        records = load_dataset(args.dataset.resolve())
    except (OSError, ValueError) as exc:
        print(f"Cannot load the evaluation dataset: {exc}")
        return 2

    print(f"{len(records)} questions, search mode: {args.mode}\n\nRunning the RAG pipeline...")
    try:
        run_pipeline(records, use_bm25=args.mode == "hybrid")
        model, llm, embeddings = build_evaluator()
    except Exception as exc:
        # SetupError from ask.py (no key, no vector store) lands here too.
        print(f"Setup failed: {exc}")
        return 2

    print(f"\nScoring with evaluator {model}...")
    try:
        outcomes = asyncio.run(score_all(records, llm, embeddings))
    except Exception as exc:
        outcomes = [({m: math.nan for m in METRICS}, [f"{type(exc).__name__}: {exc}"])] * len(records)

    rows = []
    print()
    for i, (record, (scores, errors)) in enumerate(zip(records, outcomes), 1):
        if record.get("pipeline_error"):
            status = "pipeline_error"
        elif not errors:
            status = "ok"
        else:
            status = "failed" if len(errors) >= len(METRICS) else "partial"
        row = {
            "user_input": record["user_input"],
            "response": record["response"],
            "reference": record["reference"],
            "retrieved_context_count": len(record["retrieved_contexts"]),
            **scores,
            "completeness_raw_0_5": record.get("completeness_raw_0_5", ""),
            "completeness_reason": record.get("completeness_reason", ""),
            "search_mode": args.mode,
            "status": status,
            "errors": " | ".join(errors),
        }
        rows.append(row)
        shown = "  ".join(
            f"{m}={row[m]:.2f}" if not math.isnan(row[m]) else f"{m}=n/a" for m in METRICS
        )
        print(f"  [{i}/{len(records)}] {status:<14} {shown}")

    latest, stamped = write_results(rows)

    print("\nAverages (n/a scores left out):")
    for m in METRICS:
        values = [r[m] for r in rows if not math.isnan(r[m])]
        mean = f"{sum(values) / len(values):.3f}" if values else "n/a"
        print(f"  {m:<18} {mean}   ({len(values)}/{len(rows)} scored)")
    print(f"\nSaved: {latest}\n       {stamped}")
    return 0 if all(r["status"] == "ok" for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
