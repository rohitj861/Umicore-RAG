"""Completeness: a custom LLM-as-a-judge metric.

ragas 0.4.3 has no metric that asks "does the answer cover everything the
question needs?" - answer relevancy checks the answer is on topic, and
faithfulness that it is supported, but an answer can pass both while leaving
out half of a two-part question. So this one is ours.

It uses the same evaluator LLM as the ragas metrics (built in run_ragas.py),
so there is one model and one key to configure. Nothing in the app imports
this file.
"""

from pydantic import BaseModel, Field

RUBRIC = """\
5 = fully complete; covers all important points
4 = mostly complete; a minor detail is missing
3 = partially complete; some important information is missing
2 = major gaps
1 = largely incomplete
0 = does not meaningfully answer the question"""

PROMPT = """\
You are grading whether an answer from a question-answering system over a
company annual report is COMPLETE.

Judge only coverage: does the answer contain every important piece of
information needed to answer the question? The reference answer lists what a
complete answer must contain; the retrieved evidence shows what the system had
available. Do not penalise extra correct detail, citations, or wording
differences. Figures count as present when they state the same amount in a
different unit or format (e.g. € 847 million = 847,000 thousand EUR). A figure
that is present but wrong counts as missing.

Score on this rubric:
{rubric}

Question:
{question}

Reference answer:
{reference}

Retrieved evidence (excerpts):
{evidence}

Answer to grade:
{response}

List the important points the answer is missing (empty if none), then give the
score."""

# Evidence is supporting material for the judge, not the thing being graded.
# The reference already says what a complete answer needs, so a trimmed view of
# the evidence keeps the prompt small without changing the verdict.
MAX_EVIDENCE_CHARS = 12000


class CompletenessVerdict(BaseModel):
    missing_points: list[str] = Field(
        description="Important points the answer leaves out or gets wrong"
    )
    score: int = Field(ge=0, le=5, description="Rubric score, 0 to 5")


async def score_completeness(
    llm, user_input: str, response: str, reference: str, retrieved_contexts: list[str]
) -> tuple[float, int, str]:
    """(normalised 0-1 score, raw 0-5 score, reason). Raises on LLM failure."""
    evidence = "\n---\n".join(retrieved_contexts)[:MAX_EVIDENCE_CHARS] or "(none)"
    verdict = await llm.agenerate(
        PROMPT.format(
            rubric=RUBRIC,
            question=user_input,
            reference=reference,
            evidence=evidence,
            response=response,
        ),
        CompletenessVerdict,
    )
    reason = "; ".join(verdict.missing_points) or "nothing missing"
    # Divided by 5 so it sits on the same 0-1 scale as the ragas metrics.
    return verdict.score / 5, verdict.score, reason
