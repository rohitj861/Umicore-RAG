# citations.py
"""Check each page citation in an answer against the excerpts it was written from.

The prompt asks the model to cite the page of the excerpt it took each figure
from, and gpt-4o-mini mostly does. Where it does not, the error is stubborn
rather than random: asked for 2024 turnover it gave the right figure, € 14.85
billion, and cited page 149 in every graded run - an EU Taxonomy table whose
"Turnover" row sits beside the words "previous financial year (2024)" but
prints 2025's figure. A prompt rule telling it to check the digits changed
nothing.

So the check is made in code, where it is not a matter of following
instructions. For each "(page N)" the answer writes, the figures in front of it
are looked for in the excerpt from page N. If that excerpt does not print them
and another excerpt the model was given does, the citation is moved there. In
every other case it is left exactly as written: nothing is removed, nothing is
guessed, and a citation that cannot be checked is not touched.

evaluate.py grades citations with its own, separate implementation, so a
mistake here shows up as a failing case rather than being shared by the code
under test and the test.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document

# A figure as the answer or a page writes it: "19,374,073", "14.85", "847".
NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "mn": 1e6, "bn": 1e9}

# The citation form rule 11 of the prompt asks for. Other forms ("pages 19-24")
# are left alone rather than half-understood.
CITATION = re.compile(r"\(page (\d{1,3})\)")

# What a citation covers starts after the previous citation or at the start of
# its sentence. A decimal point is never followed by whitespace, so "€ 14.85
# billion" does not end a sentence.
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n")

AMOUNT = re.compile(
    rf"(?<![\d.,])(?P<n>{NUMBER})\s*(?P<s>thousand|million|billion|mn|bn)\b", re.I
)
PERCENT = re.compile(r"(?<![\d.,])(\d+(?:\.\d+)?)\s*%")
PLAIN = re.compile(rf"(?<![\d.,])({NUMBER})(?![\d])")

# Every page of this report prints 2024 and 2025. A year matched as a figure
# would let almost any page "support" almost any answer.
YEAR = re.compile(r"(?:19|20)\d{2}")

# How the report prints table figures: in thousands of EUR, in millions, or -
# in the taxonomy tables - in euros.
TABLE_UNITS = (1e3, 1e6, 1.0)


def _value(number: str) -> float:
    return float(number.replace(",", ""))


def _slack(number: str, multiplier: float) -> float:
    """How far a written figure may be from the true one: its own rounding."""
    decimals = len(number.split(".")[1]) if "." in number else 0
    return 0.5 * 10 ** -decimals * multiplier


def _figures(segment: str) -> tuple[list[tuple[float, float]], set[str]]:
    """The figures a stretch of answer text states.

    Returns amounts written with a scale word, as (EUR, rounding allowance),
    and every other figure as the literal digits to look for.
    """
    amounts = []
    for match in AMOUNT.finditer(segment):
        multiplier = SCALES[match["s"].lower()]
        amounts.append((_value(match["n"]) * multiplier, _slack(match["n"], multiplier)))

    literals = set(PERCENT.findall(segment))
    covered = [m.span() for m in AMOUNT.finditer(segment)]
    for match in PLAIN.finditer(segment):
        if any(start <= match.start() < end for start, end in covered):
            continue  # already read as part of an amount
        token = match.group(1)
        if YEAR.fullmatch(token):
            continue
        # A bare short number ("in 3 segments") identifies nothing.
        if "," in token or "." in token or len(token) >= 3:
            literals.add(token)
    return amounts, literals


def _prints(text: str, token: str) -> bool:
    """Whether `text` prints this figure as a whole number, not inside another."""
    return bool(
        re.search(r"(?<![\d,.])" + re.escape(token) + r"(?![\d]|[.,]\d)", text)
    )


# How strongly an excerpt supports a citation.
NONE, ROUNDED, EXACT = 0, 1, 2


def support(text: str, amounts: list[tuple[float, float]], literals: set[str]) -> int:
    """How well an excerpt prints the figures a citation covers.

    EXACT: it prints a literal's digits, or a number that agrees with an
    amount and is written at least as precisely - "14,853,681" in a table in
    thousands for "€ 14.85 billion", "(1,025)" in a table in millions for
    "€ 1.03 billion".

    ROUNDED: it only states the amount more coarsely than the answer does -
    "€ 3.6 bn" for "€ 3.56 billion". That still supports a citation the model
    made, but it is weak evidence for choosing a page: "€ 1.03 billion" of
    2024 EBITDA also agrees with "€ 1.0 billion", which on page 12 is a 2028
    target. Measured on saved answers, taking the first such page moved a
    citation there instead of to page 18, which prints (1,025).

    Two figures agree when they are within the rounding each is written to.
    """
    if any(_prints(text, token) for token in literals):
        return EXACT
    if not amounts:
        return NONE

    printed = []
    for match in AMOUNT.finditer(text):
        multiplier = SCALES[match["s"].lower()]
        printed.append((_value(match["n"]) * multiplier, _slack(match["n"], multiplier)))
    for match in PLAIN.finditer(text):
        token = match.group(1)
        if YEAR.fullmatch(token):
            continue
        for unit in TABLE_UNITS:
            printed.append((_value(token) * unit, _slack(token, unit)))

    best = NONE
    for page_value, page_slack in printed:
        if not page_value:
            continue
        for value, slack in amounts:
            if abs(page_value - value) > (page_slack + slack) * (1 + 1e-9):
                continue
            if page_slack <= slack * (1 + 1e-9):
                return EXACT
            best = ROUNDED
    return best


def _page_texts(docs: list[Document]) -> dict[int, str]:
    """Each retrieved page's excerpts joined, keyed by the page answers cite."""
    pages: dict[int, list[str]] = {}
    for doc in docs:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.setdefault(page + 1, []).append(doc.page_content)
    return {page: "\n".join(texts) for page, texts in pages.items()}


def correct_citations(answer: str, docs: list[Document]) -> tuple[str, list[tuple[int, int]]]:
    """The answer with unsupported page citations moved to a supporting page.

    Returns the answer and the (cited, corrected) page pairs that changed.

    A citation is kept if its page supports the figure at all, even only
    ROUNDED. It is moved only if its page does not support it, and then to
    the page with the strongest support - EXACT before ROUNDED - taking the
    earliest in retrieval order, the order the model was shown them in, when
    several are equally strong.
    """
    pages = _page_texts(docs)
    order = list(dict.fromkeys(page for page in (
        doc.metadata.get("page") for doc in docs
    ) if isinstance(page, int)))
    order = [page + 1 for page in order]

    changes: list[tuple[int, int]] = []
    pieces: list[str] = []
    position = 0
    for citation in CITATION.finditer(answer):
        start = position
        for boundary in SENTENCE_BREAK.finditer(answer, position, citation.start()):
            start = boundary.end()
        amounts, literals = _figures(answer[start : citation.start()])

        cited = int(citation.group(1))
        replacement = cited
        if (amounts or literals) and not support(pages.get(cited, ""), amounts, literals):
            strongest = NONE
            for page in order:
                strength = support(pages[page], amounts, literals)
                if strength > strongest:
                    strongest, replacement = strength, page
                    if strength == EXACT:
                        break

        pieces.append(answer[position : citation.start()])
        if replacement != cited:
            changes.append((cited, replacement))
            pieces.append(f"(page {replacement})")
        else:
            pieces.append(citation.group(0))
        position = citation.end()

    pieces.append(answer[position:])
    return "".join(pieces), changes
