# evaluate.py
"""Grade the bot against figures read straight out of the report.

Run this after changing anything that decides what reaches the model - the
splitter, the retrieval settings, the prompt - because none of those changes
announce themselves. A wrong figure looks exactly like a right one.

    python evaluate.py            # both retrieval modes
    python evaluate.py --hybrid   # hybrid only, roughly half the API calls
    python evaluate.py --semantic # semantic only (Cohere-reranked if keyed)
    python evaluate.py --quick    # the group key figures only

Each case names the figure the report gives and, where the report also prints a
lookalike, the figures that would be wrong. A case passes when the answer
contains the right one and none of the wrong ones - so a plausible answer drawn
from the wrong column, the wrong business group or the wrong statement fails
rather than passing on a keyword.

The right figure is not enough on its own, either. An answer also fails when
the original it gives in brackets is a different amount from the converted
figure, when a page it cites prints none of its figures, or when it credits a
primary statement that does not print the figure. See grade().

Ground truth is the Group key figures table on page 18 (full-year columns), the
segment tables on pages 19-24, and the consolidated income statement on page
62. Everything here was read off those pages by hand; re-check them against the
PDF if you point this project at a different report.

Costs one or two chat completions per case per mode - cents, not dollars, but
it is not free.
"""

import re
import sys
from decimal import ROUND_HALF_UP, Decimal

from ask import PdfChatbot, SetupError, explain_api_error, open_retriever


def spellings(value: float) -> set[str]:
    """Every way the model might legitimately write one figure.

    The report prints thousands; answers are asked to convert, so 771,739
    reaches the user as "771.74 million" or "771.7" and 384,548 as "385". All
    of those are the same figure and must all count as right - matching the
    literal digits of the table instead marks a correct answer wrong, which is
    worse than useless in a file whose whole job is to say what is wrong.

    Scales are exact rather than a tolerance on purpose: 847 (Group adjusted
    EBITDA) and 845.345 (profit before tax of consolidated companies) are 0.2%
    apart, so any tolerance loose enough to accept a rounding is loose enough
    to accept a different line.

    A rounding is only a spelling of the figure while it still identifies it.
    4,346 scaled to millions and rounded to no decimals is "4" - which matches
    inside "4.48 billion" and reported a correct answer as the wrong figure.
    So a form is kept only if it is faithful to within 0.5%, which drops "4"
    (8% out) and "4.3" (1% out) while keeping "4.35".

    Rounding is half-up, the way people and the model write figures. Python's
    own formatting rounds the binary float, so 1,025 million came out as
    "1.02" billion and a correct "€ 1.03 billion" was graded as missing.
    """
    out = set()
    for scale in (1, 1_000, 1_000_000):
        scaled = Decimal(str(value)) / scale
        if scaled < Decimal("0.5"):
            continue
        for places in (0, 1, 2):
            rounded = scaled.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
            if abs(rounded - scaled) > Decimal("0.005") * scaled:
                continue
            out.add(f"{rounded:,.{places}f}")
            out.add(f"{rounded:.{places}f}")
    return out


# A figure this large followed by "million" or "billion" is a table figure that
# was never converted: the report prints thousands, and Umicore's largest line
# is 19,374 million, so anything above six digits called a million overstates
# it a thousandfold. Brackets are allowed for either side because that is how
# the statements write a negative - and how the one measured failure looked,
# "€ (1,424,122) million".
UNCONVERTED = re.compile(
    r"€?\s*\(?(\d{1,3}(?:,\d{3}){2,}|\d{7,})(?:\.\d+)?\)?\s*(million|billion)\b",
    re.I,
)

# A euro amount written as raw statement digits with no scale word at all -
# "€ 19,374,073" - which rule 6 forbids for the same reason.
#
# Any thousands-separated figure counts, not only seven digits and up: "€
# 771,739" is 771,739 thousand read as euros, and a run passed it while failing
# the "€ (1,424,122)" beside it. The report's figures in euros under a million
# are not what any case asks about. A decimal after the digits ("€ 1,357.3
# million") is part of the figure, so the match may not stop short of it.
EURO_RAW = re.compile(
    # Answers write the currency both ways, so both have to be caught.
    r"(?:€|\bEUR)\s*\(?(\d{1,3}(?:,\d{3})+|\d{7,})(?![\d.,]\d)\)?"
    r"(?!\s*\)?\s*(?:thousand|million|billion))",
    re.I,
)


def unit_errors(answer: str) -> list[str]:
    """Figures whose written unit is a thousandfold out.

    Checked separately from the figure itself because the two fail
    independently: an answer can name the right row and still report it a
    thousand times too large, which is the more dangerous of the two - the
    digits look right to anyone spot-checking against the PDF.
    """
    found = [f"{figure} {scale}" for figure, scale in UNCONVERTED.findall(answer)]
    found += [f"EUR {figure} with no scale word" for figure in EURO_RAW.findall(answer)]
    return found


def mentions(answer: str, wanted) -> bool:
    """Whether `answer` states this figure, in any of its spellings.

    A string is matched literally; a number is matched against its spellings,
    each bounded so that "385" does not match inside "1,385,000" or "384.55".
    """
    if isinstance(wanted, str):
        return wanted in answer

    return any(
        re.search(r"(?<![\d.,])" + re.escape(form) + r"(?![\d])", answer)
        for form in spellings(wanted)
    )


# --- Checks on what surrounds the figure ---------------------------------------
#
# An answer can state the right figure and still be wrong about it. Each case
# below passed the checks above in a measured run:
#
#   "€ 385 million (389,501 thousand EUR)" - the headline is Group share, the
#       bracket is total profit including minorities, a different row;
#   "€ -1.03 billion (1,025 thousand EUR)" - the bracket is a thousandfold out;
#   "adjusted EBITDA ... € 847 million ... (page 6)" - page 6 does not print it;
#   "gearing ratio ... from the consolidated balance sheet" - it is a key
#       figure, and the balance sheet does not print it.
#
# The first two are caught by comparing the two halves of the pair, the last
# two by looking at the pages the answer credits.

NUMBER = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
SCALES = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "mn": 1e6, "bn": 1e9}

# An amount written with its scale word, in an answer or on a page: "€ 3.56
# billion", "€ 3.6 bn", "847 million".
STATED_AMOUNT = re.compile(
    rf"(?<![\d.,])(?P<n>{NUMBER})\s*(?P<s>thousand|million|billion|mn|bn)\b", re.I
)

# A converted figure followed by the original in brackets, as rule 6 of the
# prompt asks for: "€ 19.37 billion (19,374,073 thousand EUR)". Signs and
# accounting brackets are allowed on either side; only magnitudes are compared.
FIGURE_PAIR = re.compile(
    rf"(?:€|\bEUR)\s*[-−–]?\s*\(?(?P<a>{NUMBER})\)?\s*(?P<a_scale>thousand|million|billion)\b"
    rf"(?:\s*(?:EUR|€))?\s*"
    rf"\(\s*[-−–]?\s*\(?(?P<b>{NUMBER})\)?\s*(?P<b_scale>thousand|million|billion)\b[^)]*\)",
    re.I,
)


def _amount(number: str, scale: str) -> tuple[float, float]:
    """(value in EUR, how far the written rounding lets it be from the truth)."""
    decimals = len(number.split(".")[1]) if "." in number else 0
    multiplier = SCALES[scale.lower()]
    return (
        float(number.replace(",", "")) * multiplier,
        0.5 * 10 ** -decimals * multiplier,
    )


def stated_amounts(text: str) -> list[tuple[float, float]]:
    """Every amount written with a scale word, as (EUR, rounding allowance)."""
    return [_amount(match["n"], match["s"]) for match in STATED_AMOUNT.finditer(text)]


def invented_originals(answer: str, page_text: dict[int, str]) -> list[str]:
    """Bracketed originals that the report does not print anywhere.

    A bracket presents its digits as the report's own, so they must be. The
    model has written "€ 763 million (763,000 thousand EUR)" beside a key
    figure printed in millions - the right amount, so pair_errors passes it,
    but 763,000 is on no page of the report.
    """
    everything = "\n".join(page_text.values())
    return [
        match.group(0)
        for match in FIGURE_PAIR.finditer(answer)
        if not prints(everything, {match["b"]})
    ]


def pair_errors(answer: str) -> list[str]:
    """Converted figures whose bracketed original is a different amount.

    Two ways this goes wrong, and one check covers both: the bracket carries
    another row's figure, or it carries the right digits under the wrong scale
    word. Either way the two halves stop describing the same amount.

    The allowance is the rounding each side is written to - "€ 385 million"
    may stand for anything from 384.5 to 385.5 million - so a correct pair
    always passes and a pair that is off by more than its own precision does
    not. 389,501 thousand is 4.5 million away from "385 million", and 1,025
    thousand is a billion away from "1.03 billion".
    """
    found = []
    for match in FIGURE_PAIR.finditer(answer):
        headline, headline_slack = _amount(match["a"], match["a_scale"])
        original, original_slack = _amount(match["b"], match["b_scale"])
        slack = headline_slack + original_slack
        if abs(headline - original) > slack * (1 + 1e-9):
            found.append(match.group(0))
    return found


# Where the primary statements are, from the PDF's own bookmarks. Their names
# cannot be looked for in the page text instead: pages 61-67 all carry every
# statement's name in their navigation, so the text would vouch for any of
# them.
STATEMENT_PAGES = {
    "consolidated income statement": 62,
    "consolidated statement of comprehensive income": 63,
    "consolidated balance sheet": 64,
    "consolidated statement of changes in equity": 65,
    "consolidated statement of cash flows": 66,
    "consolidated cash flow statement": 66,
}
STATEMENT = re.compile(
    "|".join(re.escape(name) for name in STATEMENT_PAGES), re.I
)

# "(page 18)", "pages 19-24", "pages 62 and 90". A page number is at most three
# digits and never runs straight into a fourth, so "page 62, 2025" reads as
# page 62 alone.
CITATION = re.compile(
    r"\bpages?\s+((?:\d{1,3}(?!\d)(?:\s*(?:,|and|&|[-–]|to)\s*)?)+)", re.I
)
CITED_PAGE = re.compile(r"(\d{1,3})(?:\s*(?:[-–]|to)\s*(\d{1,3}))?")

# Figures as the answer writes them, for crediting a page with a figure the
# case did not ask about. Years are left out: every page of this report prints
# 2025.
ANSWER_FIGURE = re.compile(r"(?<![\d.,])(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+\.\d+|\d{3,})(?![\d])")
YEAR = re.compile(r"(?:19|20)\d{2}")

# A statement named only to set it aside - "not the consolidated income
# statement", "unlike the balance sheet" - is not a claim about the source.
NEGATION = re.compile(r"\b(?:not|unlike|rather than|instead of|differs? from)\b", re.I)


def printed_forms(expected: list) -> set[str]:
    """How the report prints each expected figure: 384548 -> "384,548"."""
    return {
        wanted if isinstance(wanted, str) else f"{wanted:,}"
        for wanted in expected
    }


def prints(text: str, forms: set[str]) -> bool:
    """Whether a page prints any of these figures, as a whole number.

    Bounded both sides, and a trailing decimal counts as a different number:
    "847" is not printed by "1,847" or by "847.3".
    """
    return any(
        re.search(r"(?<![\d,.])" + re.escape(form) + r"(?![\d]|[.,]\d)", text)
        for form in forms
    )


def cited_pages(answer: str) -> list[range]:
    """Every page citation in the answer, a single page as a range of one."""
    cited = []
    for citation in CITATION.finditer(answer):
        for first, last in CITED_PAGE.findall(citation.group(1)):
            low = int(first)
            high = int(last) if last else low
            if low <= high <= low + 20:  # a wider span is not a citation
                cited.append(range(low, high + 1))
    return cited


def citation_errors(answer: str, expected: list, page_text: dict[int, str]) -> list[str]:
    """Cited pages that print neither the figure asked for nor any the answer gives.

    A page passes on any figure the answer states, not only the one the case
    asked about: an answer may cite a second page for a comparison figure, and
    that citation is right. What fails is a page that prints nothing the answer
    says - which is what "(page 6)" for adjusted EBITDA was. A range passes if
    any page in it prints one.

    A converted figure ("€ 1.42 billion") is never printed as such in a table,
    so the case's own figures are looked for too, as the report prints them.
    And a page also passes if it states the answer's amount at its own
    rounding: page 15 prints revenues as "€ 3.6 bn", which supports an answer
    of "€ 3.56 billion (page 15)". Two amounts agree when they are within the
    rounding each is written to, the allowance pair_errors uses.
    """
    cited = cited_pages(answer)
    if not cited:
        return []

    page_numbers = {str(page) for pages in cited for page in pages}
    forms = printed_forms(expected) | {
        figure
        for figure in ANSWER_FIGURE.findall(answer)
        if not YEAR.fullmatch(figure) and figure not in page_numbers
    }
    answer_amounts = stated_amounts(answer)

    def supports(text: str) -> bool:
        if prints(text, forms):
            return True
        return any(
            abs(page_value - value) <= (page_slack + slack) * (1 + 1e-9)
            for page_value, page_slack in stated_amounts(text)
            for value, slack in answer_amounts
        )

    found = []
    for pages in cited:
        if not any(supports(page_text.get(page, "")) for page in pages):
            label = f"{pages.start}-{pages.stop - 1}" if len(pages) > 1 else str(pages.start)
            found.append(f"page {label}")
    return found


def attribution_errors(answer: str, expected: list, page_text: dict[int, str]) -> list[str]:
    """Primary statements named as a source that do not print the figure.

    Stricter than citation_errors on purpose: the named statement must print
    one of the case's own figures, not just any figure in the answer. Calling
    a key figure "from the consolidated income statement" is the claim being
    tested, and the income statement prints hundreds of numbers - letting any
    of them vouch for it would pass almost everything.
    """
    forms = printed_forms(expected)
    found = []
    for sentence in re.split(r"(?<=[.!?])\s+|\n", answer):
        for match in STATEMENT.finditer(sentence):
            if NEGATION.search(sentence[: match.start()]):
                continue
            name = match.group(0).lower()
            if not prints(page_text.get(STATEMENT_PAGES[name], ""), forms):
                found.append(f"{name} (page {STATEMENT_PAGES[name]})")
    return list(dict.fromkeys(found))


def page_texts(retriever) -> dict[int, str]:
    """All of each page's chunk text, keyed by the page number answers cite.

    Taken from the store rather than the PDF, which a clone may not have. The
    table headers ingest.py prepends to chunks carry units and years, not
    figures, so they cannot vouch for a page that does not print the figure.
    """
    pages: dict[int, list[str]] = {}
    for doc in retriever.bm25.documents:
        page = doc.metadata.get("page")
        if isinstance(page, int):
            pages.setdefault(page + 1, []).append(doc.page_content)
    return {page: "\n".join(texts) for page, texts in pages.items()}


def grade(answer: str, expected: list, wrong: list, page_text: dict[int, str]) -> list[str]:
    """Every problem with one answer, each prefixed with its kind. Empty = pass.

    All checks run, not just the first to fail, so a report shows an answer
    that has the wrong figure *and* cites the wrong page as both.
    """
    problems = []

    if not any(mentions(answer, e) for e in expected):
        problems.append(f"missing: none of {expected}")
    miss = [str(w) for w in wrong if mentions(answer, w)]
    if miss:
        problems.append(f"wrong figure: reported {miss}")
    for error in unit_errors(answer):
        problems.append(f"unit: {error}")
    for pair in pair_errors(answer):
        problems.append(f"bracket: {pair!r} states two different amounts")
    for pair in invented_originals(answer, page_text):
        problems.append(f"bracket: {pair!r} gives an original the report does not print")
    for page in citation_errors(answer, expected, page_text):
        problems.append(f"citation: {page} prints none of the figures given")
    for statement in attribution_errors(answer, expected, page_text):
        problems.append(f"attribution: {statement} does not print {sorted(printed_forms(expected))}")

    return problems

# (question, right figures, figures that would be wrong, note)
#
# Figures are given as the report prints them; `spellings` handles the unit
# conversions. Percentages stay strings - there is no scale to convert.
#
# The wrong-figure lists are the point of this file. "Adjusted EBITDA" alone
# appears five times in the report with five different values; an answer of
# "450" is not a near miss, it is Catalysis reported as the Group.
GROUP_KEY_FIGURES = [
    ("What was the adjusted EBITDA in 2025?", [847], [450, 371, 108],
     "Group, page 18. Wrong: Catalysis 450, Recycling 371, Specialty 108"),
    # The thousands forms in this case and the two marked below are printed on
    # pages 87, 91 and 86, which semantic-only search cites; without them a
    # right citation there failed the citation check.
    ("What was the adjusted EBIT in 2025?", [579, 579280], [383, 296],
     "Group 579. Wrong: Catalysis 383, Recycling 296"),
    ("What was the turnover in 2025?", [19374, 19374073], [4482, 13826],
     "Group 19,374. Wrong: Catalysis 4,482, Recycling 13,826"),
    # No wrong-list: an answer that gives the Group figure and then breaks it
    # down by business group is right, not wrong, so the segment numbers
    # appearing in the text prove nothing on their own. This case can only
    # check that the Group figure is present.
    ("What were the revenues in 2025?", [3562, 3562474], [],
     "Group revenues excluding metal 3,562"),
    ("What was the adjusted EBITDA margin in 2025?", ["24.0", "24%"], ["27.0", "39.2"],
     "Group 24.0%. Wrong: Catalysis 27.0%, Recycling 39.2%"),
    ("What was the net profit, Group share, in 2025?", [385, 384548], [288],
     "385. Wrong: 288 is ADJUSTED net profit"),
    ("What was the adjusted net profit, Group share, in 2025?", [288, 288000], [384548],  # thousands: p91
     "288. Wrong: 385 is unadjusted"),
    ("What was the R&D expenditure in 2025?", [206, 205702], [86, 74],
     "Group 206. Wrong: segment figures"),
    ("What was the effective adjusted tax rate in 2025?", ["26.1"], ["29.4", "20.6"],
     "FY2025 26.1%. Wrong: 29.4% is FY2024, 20.6% is H2"),
    ("What was the gearing ratio at the end of 2025?", ["37.4"], ["42.6"],
     "37.4%. Wrong: 42.6% is 2024"),
    ("What was the consolidated net financial debt at the end of 2025?",
     [1357], [1425], "1,357. Wrong: 1,425 is 2024"),
    ("What was the return on capital employed in 2025?", ["15.7", "15.67"], ["12.3", "12.31"],
     "15.7% (15.67% in note F32). Wrong: 12.3% is 2024"),
]

# The key figures tables print H2 before the full year, so the full-year column
# is third and fourth, not first and second. Asking about 2024 is what exposes
# this: for 2025 the wanted column is last and hard to get wrong. Every wrong
# figure here is that row's H2 2024 column.
FULL_YEAR_VS_HALF_YEAR = [
    ("What was the EBITDA in 2024?", [1025, 1025321], [244],  # thousands: p86
     "FY2024 (1,025), a loss. Wrong: 244 is H2 2024"),
    ("What was the adjusted EBITDA in 2024?", [763], [370],
     "FY2024 763. Wrong: 370 is H2"),
    ("What was the adjusted EBIT in 2024?", [478], [237],
     "FY2024 478. Wrong: 237 is H2"),
    ("What was the R&D expenditure in 2024?", [258, 257555], [126],
     "FY2024 258. Wrong: 126 is H2"),
    ("What was the capital expenditure in 2024?", [555, 554665], [285],
     "FY2024 555. Wrong: 285 is H2"),
    # 3,461,000 too: the 2024 segment table (page 86) prints it in thousands,
    # and citing that page is right.
    ("What were the revenues in 2024?", [3461, 3461000], [1657],
     "FY2024 3,461. Wrong: 1,657 is H2"),
]

# Same row label, different scope. These are the ones that go wrong silently.
SCOPE = [
    ("What was Catalysis adjusted EBITDA in 2025?", [450], [847],
     "Catalysis 450, not the Group's 847"),
    ("What was Recycling adjusted EBITDA in 2025?", [371], [847],
     "Recycling 371"),
    ("What was Recycling turnover in 2025?", [13826, 13826338], [19374, 19374073],
     "Recycling 13,826, not the Group's 19,374"),
    # Both years are required. Asked for two figures, an answer can get one
    # right and take the other off the wrong row - which is what vector-only
    # did here, pairing 2025's pre-tax profit with 2024's post-tax loss.
    ("What was the profit before income tax in 2025 and 2024?",
     [771739], [845345, 1531076],
     "p62: 2025 = 771,739, 2024 = (1,424,122). Wrong: 845,345 is note F13 "
     "(consolidated companies); 1,531,076 is the POST-tax 2024 loss"),
    ("What was the profit before income tax in 2024?",
     [1424122], [1531076, 1375542],
     "p62 2024 = (1,424,122). Wrong: 1,531,076 is post-tax, 1,375,542 is F13"),
    ("What was the turnover in 2024?", [14853681, 14854], [18849795],
     "p62/p90. Wrong: 18,849,795 is the 2025 ADJUSTED column"),
    ("What was Catalysis turnover for the full year 2025?", [4482], [2178, 4346],
     "FY 4,482. Wrong: 2,178 is H2, 4,346 is FY2024"),
]

# Prose figures, and the one question the report cannot answer.
OTHER = [
    ("How many employees does the group have?", [11230], [2094],
     "11,230 fully consolidated. Wrong: 2,094 is associates/JVs"),
    ("What were the total R&D expenditures in 2025?", [205702], [],
     "p91, thousands of EUR"),
    ("What is Umicore's policy on cryptocurrency mining?",
     ["I don't know about this."], [], "must refuse - not in the report"),
]


def run(cases: list, modes: list[tuple[str, bool]]) -> tuple[int, int, list]:
    retriever = open_retriever()
    # Pace Cohere so a trial key's 10 calls a minute is never exceeded: an
    # answer that falls back to embedding order is not a reranked answer, and
    # grading it as one measures the wrong thing. COHERE_CALLS_PER_MINUTE
    # raises the pace for a production key. Only semantic-only runs call Cohere.
    reranker = retriever.reranker
    if reranker is not None and reranker.calls_per_minute is None:
        reranker.calls_per_minute = 10.0
    page_text = page_texts(retriever)
    passed = total = 0
    failures = []
    # Semantic-only answers that were not reranked - no key, or Cohere failed
    # and the search fell back to embedding order. Counted and shown, because
    # a run full of them grades plain vector search, not the reranked mode.
    not_reranked = 0

    for question, expected, wrong, note in cases:
        print(f"\n{question}")
        print(f"  ({note})")
        for label, use_bm25 in modes:
            # A fresh bot per case: conversation memory would let one answer
            # steer the next, which is not what is being measured here.
            bot = PdfChatbot(retriever=retriever, use_bm25=use_bm25)
            try:
                answer, _ = bot.ask(question)
            except Exception as exc:
                print(f"  [{label}] ERROR {explain_api_error(exc)}")
                failures.append((question, label, "error"))
                total += 1
                continue

            problems = grade(answer, expected, wrong, page_text)
            ok = not problems

            total += 1
            passed += ok
            flag = "pass" if ok else "FAIL"
            # In full on a failure: the problem is often in the citation or
            # the bracket at the end, which a truncated line cuts off.
            shown = answer if problems else answer[:150]
            print(f"  [{label}] {flag}  {shown.replace(chr(10), ' ')}")
            for problem in problems:
                print(f"         -> {problem}")
                failures.append((question, label, problem))

            status = bot.last_search.get("rerank")
            if status and not status.startswith("reranked"):
                not_reranked += 1
                print(f"         (not reranked - {status})")

    if any(not use_bm25 for _, use_bm25 in modes) and reranker is not None:
        print(
            f"\nCohere rerank ({reranker.model}): {reranker.calls} calls, "
            f"{reranker.failures} failed; {not_reranked} semantic-only "
            "answers were NOT reranked"
            + ("" if reranker.enabled else " - COHERE_API_KEY is not set")
        )

    return passed, total, failures


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = set(sys.argv[1:])
    if "--hybrid" in args:
        modes = [("hybrid", True)]
    elif "--semantic" in args:
        modes = [("semantic", False)]
    else:
        modes = [("hybrid", True), ("semantic", False)]
    cases = (GROUP_KEY_FIGURES if "--quick" in args
             else GROUP_KEY_FIGURES + FULL_YEAR_VS_HALF_YEAR + SCOPE + OTHER)

    print(f"{len(cases)} cases x {len(modes)} mode(s) = {len(cases) * len(modes)} questions\n")

    try:
        passed, total, failures = run(cases, modes)
    except SetupError as exc:
        sys.exit(str(exc))

    print("\n" + "=" * 70)
    print(f"{passed}/{total} passed")
    if failures:
        # By kind first: a wrong figure and a wrong page citation are both
        # failures, but not equally bad, and a run that only broke citations
        # should read that way at a glance.
        kinds: dict[str, int] = {}
        for _, _, problem in failures:
            kind = problem.split(":", 1)[0]
            kinds[kind] = kinds.get(kind, 0) + 1
        print("problems by kind: " + ", ".join(f"{k} {n}" for k, n in kinds.items()))

        print(f"\n{len(failures)} problem(s):")
        for question, label, reason in failures:
            print(f"  [{label}] {question}\n         {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
