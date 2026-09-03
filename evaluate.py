# evaluate.py
"""Grade the bot against figures read straight out of the report.

Run this after changing anything that decides what reaches the model - the
splitter, the retrieval settings, the prompt - because none of those changes
announce themselves. A wrong figure looks exactly like a right one.

    python evaluate.py            # both retrieval modes
    python evaluate.py --hybrid   # hybrid only, roughly half the API calls
    python evaluate.py --quick    # the group key figures only

Each case names the figure the report gives and, where the report also prints a
lookalike, the figures that would be wrong. A case passes when the answer
contains the right one and none of the wrong ones - so a plausible answer drawn
from the wrong column, the wrong business group or the wrong statement fails
rather than passing on a keyword.

Ground truth is the Group key figures table on page 18 (full-year columns), the
segment tables on pages 19-24, and the consolidated income statement on page
62. Everything here was read off those pages by hand; re-check them against the
PDF if you point this project at a different report.

Costs one or two chat completions per case per mode - cents, not dollars, but
it is not free.
"""

import re
import sys

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
    """
    out = set()
    for scale in (1, 1_000, 1_000_000):
        scaled = value / scale
        if scaled < 0.5:
            continue
        for places in (0, 1, 2):
            rounded = round(scaled, places)
            if abs(rounded - scaled) > 0.005 * scaled:
                continue
            out.add(f"{scaled:,.{places}f}")
            out.add(f"{scaled:.{places}f}")
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
EURO_RAW = re.compile(
    # Answers write the currency both ways, so both have to be caught.
    r"(?:€|\bEUR)\s*\(?(\d{1,3}(?:,\d{3}){2,}|\d{7,})\)?"
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
    ("What was the adjusted EBIT in 2025?", [579], [383, 296],
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
    ("What was the adjusted net profit, Group share, in 2025?", [288], [384548],
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
    ("What was the EBITDA in 2024?", [1025], [244],
     "FY2024 (1,025), a loss. Wrong: 244 is H2 2024"),
    ("What was the adjusted EBITDA in 2024?", [763], [370],
     "FY2024 763. Wrong: 370 is H2"),
    ("What was the adjusted EBIT in 2024?", [478], [237],
     "FY2024 478. Wrong: 237 is H2"),
    ("What was the R&D expenditure in 2024?", [258, 257555], [126],
     "FY2024 258. Wrong: 126 is H2"),
    ("What was the capital expenditure in 2024?", [555, 554665], [285],
     "FY2024 555. Wrong: 285 is H2"),
    ("What were the revenues in 2024?", [3461], [1657],
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
    passed = total = 0
    failures = []

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

            hit = any(mentions(answer, e) for e in expected)
            miss = [str(w) for w in wrong if mentions(answer, w)]
            units = unit_errors(answer)
            ok = hit and not miss and not units

            total += 1
            passed += ok
            flag = "pass" if ok else "FAIL"
            print(f"  [{label}] {flag}  {answer[:150].replace(chr(10), ' ')}")
            if not ok:
                if miss:
                    reason = f"reported {miss}"
                elif not hit:
                    reason = f"none of {expected}"
                else:
                    reason = f"unit error: {units}"
                print(f"         -> {reason}")
                failures.append((question, label, reason))

    return passed, total, failures


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = set(sys.argv[1:])
    modes = [("hybrid", True)] if "--hybrid" in args else [("hybrid", True), ("vector", False)]
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
        print(f"\n{len(failures)} failure(s):")
        for question, label, reason in failures:
            print(f"  [{label}] {question}\n         {reason}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
