# chunking.py
"""Semantic chunking: split a page where its meaning changes, not every N characters.

RecursiveCharacterTextSplitter cuts at a fixed length and only prefers a
paragraph or line break near it. On an annual report that repeatedly severs a
sentence mid-clause and, worse, cuts a table between two rows that belong to
the same block of figures - so a chunk arrives holding the tail of one topic
and the head of the next, and matches neither well.

This splitter instead embeds the page in small units, measures how far
consecutive units drift apart in embedding space, and cuts only where that
drift spikes. Rows of one table sit close together and stay together; the
sentence that opens the next section sits far from the row above it and
becomes a boundary.

Two properties matter to the rest of the pipeline and are preserved here:

* every chunk carries `start_index`, its character offset into the page text,
  because ingest.add_table_headers uses it to decide which table header
  governs the chunk;
* a chunk's text is an exact substring of the page, never a re-joined
  approximation, so those offsets stay truthful and the table layout that
  pypdf extracted survives into the store.
"""

from __future__ import annotations

import math
import re

from langchain_core.documents import Document

# Splits a line into sentences. The lookbehind requires whitespace after the
# period, which is what keeps decimals intact: "1.5 million" has no space after
# the point, so it is never a boundary. Table rows rarely contain ". " at all
# and therefore pass through whole.
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9€])")

# Defaults, all overridable per call. They are expressed in characters rather
# than tokens because every other size in this project is - see ingest.py.
BREAKPOINT_PERCENTILE = 92.0  # cut at distances above this percentile
BUFFER_SIZE = 1  # neighbouring units included when embedding a unit
MIN_CHUNK_CHARS = 250
MAX_CHUNK_CHARS = 1200

Span = tuple[int, int]


def _line_spans(text: str) -> list[Span]:
    """(start, end) of every line, as offsets into `text`."""
    spans, offset = [], 0
    for line in text.splitlines(keepends=True):
        spans.append((offset, offset + len(line)))
        offset += len(line)
    return spans


def _trim(text: str, start: int, end: int) -> Span | None:
    """Shrink a span past leading and trailing whitespace, or drop it if blank."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if end > start else None


def unit_spans(text: str) -> list[Span]:
    """The atomic units of a page, in order, as offsets into `text`.

    A unit is a line, or a sentence within a line where the line holds several.
    Lines are the primary unit deliberately: pypdf emits one line per visual
    line, so a table row, a heading and a figure caption each arrive as their
    own line already, and cutting on lines never splits a row of figures down
    the middle.
    """
    spans = []
    for line_start, line_end in _line_spans(text):
        line = text[line_start:line_end]
        cursor = 0
        for match in SENTENCE_END.finditer(line):
            span = _trim(text, line_start + cursor, line_start + match.start())
            if span:
                spans.append(span)
            cursor = match.end()
        span = _trim(text, line_start + cursor, line_end)
        if span:
            spans.append(span)
    return spans


def _context_texts(text: str, spans: list[Span], buffer_size: int) -> list[str]:
    """Each unit widened by its neighbours, which is what actually gets embedded.

    A bare unit here is often a single extracted line - "Turnover 19,374,073" -
    and the embedding of six digits on its own is mostly noise. Embedding it
    together with the lines either side gives the comparison something stable
    to measure, so a boundary reflects a change of subject rather than the
    accident of how one line happened to be extracted.
    """
    texts = []
    for i in range(len(spans)):
        low = max(0, i - buffer_size)
        high = min(len(spans), i + buffer_size + 1)
        texts.append(" ".join(text[s:e] for s, e in spans[low:high]))
    return texts


def _cosine_distance(a: list[float], b: list[float]) -> float:
    """1 - cosine similarity. 0.0 when either vector is degenerate."""
    dot = sum(x * y for x, y in zip(a, b))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return 1.0 - dot / norm if norm else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    """Linear-interpolated percentile, matching numpy's default method.

    Written out rather than imported so this module needs nothing beyond the
    standard library; the input is one page's worth of distances, so the cost
    of sorting it is irrelevant.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _split_oversized(text: str, span: Span, max_chars: int) -> list[Span]:
    """Last-resort character split for a single unit longer than the cap.

    Reached only when one line of the PDF is itself longer than a whole chunk -
    a table extracted without line breaks, most often. Prefers to break at
    whitespace so a number is not cut in half.
    """
    start, end = span
    pieces = []
    while end - start > max_chars:
        cut = text.rfind(" ", start + max_chars // 2, start + max_chars)
        if cut == -1:
            cut = start + max_chars
        pieces.append((start, cut))
        start = cut + 1 if text[cut : cut + 1] == " " else cut
    if end > start:
        pieces.append((start, end))
    return pieces


class SemanticChunker:
    """Splits Documents where their embeddings say the subject changes."""

    def __init__(
        self,
        embeddings,
        breakpoint_percentile: float = BREAKPOINT_PERCENTILE,
        buffer_size: int = BUFFER_SIZE,
        min_chunk_chars: int = MIN_CHUNK_CHARS,
        max_chunk_chars: int = MAX_CHUNK_CHARS,
    ):
        self.embeddings = embeddings
        self.breakpoint_percentile = breakpoint_percentile
        self.buffer_size = buffer_size
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars

    # -- grouping ------------------------------------------------------------

    def _breakpoints(self, distances: list[float]) -> set[int]:
        """Indices after which to cut, i.e. where the drift spikes.

        The threshold is relative to the page, not absolute: a page of dense
        table rows drifts far less between units than a page of prose, and one
        fixed cut-off would leave the first page whole and shred the second. A
        percentile adapts to whatever the page is made of.
        """
        if not distances:
            return set()
        threshold = _percentile(distances, self.breakpoint_percentile)
        return {i for i, distance in enumerate(distances) if distance > threshold}

    def _enforce_max(
        self, group: list[int], spans: list[Span], distances: list[float]
    ) -> list[list[int]]:
        """Split a group that exceeds the cap at its own weakest seam.

        The cap exists because retrieval hands whole chunks to the model: one
        page-long chunk crowds out several others that would have fitted. Where
        a split has to happen, the largest remaining distance inside the group
        is the least damaging place to put it - the same rule that chose the
        boundaries in the first place, applied one level down.
        """
        length = spans[group[-1]][1] - spans[group[0]][0]
        if length <= self.max_chunk_chars or len(group) == 1:
            return [group]

        seam = max(range(group[0], group[-1]), key=lambda i: distances[i])
        left = [i for i in group if i <= seam]
        right = [i for i in group if i > seam]
        return self._enforce_max(left, spans, distances) + self._enforce_max(
            right, spans, distances
        )

    def _merge_undersized(
        self, groups: list[list[int]], spans: list[Span]
    ) -> list[list[int]]:
        """Fold a group too short to stand alone into its neighbour.

        A percentile threshold always fires somewhere, including between two
        lines of one sentence. A three-word chunk retrieves badly and reads as
        a fragment when it reaches the model, so it is put back with the text
        it came from - as long as that does not push the result over the cap.

        Either side being short is enough to merge, not just the group in hand.
        Testing only the current group would strand a short *first* group,
        which has no predecessor to be folded into and would otherwise survive
        as the one fragment this pass exists to remove - and a page's first
        line is exactly where a stray short unit tends to be, because that is
        where the running header sits.
        """
        merged: list[list[int]] = []
        for group in groups:
            length = spans[group[-1]][1] - spans[group[0]][0]
            if merged:
                previous = merged[-1]
                previous_length = spans[previous[-1]][1] - spans[previous[0]][0]
                combined = previous + group
                if (
                    (length < self.min_chunk_chars
                     or previous_length < self.min_chunk_chars)
                    and spans[combined[-1]][1] - spans[combined[0]][0]
                    <= self.max_chunk_chars
                ):
                    merged[-1] = combined
                    continue
            merged.append(group)
        return merged

    def _chunk_spans(
        self, text: str, spans: list[Span], vectors: list[list[float]]
    ) -> list[Span]:
        """The page's chunks, as (start, end) offsets into `text`."""
        if not spans:
            return []
        if len(spans) == 1:
            return _split_oversized(text, spans[0], self.max_chunk_chars)

        distances = [
            _cosine_distance(vectors[i], vectors[i + 1])
            for i in range(len(vectors) - 1)
        ]
        breakpoints = self._breakpoints(distances)

        groups, current = [], [0]
        for i in range(1, len(spans)):
            if i - 1 in breakpoints:
                groups.append(current)
                current = []
            current.append(i)
        groups.append(current)

        capped = [
            piece
            for group in groups
            for piece in self._enforce_max(group, spans, distances)
        ]

        chunks = []
        for group in self._merge_undersized(capped, spans):
            span = (spans[group[0]][0], spans[group[-1]][1])
            if span[1] - span[0] > self.max_chunk_chars:
                # Only reachable when a single unit is longer than the cap:
                # _enforce_max cannot split a group of one.
                chunks.extend(_split_oversized(text, span, self.max_chunk_chars))
            else:
                chunks.append(span)
        return chunks

    # -- public API ----------------------------------------------------------

    def split_documents(self, docs: list[Document]) -> list[Document]:
        """Split every Document, carrying its metadata onto each chunk.

        Every page's units are embedded in one call rather than one call per
        page: the report runs to a couple of hundred pages, and paying a
        network round trip for each of them is most of the wall-clock time of
        an ingest.
        """
        per_doc_spans = [unit_spans(doc.page_content) for doc in docs]

        texts, offsets = [], []
        for doc, spans in zip(docs, per_doc_spans):
            offsets.append(len(texts))
            texts.extend(_context_texts(doc.page_content, spans, self.buffer_size))
        offsets.append(len(texts))

        vectors = self.embeddings.embed_documents(texts) if texts else []

        chunks = []
        for i, (doc, spans) in enumerate(zip(docs, per_doc_spans)):
            page_vectors = vectors[offsets[i] : offsets[i + 1]]
            for start, end in self._chunk_spans(doc.page_content, spans, page_vectors):
                chunks.append(
                    Document(
                        page_content=doc.page_content[start:end],
                        # start_index is what add_table_headers matches against.
                        # The key name is RecursiveCharacterTextSplitter's, kept
                        # so the store's metadata shape does not change.
                        metadata={**doc.metadata, "start_index": start},
                    )
                )
        return chunks
