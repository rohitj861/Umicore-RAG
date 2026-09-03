# retrieval.py
"""Hybrid retrieval: dense vector search and BM25, fused with reciprocal rank fusion.

Vector search matches meaning and is weakest where meaning is thinnest. A bare
figure is the clearest case: measured on this report, a search for "19,374,073"
returns the employee-numbers table first, because six digits give an embedding
almost nothing to match on. BM25 finds the pages that literally print that
figure. BM25 has the opposite failure - it needs the question to share wording
with the report, and returns nothing for a paraphrase.

Both retrievers already handle a well-phrased question well, and on this report
they usually agree on the answer; what fusion buys is the tail, where one of
them has nothing useful to offer.

Running both and fusing the two rankings is what covers both cases. The fusion
is reciprocal rank fusion (Cormack, Clarke & Buettcher, 2009):

    score(d) = sum over rankings r of  weight_r / (K + rank_r(d))

Rank, not score, is what it adds up - which is the point. A cosine similarity
of 0.83 and a BM25 score of 11.4 are not comparable quantities and normalising
them requires a calibration that changes with every query; positions in a list
are comparable without any. A document ranked well by both retrievers finishes
above one ranked first by a single retriever and nowhere by the other, so the
agreement between two different notions of relevance is what gets rewarded.
"""

from __future__ import annotations

import re

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

# Chunks fetched per retriever, per search query. This is the size of the
# candidate pool, not of the context: ask.MAX_CONTEXT_CHUNKS caps what actually
# reaches the model, so raising it buys better candidates to fuse without
# spending a single extra prompt token.
#
# Raised from 10 after a miss that 10 could not have caught. The group revenue
# row on page 18 ranks 10th-12th for its own best query in both retrievers, so
# at k=10 neither returned it and fusion had nothing to rerank - the model
# never saw the figure and answered with turnover instead. At 16 both page 18
# and page 87 (the same figure in thousands) come in. Higher is not better:
# at 24 page 87 is crowded back out.
TOP_K = 16
RRF_K = 60  # the K above: the rank at which a hit is worth half of rank 0

# BM25 is weighted slightly higher, and only to settle ties. Where the two
# retrievers return no chunk in common - which is what an exact-string query
# like "19,374,073" produces, because a bare figure carries almost no meaning
# for an embedding to match on - every rank-1 candidate scores identically, and
# equal weights break that tie by list order, handing first place to whichever
# ranking was fused first. That is arbitrary, and it was reliably wrong: on
# "19,374,073" the top three chunks were the employee-numbers table, against
# pages 87, 91 and 88 (segment turnover, the income statement, geographical
# turnover) at 1.2. Measured across the graded questions, 1.2 changed no answer
# for the worse and improved one. Higher values change nothing further - the
# tie is already broken - so this is deliberately the smallest value that works
# rather than a claim that keywords matter 20% more than meaning.
#
# These two are the lever to reach for when answers miss in a way that is
# characteristically one retriever's fault: keyword misses argue for more BM25,
# paraphrase misses for more vector.
VECTOR_WEIGHT = 1.0
BM25_WEIGHT = 1.2

# Most chunks any one page may occupy in the context.
#
# Without a cap a page that discusses a figure at length beats the page that
# authoritatively reports it. Asked for profit before income tax, note F13
# (Income taxes, page 95) took three of sixteen slots and the consolidated
# income statement that states the figure was pushed to ninth - or out
# entirely - and the model answered with the note's differently-scoped
# subtotal, 845,345 instead of 771,739. Both retrievers agreed on the note,
# so this is not something fusion can fix: it is what both were asked for.
#
# 2 measured better than both 1 and no cap. At 1 a table split across several
# chunks loses its own rows, which cost more than the crowding did.
MAX_PER_PAGE = 2

# A token is a word or a figure, where a figure keeps its separators:
# "19,374,073" and "1.5" stay whole rather than becoming "19", "374", "073".
# That matters here because the answers this bot exists to find are figures.
TOKEN = re.compile(r"[a-z0-9€]+(?:[.,][0-9]+)*")

# Removed before indexing. Deliberately short: it covers the words that appear
# in most sentences of any English document and so carry no signal, and stops
# there. Domain words are never stopwords - "report", "year" and "group" all
# discriminate between pages of an annual report.
STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the "
    "to was were what when where which who will with".split()
)


def tokenize(text: str) -> list[str]:
    """Words and figures of `text`, lowercased, stopwords dropped.

    A separated figure is emitted twice, once as written and once bare, so that
    a question typed as "19374073" still matches the row that prints
    "19,374,073". The duplicate costs a little in BM25's length normalisation
    and buys back the more likely of the two lookups.
    """
    tokens = []
    for match in TOKEN.finditer(text.lower()):
        token = match.group()
        if token in STOPWORDS:
            continue
        tokens.append(token)
        bare = token.replace(",", "")
        if bare != token:
            tokens.append(bare)
    return tokens


def doc_key(doc: Document) -> tuple:
    """Identity of a chunk, for de-duplicating hits across searches.

    Page and start offset together pin a chunk to one place in one PDF, which
    is what makes this stable across the two retrievers: BM25 rebuilds its
    Documents from the store's raw records, so object identity is no use and
    the text alone would merge two genuinely different chunks that happen to
    repeat a boilerplate line.
    """
    return (
        doc.metadata.get("source"),
        doc.metadata.get("page"),
        doc.metadata.get("start_index"),
    )


class BM25Index:
    """Lexical search over the chunks, held in memory.

    Built from the Chroma store rather than persisted next to it, so there is
    one copy of the corpus and no way for the two indexes to fall out of step
    after a re-ingest. A report's worth of chunks is a few megabytes of text;
    the index takes a second or so to build at start-up and none thereafter.
    """

    def __init__(self, documents: list[Document]):
        self.documents = documents
        # BM25Okapi divides by the corpus average document length, so it cannot
        # be constructed over an empty corpus at all - hence the None.
        self.bm25 = (
            BM25Okapi([tokenize(doc.page_content) for doc in documents])
            if documents
            else None
        )

    @classmethod
    def from_chroma(cls, vectordb) -> "BM25Index":
        record = vectordb.get(include=["documents", "metadatas"])
        documents = [
            Document(page_content=text, metadata=metadata or {})
            for text, metadata in zip(
                record.get("documents") or [], record.get("metadatas") or []
            )
        ]
        return cls(documents)

    def __len__(self) -> int:
        return len(self.documents)

    def search(self, query: str, k: int = TOP_K) -> list[Document]:
        """The k best-scoring chunks for `query`, best first.

        Zero-scoring chunks are dropped rather than padded in: a chunk sharing
        no term with the query is not a weak match, it is a non-match, and
        passing it on would give it a rank - and therefore fusion score - it
        has not earned.
        """
        tokens = tokenize(query)
        if not self.bm25 or not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.documents[i] for i in ranked[:k] if scores[i] > 0]


def reciprocal_rank_fusion(
    rankings: list[list[Document]],
    weights: list[float] | None = None,
    rrf_k: int = RRF_K,
) -> list[Document]:
    """Merge several ranked lists into one, best first.

    Ties keep the order in which documents were first seen, so the same inputs
    always produce the same output - two chunks found at rank 0 by one
    retriever each and nowhere else score identically, and an unstable sort
    there would make the context handed to the model vary between runs.
    """
    if weights is None:
        weights = [1.0] * len(rankings)

    scores: dict[tuple, float] = {}
    found: dict[tuple, Document] = {}

    for ranking, weight in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            key = doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight / (rrf_k + rank + 1)
            found.setdefault(key, doc)

    order = sorted(scores, key=lambda key: -scores[key])
    return [found[key] for key in order]


def limit_per_page(
    docs: list[Document], limit: int, max_per_page: int
) -> list[Document]:
    """Take the best `limit` chunks, letting no page contribute more than a few.

    Order within the ranking is preserved; a chunk over its page's quota is set
    aside rather than dropped, and the set-aside chunks fill any shortfall at
    the end. So this never returns fewer chunks than it was given - it only
    changes which ones, and only when a page was monopolising the context.
    """
    seen: dict[tuple, int] = {}
    kept: list[Document] = []
    overflow: list[Document] = []

    for doc in docs:
        page = (doc.metadata.get("source"), doc.metadata.get("page"))
        if seen.get(page, 0) < max_per_page:
            seen[page] = seen.get(page, 0) + 1
            kept.append(doc)
            if len(kept) == limit:
                return kept
        else:
            overflow.append(doc)

    return (kept + overflow)[:limit]


class HybridRetriever:
    """Vector search and BM25 over one store, fused into a single ranking."""

    def __init__(
        self,
        vectordb,
        bm25: BM25Index | None = None,
        k: int = TOP_K,
        rrf_k: int = RRF_K,
        vector_weight: float = VECTOR_WEIGHT,
        bm25_weight: float = BM25_WEIGHT,
        max_per_page: int = MAX_PER_PAGE,
    ):
        self.vectordb = vectordb
        self.bm25 = bm25 if bm25 is not None else BM25Index.from_chroma(vectordb)
        self.k = k
        self.rrf_k = rrf_k
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.max_per_page = max_per_page

    def search(
        self, queries: list[str], limit: int, use_bm25: bool = True
    ) -> list[Document]:
        """The best `limit` chunks across every query and both retrievers.

        Each query contributes two rankings, and all of them go into one
        fusion. That is also what balances a multi-part question: the previous
        code interleaved per-query hit lists by hand to stop the first
        sub-question filling every slot, and fusion does the same job as a
        property of the arithmetic - each ranking's first hit is worth the
        same, whichever query produced it.

        `use_bm25=False` drops the keyword half and searches with vectors
        alone, which is what the UI's mode switch selects. It is a per-call
        argument rather than an attribute deliberately: app.py shares one
        retriever across every browser session, so a mode stored on the object
        would be one visitor silently changing another visitor's results.

        Vector-only still fuses, because with several sub-queries there is
        still more than one ranking to merge. On a single query fusion over one
        ranking is order-preserving - score falls monotonically with rank - so
        the result is exactly that ranking, untouched.
        """
        rankings, weights = [], []
        for query in queries:
            rankings.append(self.vectordb.similarity_search(query, k=self.k))
            weights.append(self.vector_weight)
            if use_bm25:
                rankings.append(self.bm25.search(query, k=self.k))
                weights.append(self.bm25_weight)

        fused = reciprocal_rank_fusion(rankings, weights, self.rrf_k)
        return limit_per_page(fused, limit, self.max_per_page)
