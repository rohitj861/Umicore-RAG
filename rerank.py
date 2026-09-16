# rerank.py
"""Cohere Rerank, the second stage of the semantic-only search mode.

Measured on the graded questions, semantic-only search did find the chunk that
prints the answer - the Group key figures table, the consolidated income
statement - but ranked it 16th to 38th, just outside the 16 it keeps. A table
chunk is mostly numbers, and a question's embedding sits nearer to prose that
discusses a figure than to the table row that prints it. BM25 ranks the same
chunks 2nd or 3rd, which is why hybrid search gets those questions right.

A reranker reads the query and each candidate together instead of comparing
two independently made embeddings, which is the comparison that goes wrong on
a row like "Adjusted EBITDA 370 414 763 847". So semantic-only keeps embedding
search as its first stage, fetching a deeper pool of candidates, and Cohere
Rerank orders that pool. It stays a semantic search - no keyword matching and
no rewritten query is added - and hybrid search does not use it at all.

Called over HTTP rather than through Cohere's SDK: httpx is already installed
for the OpenAI client, while the SDK would add packages that need building,
which is a deployment risk for one POST request.
"""

from __future__ import annotations

import os
import time

import httpx
from langchain_core.documents import Document

COHERE_RERANK_URL = "https://api.cohere.com/v2/rerank"

# Cohere's recommended model for quality. COHERE_RERANK_MODEL overrides it -
# "rerank-v4.0-fast" trades some quality for latency - without a code change.
DEFAULT_MODEL = "rerank-v4.0-pro"

# A timeout counts as a failure, and a failure falls back to plain embedding
# order rather than stalling an answer.
TIMEOUT_SECONDS = 15.0

# Retries for responses worth retrying: 429 (rate limited) and 5xx. Trial keys
# are rate limited, and an evaluation run issues dozens of calls in a minute.
MAX_RETRIES = 3
MAX_WAIT_SECONDS = 10.0
RETRYABLE = {429, 500, 502, 503, 504}


class RerankError(RuntimeError):
    """Cohere could not rerank - no key, a refused request, or no response."""


class CohereReranker:
    """Orders candidate chunks by Cohere Rerank's relevance to one query.

    Holds no state between calls other than the counters, so one instance is
    safely shared by every browser session, as the retriever that owns it is.
    The counters are for evaluate.py, which has to know whether the answers it
    graded were actually reranked.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        client: httpx.Client | None = None,
    ):
        self.api_key = (
            api_key if api_key is not None else os.getenv("COHERE_API_KEY", "")
        ).strip()
        self.model = (
            model or os.getenv("COHERE_RERANK_MODEL", "").strip() or DEFAULT_MODEL
        )
        # Injectable so the tests can answer without a network.
        self.client = client or httpx.Client(timeout=TIMEOUT_SECONDS)
        self.calls = 0
        self.failures = 0
        # Optional client-side pacing: at most this many requests a minute,
        # spaced evenly. A trial key allows 10, and retrying a 429 for a few
        # seconds does not get under a per-minute limit - the first graded run
        # fell back to embedding order on 5 of 28 questions that way. Off by
        # default, because in the app it would make a visitor wait; evaluate.py
        # turns it on, where waiting is better than grading unreranked answers.
        calls_per_minute = os.getenv("COHERE_CALLS_PER_MINUTE", "").strip()
        self.calls_per_minute = float(calls_per_minute) if calls_per_minute else None
        self._last_request = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def rerank(self, query: str, docs: list[Document], top_n: int) -> list[Document]:
        """The `top_n` most relevant of `docs` for `query`, most relevant first.

        Raises RerankError on anything but a well-formed answer; the caller
        decides what to fall back to.
        """
        if not self.enabled:
            raise RerankError("COHERE_API_KEY is not set")
        if not docs:
            return []

        self.calls += 1
        try:
            results = self._request(query, [doc.page_content for doc in docs], top_n)
            ranked = [docs[result["index"]] for result in results]
        except RerankError:
            self.failures += 1
            raise
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            self.failures += 1
            raise RerankError(f"unexpected response from Cohere: {exc}") from exc
        return ranked[:top_n]

    def _request(self, query: str, documents: list[str], top_n: int) -> list[dict]:
        body = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": min(top_n, len(documents)),
        }
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(MAX_RETRIES + 1):
            self._pace()
            try:
                response = self.client.post(COHERE_RERANK_URL, json=body, headers=headers)
            except httpx.HTTPError as exc:
                if attempt == MAX_RETRIES:
                    raise RerankError(f"could not reach Cohere: {exc}") from exc
                time.sleep(self._backoff(attempt, None))
                continue

            if response.status_code == 200:
                results = response.json()["results"]
                # Cohere returns them best first; sorted again rather than
                # trusted, since the order is the whole point.
                return sorted(results, key=lambda r: -r["relevance_score"])

            if response.status_code in RETRYABLE and attempt < MAX_RETRIES:
                time.sleep(self._backoff(attempt, response.headers.get("retry-after")))
                continue

            raise RerankError(
                f"Cohere returned HTTP {response.status_code}: {response.text[:200]}"
            )

        raise RerankError("Cohere did not answer")  # unreachable; keeps types honest

    def _pace(self) -> None:
        """Wait until the next request fits under `calls_per_minute`, if set."""
        if not self.calls_per_minute:
            return
        interval = 60.0 / self.calls_per_minute
        wait = self._last_request + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        """Seconds to wait: Cohere's Retry-After if given, else 1, 2, 4, capped."""
        try:
            wait = float(retry_after) if retry_after else 2.0 ** attempt
        except ValueError:
            wait = 2.0 ** attempt
        return min(wait, MAX_WAIT_SECONDS)
