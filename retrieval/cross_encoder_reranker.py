#!/usr/bin/env python3
"""Cross-encoder reranker for ore_algebra retrieval results.

The base retrieval pipeline ranks chunks with TF-IDF + dense biencoder
scores and then applies symbolic/workflow-aware sort keys. Both stages
are cheap but lossy: they cannot tell that a query about a *commutator*
in the differential algebra wants the OreAlgebra constructor and not a
numerical-evaluation entry that happens to share several tokens.

A cross-encoder scores ``(query, candidate_text)`` pairs jointly and is
much better at this kind of fine-grained relevance decision. We only run
it on the top-N candidates returned by the cheap pipeline, so latency
stays bounded.

Design notes:

- Opt-in via ``ORE_ASSISTANT_USE_CROSS_ENCODER=1`` so existing benchmarks
  stay reproducible until we have collected enough signal to flip the
  default.
- Lazy model load. The ``CrossEncoder`` is built once and cached, so the
  first call pays the model-load cost and subsequent calls are cheap.
- Drop-in: takes a list of ``RetrievalResult`` and returns a re-ordered
  list. Falls back to the input order on any failure (model unavailable,
  empty input, exception during scoring).
- Default model is ``cross-encoder/ms-marco-MiniLM-L-6-v2`` because it is
  small (~90MB), CPU-friendly, and already shipped by the
  ``sentence-transformers`` package we depend on.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Iterable, List, Sequence

from core.ore_rag_assistant import RetrievalResult


CROSS_ENCODER_ENV_VAR = "ORE_ASSISTANT_USE_CROSS_ENCODER"
CROSS_ENCODER_MODEL_ENV_VAR = "ORE_ASSISTANT_CROSS_ENCODER_MODEL"
DEFAULT_CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Cap how many chunks of context we feed the cross-encoder per call.
# More than this is rarely useful and slows everything down.
DEFAULT_TOPN_FOR_RERANK = 20

# Maximum length (chars) of the candidate text we feed the cross-encoder.
# We trim aggressively so the model's 512-token window mostly captures
# the symbol header / signature / first sentences of the docstring.
MAX_CANDIDATE_CHARS = 800


def cross_encoder_enabled() -> bool:
    raw = (os.getenv(CROSS_ENCODER_ENV_VAR, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _candidate_text(result: RetrievalResult) -> str:
    """Build the text the cross-encoder scores against the query."""

    parts: List[str] = []
    if result.qualname:
        parts.append(result.qualname)
    elif result.symbol_id:
        parts.append(result.symbol_id)
    if result.signature:
        parts.append(result.signature)
    if result.summary:
        parts.append(result.summary)
    if result.text and not result.summary:
        parts.append(result.text)
    joined = "\n".join(part for part in parts if part)
    if len(joined) > MAX_CANDIDATE_CHARS:
        joined = joined[:MAX_CANDIDATE_CHARS]
    return joined


@lru_cache(maxsize=2)
def _load_cross_encoder(model_name: str):
    """Load and cache a CrossEncoder by name. Returns None on failure."""

    try:
        from sentence_transformers import CrossEncoder  # local import: optional
    except Exception:
        return None
    try:
        return CrossEncoder(model_name)
    except Exception:
        return None


def get_default_cross_encoder():
    model_name = (os.getenv(CROSS_ENCODER_MODEL_ENV_VAR) or DEFAULT_CROSS_ENCODER_MODEL).strip()
    return _load_cross_encoder(model_name or DEFAULT_CROSS_ENCODER_MODEL)


def cross_encoder_rerank(
    *,
    query: str,
    results: Iterable[RetrievalResult],
    top_n: int = DEFAULT_TOPN_FOR_RERANK,
    model=None,
) -> list[RetrievalResult]:
    """Re-rank ``results`` by cross-encoder relevance to ``query``.

    Only the first ``top_n`` results are scored; anything beyond is
    appended unchanged at the end. The ranking is stable: ties keep
    their original order.

    Returns the input list (as a list) on any failure or empty input,
    so callers can use this as a transparent drop-in.
    """

    result_list = list(results)
    if not result_list or not query.strip():
        return result_list

    head = result_list[:top_n]
    tail = result_list[top_n:]

    encoder = model if model is not None else get_default_cross_encoder()
    if encoder is None:
        return result_list

    pairs = [(query, _candidate_text(item)) for item in head]
    if not any(text for _, text in pairs):
        return result_list

    try:
        scores = encoder.predict(pairs)
    except Exception:
        return result_list

    indexed = list(enumerate(head))
    # Stable sort by descending score; preserve original order on ties.
    indexed.sort(key=lambda pair: (-float(scores[pair[0]]), pair[0]))
    reordered_head = [item for _, item in indexed]
    return reordered_head + tail


def rerank_if_enabled(
    *,
    query: str,
    results: Sequence[RetrievalResult],
    top_n: int = DEFAULT_TOPN_FOR_RERANK,
) -> list[RetrievalResult]:
    """Run the cross-encoder reranker iff the env flag is set.

    This is the convenience entry point for callers that want to keep
    the new path strictly opt-in.
    """

    if not cross_encoder_enabled():
        return list(results)
    return cross_encoder_rerank(query=query, results=results, top_n=top_n)
