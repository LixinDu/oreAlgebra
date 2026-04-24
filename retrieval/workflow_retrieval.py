#!/usr/bin/env python3
"""Workflow-aware retrieval helpers without Streamlit UI dependencies."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

from retrieval.cross_encoder_reranker import rerank_if_enabled as _cross_encoder_rerank_if_enabled
from retrieval.knowledge_base import default_graph_path
from retrieval.knowledge_graph import (
    apply_graph_assisted_expansion,
    build_retrieval_graph_from_chunks,
    load_retrieval_graph,
)
from core.ore_rag_assistant import RetrievalResult, select_retrieval
from workflows.task_workflows import WorkflowSelection, build_workflow_retrieval_queries, choose_workflow

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SYMBOL_PART_RE = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|[0-9]+")
QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "compute",
    "find",
    "for",
    "from",
    "handle",
    "of",
    "or",
    "the",
    "this",
    "to",
    "use",
    "with",
}
SYMBOL_TOKEN_ALIASES = {
    "rec": ("recurrence",),
    "deq": ("differential", "equation"),
}
SYMBOL_QUERY_ALIASES = {
    "desingularize": ("apparent", "singularities", "remove"),
    "differentialoperators": ("derivation", "differential", "operator"),
    "guess_deq": ("infer", "recover", "differential", "equation"),
    "guess_rec": ("infer", "recover", "recurrence", "sample", "coefficients"),
    "numerical_solution": ("evaluate", "solution", "initial", "condition"),
    "numerical_transition_matrix": ("transition", "matrix", "path"),
    "orealgebra": ("derivation", "operator", "algebra"),
    "symmetric_power": ("square", "annihilator", "solution", "d-finite"),
    "symmetric_product": ("product", "annihilator"),
    "to_list": ("terms", "values", "sequence"),
    "to_s": ("taylor", "coefficient", "coefficients", "recurrence"),
}


def merge_retrieval_result_sets(result_sets: Iterable[Iterable[RetrievalResult]]) -> list[RetrievalResult]:
    best_by_chunk: dict[int, RetrievalResult] = {}
    hit_counts: dict[int, int] = {}
    best_ranks: dict[int, int] = {}
    first_seen: dict[int, int] = {}
    sequence = 0

    for results in result_sets:
        for rank, result in enumerate(results):
            chunk_id = result.chunk_id
            hit_counts[chunk_id] = hit_counts.get(chunk_id, 0) + 1
            best_ranks[chunk_id] = min(best_ranks.get(chunk_id, rank), rank)
            if chunk_id not in first_seen:
                first_seen[chunk_id] = sequence
                sequence += 1
            current = best_by_chunk.get(chunk_id)
            if current is None or result.score > current.score:
                best_by_chunk[chunk_id] = result

    merged = list(best_by_chunk.values())
    merged.sort(
        key=lambda item: (
            hit_counts.get(item.chunk_id, 0),
            item.score,
            -best_ranks.get(item.chunk_id, 10**6),
            -first_seen.get(item.chunk_id, 10**6),
        ),
        reverse=True,
    )
    return merged


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _query_tokens(question: str) -> tuple[str, ...]:
    tokens = []
    for token in TOKEN_RE.findall(str(question or "").lower()):
        if len(token) <= 1 or token in QUERY_STOPWORDS:
            continue
        tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _token_forms(token: str) -> tuple[str, ...]:
    normalized = str(token or "").strip().lower()
    if not normalized:
        return ()
    forms = [normalized]
    if len(normalized) > 4 and normalized.endswith("ies"):
        forms.append(normalized[:-3] + "y")
    elif len(normalized) > 3 and normalized.endswith("s") and not normalized.endswith("ss"):
        forms.append(normalized[:-1])
    return tuple(dict.fromkeys(forms))


def _split_symbol_words(text: str) -> tuple[str, ...]:
    raw_parts = [part for part in re.split(r"[^A-Za-z0-9]+", str(text or "").strip()) if part]
    words: list[str] = []
    for part in raw_parts:
        matched = SYMBOL_PART_RE.findall(part)
        if matched:
            words.extend(item.lower() for item in matched if item)
        else:
            words.append(part.lower())
    return tuple(words)


def _search_text(result: RetrievalResult) -> str:
    return _normalize_text(
        " ".join(
            part
            for part in (
                result.qualname,
                result.symbol_id,
                result.module,
                result.signature,
                result.summary,
                result.text,
            )
            if part
        )
    )


def _query_overlap_count(query_tokens: tuple[str, ...], haystack: str) -> int:
    search_text = _normalize_text(haystack)
    return sum(1 for token in query_tokens if any(form in search_text for form in _token_forms(token)))


def _preferred_symbol_match(result: RetrievalResult, preferred_symbols: tuple[str, ...]) -> tuple[int, int, str]:
    qualname = _normalize_text(result.qualname)
    symbol_id = _normalize_text(result.symbol_id)
    for index, symbol in enumerate(preferred_symbols):
        normalized = _normalize_text(symbol)
        if not normalized:
            continue
        if qualname == normalized or symbol_id == normalized:
            return 3, index, normalized
        if qualname.endswith(normalized) or symbol_id.endswith(normalized):
            return 3, index, normalized
        tail = normalized.rsplit(".", 1)[-1]
        if tail and (qualname.endswith(tail) or symbol_id.endswith(tail)):
            return 2, index, normalized
    return 0, 10**6, ""


def _preferred_symbol_query_affinity(symbol: str, query_tokens: tuple[str, ...]) -> int:
    tail = symbol.rsplit(".", 1)[-1].lower()
    raw_parts = list(_split_symbol_words(tail))
    expanded_parts: list[str] = [tail]
    for part in raw_parts:
        expanded_parts.append(part)
        expanded_parts.extend(SYMBOL_TOKEN_ALIASES.get(part, ()))
    expanded_parts.extend(SYMBOL_QUERY_ALIASES.get(tail, ()))
    expanded_set = {item.lower() for item in expanded_parts if item}
    return sum(1 for token in query_tokens if any(form in expanded_set for form in _token_forms(token)))


def _result_symbol_query_affinity(result: RetrievalResult, query_tokens: tuple[str, ...]) -> int:
    parts: set[str] = set()
    for symbol in (result.qualname, result.symbol_id):
        tail = str(symbol or "").strip().rsplit(".", 1)[-1].lower()
        if not tail:
            continue
        parts.add(tail)
        words = _split_symbol_words(tail)
        parts.update(words)
        for word in words:
            parts.update(alias.lower() for alias in SYMBOL_TOKEN_ALIASES.get(word, ()))
        parts.update(alias.lower() for alias in SYMBOL_QUERY_ALIASES.get(tail, ()))
    parts.update(_split_symbol_words(result.module))

    affinity = sum(1 for token in query_tokens if any(form in parts for form in _token_forms(token)))
    if "helper" in query_tokens and result.kind == "class":
        affinity += 2
    if "class" in query_tokens and result.kind == "class":
        affinity += 2
    if "method" in query_tokens and result.kind == "method":
        affinity += 1
    return affinity


def rerank_workflow_results(
    *,
    question: str,
    results: Iterable[RetrievalResult],
    workflow_selection: WorkflowSelection,
) -> list[RetrievalResult]:
    result_list = list(results)
    if not result_list or not workflow_selection.preferred_symbols:
        return result_list

    query_tokens = _query_tokens(question)
    preferred_symbols = tuple(workflow_selection.preferred_symbols)

    def sort_key(item: tuple[int, RetrievalResult]) -> tuple[int, int, int, int, float, int]:
        original_index, result = item
        preferred_level, preferred_index, matched_symbol = _preferred_symbol_match(result, preferred_symbols)
        symbol_affinity = _preferred_symbol_query_affinity(matched_symbol, query_tokens) if matched_symbol else 0
        search_text = _search_text(result)
        query_overlap = _query_overlap_count(query_tokens, search_text)
        generated_bonus = 1 if result.source_type == "generated" else 0
        symbol_bonus = 1 if result.qualname or result.symbol_id else 0
        return (
            preferred_level,
            symbol_affinity,
            -preferred_index,
            query_overlap,
            generated_bonus,
            symbol_bonus,
            result.score,
            -original_index,
        )

    ranked = list(enumerate(result_list))
    ranked.sort(key=sort_key, reverse=True)
    return [result for _, result in ranked]


def rerank_graph_results(
    *,
    question: str,
    results: Iterable[RetrievalResult],
    workflow_selection: WorkflowSelection,
) -> list[RetrievalResult]:
    result_list = list(results)
    if not result_list:
        return result_list

    query_tokens = _query_tokens(question)
    preferred_symbols = tuple(workflow_selection.preferred_symbols)

    def sort_key(item: tuple[int, RetrievalResult]) -> tuple[int, int, int, int, float, int, int]:
        original_index, result = item
        result_affinity = _result_symbol_query_affinity(result, query_tokens)
        search_text = _search_text(result)
        query_overlap = _query_overlap_count(query_tokens, search_text)
        symbol_bonus = 1 if result.qualname or result.symbol_id else 0
        generated_bonus = 1 if result.source_type == "generated" else 0
        preferred_level, preferred_index, matched_symbol = _preferred_symbol_match(result, preferred_symbols)
        preferred_affinity = _preferred_symbol_query_affinity(matched_symbol, query_tokens) if matched_symbol else 0
        return (
            result_affinity,
            preferred_affinity,
            symbol_bonus,
            query_overlap,
            generated_bonus,
            result.score,
            preferred_level,
            -preferred_index,
            -original_index,
        )

    ranked = list(enumerate(result_list))
    ranked.sort(key=sort_key, reverse=True)
    return [result for _, result in ranked]


def run_single_query_retrieval(
    *,
    query: str,
    payload: dict,
    chunks: list,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
) -> tuple[str, list[RetrievalResult]]:
    return select_retrieval(
        index_payload=payload,
        chunks=chunks,
        query=query,
        k=k,
        mode=mode,
        index_path=index_path,
        hybrid_alpha=hybrid_alpha,
        source_priority=source_priority,
        symbols_ratio=symbols_ratio,
        max_pdf_extras=max_pdf_extras,
    )


def run_workflow_retrieval(
    *,
    query: str,
    payload: dict,
    chunks: list,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
    family_hint: str = "",
    workflow_selection_override: WorkflowSelection | None = None,
    strategy: str = "classic",
    graph_path: str = "",
) -> tuple[str, list[RetrievalResult], WorkflowSelection, tuple[str, ...]]:
    active_strategy = str(strategy or "classic").strip().lower()
    if active_strategy not in {"classic", "graph"}:
        raise RuntimeError(f"Unsupported workflow retrieval strategy: {strategy}")

    if workflow_selection_override is not None and workflow_selection_override.has_workflow:
        workflow_selection = workflow_selection_override
    else:
        workflow_selection = choose_workflow(
            question=query,
            family_hint=family_hint,
        )
    retrieval_queries = build_workflow_retrieval_queries(
        question=query,
        selection=workflow_selection,
        family_hint=family_hint,
    )
    if not retrieval_queries:
        retrieval_queries = (query,)

    per_query_k = max(k, 10) if len(retrieval_queries) > 1 else k
    mode_labels: list[str] = []
    result_sets: List[List[RetrievalResult]] = []
    for retrieval_query in retrieval_queries:
        mode_used, results = run_single_query_retrieval(
            query=retrieval_query,
            payload=payload,
            chunks=chunks,
            k=per_query_k,
            mode=mode,
            index_path=index_path,
            hybrid_alpha=hybrid_alpha,
            source_priority=source_priority,
            symbols_ratio=symbols_ratio,
            max_pdf_extras=max_pdf_extras,
        )
        mode_labels.append(mode_used)
        result_sets.append(results)

    merged_results = merge_retrieval_result_sets(result_sets)
    if active_strategy == "graph":
        candidate_graph_path = str(graph_path or default_graph_path()).strip()
        resolved_graph = None
        if candidate_graph_path:
            graph_file = Path(candidate_graph_path).expanduser().resolve()
            if graph_file.exists():
                resolved_graph = load_retrieval_graph(str(graph_file))
        if resolved_graph is None:
            resolved_graph = build_retrieval_graph_from_chunks(chunks)
        merged_results = apply_graph_assisted_expansion(
            question=query,
            results=merged_results,
            workflow_selection=workflow_selection,
            chunks=chunks,
            graph=resolved_graph,
        )
        merged_results = rerank_graph_results(
            question=query,
            results=merged_results,
            workflow_selection=workflow_selection,
        )
    else:
        merged_results = rerank_workflow_results(
            question=query,
            results=merged_results,
            workflow_selection=workflow_selection,
        )

    # Optional cross-encoder rerank as a fine-grained relevance pass on
    # the candidates left after symbolic/workflow reranking. Opt-in via
    # ORE_ASSISTANT_USE_CROSS_ENCODER=1; when the env flag is not set
    # this is a no-op and the original ordering is preserved.
    from retrieval.cross_encoder_reranker import cross_encoder_enabled

    cross_encoder_active = cross_encoder_enabled()
    if cross_encoder_active:
        merged_results = _cross_encoder_rerank_if_enabled(
            query=query,
            results=merged_results,
        )

    merged_results = merged_results[:k]
    mode_label = mode_labels[0] if mode_labels else mode
    if len(retrieval_queries) > 1:
        mode_label = f"{mode_label}+workflow"
    if active_strategy == "graph":
        mode_label = f"{mode_label}+graph"
    if cross_encoder_active:
        mode_label = f"{mode_label}+ce"
    return mode_label, merged_results, workflow_selection, retrieval_queries
