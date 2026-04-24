#!/usr/bin/env python3
"""Experimental graph helpers for structure-aware ore_algebra retrieval."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

from core.ore_rag_assistant import Chunk, RetrievalResult, chunk_to_result, utc_now_iso
from workflows.task_workflows import WorkflowSelection, load_workflow_registry

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
GRAPH_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "compute",
    "find",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "with",
}


@dataclass(frozen=True)
class GraphSymbolNode:
    symbol_id: str
    qualname: str
    module: str
    kind: str
    owner: str
    tail: str
    workflow_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class GraphNarrativeNode:
    section_id: str
    module: str
    section_title: str
    referenced_symbol_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalKnowledgeGraph:
    nodes_by_id: dict[str, GraphSymbolNode]
    module_symbols: dict[str, tuple[str, ...]]
    owner_symbols: dict[str, tuple[str, ...]]
    workflow_seed_symbols: dict[str, tuple[str, ...]]
    qualname_index: dict[str, tuple[str, ...]]
    tail_index: dict[str, tuple[str, ...]]
    # Cross-source: narrative ↔ symbol edges
    narrative_nodes: dict[str, GraphNarrativeNode] = field(default_factory=dict)
    narrative_refs_symbol: dict[str, tuple[str, ...]] = field(default_factory=dict)
    symbol_referred_by_narratives: dict[str, tuple[str, ...]] = field(default_factory=dict)
    module_narratives: dict[str, tuple[str, ...]] = field(default_factory=dict)


def _normalize_key(value: object) -> str:
    return str(value or "").strip().lower()


def _owner_of_qualname(qualname: str) -> str:
    text = str(qualname or "").strip()
    if "." not in text:
        return ""
    return text.rsplit(".", 1)[0]


def _tail_of_name(name: str) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    return text.rsplit(".", 1)[-1]


def _query_tokens(question: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in TOKEN_RE.findall(str(question or "").lower()):
        if len(token) <= 1 or token in GRAPH_QUERY_STOPWORDS:
            continue
        tokens.append(token)
    return tuple(dict.fromkeys(tokens))


def _query_affinity(node: GraphSymbolNode, query_tokens: tuple[str, ...]) -> int:
    haystack = " ".join(
        part
        for part in (
            node.qualname,
            node.module,
            node.owner,
            node.tail,
        )
        if part
    ).lower()
    return sum(1 for token in query_tokens if token in haystack)


def _candidate_rows_from_chunks(chunks: Sequence[Chunk]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.source_type != "generated":
            continue
        if chunk.kind == "module_reference":
            continue
        symbol_id = str(chunk.symbol_id or chunk.qualname).strip()
        qualname = str(chunk.qualname or chunk.symbol_id).strip()
        if not symbol_id or not qualname:
            continue
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        rows.append(
            {
                "symbol_id": symbol_id,
                "qualname": qualname,
                "module": str(chunk.module or "").strip(),
                "kind": str(chunk.kind or "").strip(),
                "owner": _owner_of_qualname(qualname),
                "tail": _tail_of_name(qualname or symbol_id),
            }
        )
    return rows


def _candidate_rows_from_symbols_file(symbols_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_line in symbols_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            continue
        symbol_id = str(record.get("id") or record.get("qualname") or "").strip()
        qualname = str(record.get("qualname") or record.get("id") or "").strip()
        if not symbol_id or not qualname:
            continue
        if symbol_id in seen:
            continue
        seen.add(symbol_id)
        rows.append(
            {
                "symbol_id": symbol_id,
                "qualname": qualname,
                "module": str(record.get("module") or "").strip(),
                "kind": str(record.get("kind") or "").strip(),
                "owner": _owner_of_qualname(qualname),
                "tail": _tail_of_name(qualname or symbol_id),
            }
        )
    return rows


def _resolve_symbol_refs(
    *,
    rows_by_id: dict[str, dict[str, str]],
    qualname_index: dict[str, list[str]],
    tail_index: dict[str, list[str]],
    reference: str,
) -> tuple[str, ...]:
    normalized = _normalize_key(reference)
    if not normalized or " " in normalized:
        return ()

    matches: list[str] = []
    if normalized in rows_by_id:
        matches.append(normalized)
    matches.extend(qualname_index.get(normalized, ()))

    if "." in normalized:
        for row_id, row in rows_by_id.items():
            qualname = _normalize_key(row.get("qualname"))
            if qualname.endswith(normalized):
                matches.append(row_id)
    else:
        matches.extend(tail_index.get(normalized, ()))
        for row_id, row in rows_by_id.items():
            qualname = _normalize_key(row.get("qualname"))
            if qualname == normalized or qualname.endswith(f".{normalized}"):
                matches.append(row_id)

    out: list[str] = []
    seen: set[str] = set()
    for item in matches:
        key = _normalize_key(item)
        if not key or key in seen:
            continue
        if key not in rows_by_id:
            continue
        seen.add(key)
        out.append(key)
    return tuple(out)


def _graph_payload_from_rows(
    rows: Sequence[dict[str, str]],
    *,
    generated_symbols: str,
) -> dict[str, object]:
    rows_by_id: dict[str, dict[str, str]] = {}
    qualname_index: dict[str, list[str]] = {}
    tail_index: dict[str, list[str]] = {}
    module_symbols: dict[str, list[str]] = {}
    owner_symbols: dict[str, list[str]] = {}

    for row in rows:
        symbol_id = _normalize_key(row.get("symbol_id"))
        if not symbol_id:
            continue
        qualname = _normalize_key(row.get("qualname"))
        tail = _normalize_key(row.get("tail"))
        module = str(row.get("module") or "").strip()
        owner = str(row.get("owner") or "").strip()
        rows_by_id[symbol_id] = {
            "symbol_id": str(row.get("symbol_id") or "").strip(),
            "qualname": str(row.get("qualname") or "").strip(),
            "module": module,
            "kind": str(row.get("kind") or "").strip(),
            "owner": owner,
            "tail": str(row.get("tail") or "").strip(),
        }
        if qualname:
            qualname_index.setdefault(qualname, []).append(symbol_id)
        if tail:
            tail_index.setdefault(tail, []).append(symbol_id)
        if module:
            module_symbols.setdefault(module, []).append(symbol_id)
        if owner:
            owner_symbols.setdefault(owner, []).append(symbol_id)

    workflow_seed_symbols: dict[str, list[str]] = {}
    workflow_ids_by_symbol: dict[str, set[str]] = {key: set() for key in rows_by_id}
    registry = load_workflow_registry()
    for workflow in registry.workflows:
        seed_ids: list[str] = []
        seen: set[str] = set()
        for reference in workflow.preferred_symbols:
            for symbol_id in _resolve_symbol_refs(
                rows_by_id=rows_by_id,
                qualname_index=qualname_index,
                tail_index=tail_index,
                reference=reference,
            ):
                if symbol_id in seen:
                    continue
                seen.add(symbol_id)
                seed_ids.append(symbol_id)
                workflow_ids_by_symbol.setdefault(symbol_id, set()).add(workflow.id)
        workflow_seed_symbols[workflow.id] = seed_ids

    symbols_payload: list[dict[str, object]] = []
    for symbol_id in sorted(rows_by_id):
        row = rows_by_id[symbol_id]
        symbols_payload.append(
            {
                **row,
                "workflow_ids": sorted(workflow_ids_by_symbol.get(symbol_id, set())),
            }
        )

    return {
        "version": 1,
        "generated_at": utc_now_iso(),
        "sources": {
            "generated_symbols": generated_symbols,
            "workflow_config": str(Path(__file__).resolve().parents[1] / "config" / "task_workflows.json"),
        },
        "symbols": symbols_payload,
        "module_symbols": {key: sorted(value) for key, value in sorted(module_symbols.items())},
        "owner_symbols": {key: sorted(value) for key, value in sorted(owner_symbols.items())},
        "workflow_seed_symbols": {key: sorted(value) for key, value in sorted(workflow_seed_symbols.items())},
    }


def _load_narrative_records(narratives_path: Path) -> list[dict[str, object]]:
    if not narratives_path.exists():
        return []
    records: list[dict[str, object]] = []
    for raw_line in narratives_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if isinstance(rec, dict) and rec.get("section_id"):
            records.append(rec)
    return records


def _add_narrative_edges(
    payload: dict[str, object],
    narrative_records: list[dict[str, object]],
    narratives_source: str,
) -> None:
    """Augment a graph payload with narrative nodes and cross-source edges."""
    if not narrative_records:
        return

    # Collect valid symbol IDs from the payload so we only store edges
    # that reference known symbols.
    valid_symbol_ids: set[str] = set()
    for sym in payload.get("symbols", []):
        if isinstance(sym, dict):
            sid = _normalize_key(sym.get("symbol_id"))
            if sid:
                valid_symbol_ids.add(sid)

    narratives_payload: list[dict[str, object]] = []
    narrative_refs: dict[str, list[str]] = {}
    symbol_to_narratives: dict[str, list[str]] = {}
    module_to_narratives: dict[str, list[str]] = {}

    for rec in narrative_records:
        section_id = str(rec.get("section_id") or "").strip()
        if not section_id:
            continue
        module = str(rec.get("module") or "").strip()
        section_title = str(rec.get("section_title") or "").strip()
        raw_symbol_ids = rec.get("resolved_symbol_ids", [])
        if not isinstance(raw_symbol_ids, list):
            raw_symbol_ids = []

        # Filter to only known symbol IDs
        resolved = []
        for sid in raw_symbol_ids:
            normalized = _normalize_key(sid)
            if normalized in valid_symbol_ids:
                resolved.append(normalized)

        narratives_payload.append({
            "section_id": section_id,
            "module": module,
            "section_title": section_title,
            "referenced_symbol_ids": resolved,
        })

        narrative_refs[section_id] = resolved

        # Reverse edges: symbol → narratives that reference it
        for sid in resolved:
            symbol_to_narratives.setdefault(sid, []).append(section_id)

        # Module → narratives
        if module:
            module_to_narratives.setdefault(module, []).append(section_id)

    payload["narratives"] = narratives_payload
    payload["narrative_refs_symbol"] = {
        k: sorted(set(v)) for k, v in sorted(narrative_refs.items())
    }
    payload["symbol_referred_by_narratives"] = {
        k: sorted(set(v)) for k, v in sorted(symbol_to_narratives.items())
    }
    payload["module_narratives"] = {
        k: sorted(set(v)) for k, v in sorted(module_to_narratives.items())
    }
    payload.setdefault("sources", {})["narratives"] = narratives_source  # type: ignore[union-attr]


def build_graph_payload_from_chunks(
    chunks: Sequence[Chunk],
    narratives_path: Path | None = None,
) -> dict[str, object]:
    payload = _graph_payload_from_rows(
        _candidate_rows_from_chunks(chunks),
        generated_symbols="generated:index_chunks",
    )
    if narratives_path:
        records = _load_narrative_records(narratives_path)
        _add_narrative_edges(payload, records, str(narratives_path))
    return payload


def build_graph_payload_from_symbols_file(
    symbols_path: Path,
    narratives_path: Path | None = None,
) -> dict[str, object]:
    payload = _graph_payload_from_rows(
        _candidate_rows_from_symbols_file(symbols_path),
        generated_symbols=str(symbols_path.resolve()),
    )
    if narratives_path:
        records = _load_narrative_records(narratives_path)
        _add_narrative_edges(payload, records, str(narratives_path))
    return payload


def write_graph_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _runtime_graph_from_payload(payload: dict[str, object]) -> RetrievalKnowledgeGraph:
    raw_symbols = payload.get("symbols", [])
    nodes_by_id: dict[str, GraphSymbolNode] = {}
    qualname_index: dict[str, list[str]] = {}
    tail_index: dict[str, list[str]] = {}

    if isinstance(raw_symbols, list):
        for item in raw_symbols:
            if not isinstance(item, dict):
                continue
            symbol_id_raw = str(item.get("symbol_id") or "").strip()
            symbol_id = _normalize_key(symbol_id_raw)
            if not symbol_id:
                continue
            node = GraphSymbolNode(
                symbol_id=symbol_id_raw,
                qualname=str(item.get("qualname") or "").strip(),
                module=str(item.get("module") or "").strip(),
                kind=str(item.get("kind") or "").strip(),
                owner=str(item.get("owner") or "").strip(),
                tail=str(item.get("tail") or "").strip(),
                workflow_ids=tuple(str(x).strip() for x in item.get("workflow_ids", []) if str(x).strip()),
            )
            nodes_by_id[symbol_id] = node
            qualname_index.setdefault(_normalize_key(node.qualname), []).append(symbol_id)
            tail_index.setdefault(_normalize_key(node.tail), []).append(symbol_id)

    def _tuple_map(name: str) -> dict[str, tuple[str, ...]]:
        raw = payload.get(name, {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for key, value in raw.items():
            if not isinstance(value, list):
                continue
            out[str(key)] = tuple(
                item_key
                for item in value
                for item_key in [_normalize_key(item)]
                if item_key in nodes_by_id
            )
        return out

    # Load narrative nodes and cross-source edges
    narrative_nodes: dict[str, GraphNarrativeNode] = {}
    raw_narratives = payload.get("narratives", [])
    if isinstance(raw_narratives, list):
        for item in raw_narratives:
            if not isinstance(item, dict):
                continue
            section_id = str(item.get("section_id") or "").strip()
            if not section_id:
                continue
            ref_ids = item.get("referenced_symbol_ids", [])
            if not isinstance(ref_ids, list):
                ref_ids = []
            narrative_nodes[section_id] = GraphNarrativeNode(
                section_id=section_id,
                module=str(item.get("module") or "").strip(),
                section_title=str(item.get("section_title") or "").strip(),
                referenced_symbol_ids=tuple(str(x).strip() for x in ref_ids if str(x).strip()),
            )

    def _str_tuple_map(name: str) -> dict[str, tuple[str, ...]]:
        raw = payload.get(name, {})
        if not isinstance(raw, dict):
            return {}
        out: dict[str, tuple[str, ...]] = {}
        for key, value in raw.items():
            if not isinstance(value, list):
                continue
            out[str(key)] = tuple(str(x).strip() for x in value if str(x).strip())
        return out

    return RetrievalKnowledgeGraph(
        nodes_by_id=nodes_by_id,
        module_symbols=_tuple_map("module_symbols"),
        owner_symbols=_tuple_map("owner_symbols"),
        workflow_seed_symbols=_tuple_map("workflow_seed_symbols"),
        qualname_index={key: tuple(value) for key, value in qualname_index.items() if key},
        tail_index={key: tuple(value) for key, value in tail_index.items() if key},
        narrative_nodes=narrative_nodes,
        narrative_refs_symbol=_str_tuple_map("narrative_refs_symbol"),
        symbol_referred_by_narratives=_str_tuple_map("symbol_referred_by_narratives"),
        module_narratives=_str_tuple_map("module_narratives"),
    )


@lru_cache(maxsize=4)
def load_graph_payload(path_str: str) -> dict[str, object]:
    path = Path(path_str).expanduser().resolve()
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=4)
def load_retrieval_graph(path_str: str) -> RetrievalKnowledgeGraph:
    return _runtime_graph_from_payload(load_graph_payload(path_str))


def build_retrieval_graph_from_chunks(chunks: Sequence[Chunk]) -> RetrievalKnowledgeGraph:
    return _runtime_graph_from_payload(build_graph_payload_from_chunks(chunks))


def resolve_symbol_ids(graph: RetrievalKnowledgeGraph, reference: str) -> tuple[str, ...]:
    normalized = _normalize_key(reference)
    if not normalized:
        return ()
    matches: list[str] = []
    if normalized in graph.nodes_by_id:
        matches.append(normalized)
    matches.extend(graph.qualname_index.get(normalized, ()))
    if "." in normalized:
        for symbol_id, node in graph.nodes_by_id.items():
            qualname = _normalize_key(node.qualname)
            if qualname.endswith(normalized):
                matches.append(symbol_id)
    else:
        matches.extend(graph.tail_index.get(normalized, ()))
        for symbol_id, node in graph.nodes_by_id.items():
            qualname = _normalize_key(node.qualname)
            if qualname == normalized or qualname.endswith(f".{normalized}"):
                matches.append(symbol_id)
    out: list[str] = []
    seen: set[str] = set()
    for item in matches:
        key = _normalize_key(item)
        if key and key in graph.nodes_by_id and key not in seen:
            seen.add(key)
            out.append(key)
    return tuple(out)


def _best_related_symbols(
    graph: RetrievalKnowledgeGraph,
    symbol_ids: Iterable[str],
    *,
    query_tokens: tuple[str, ...],
    limit: int,
) -> tuple[str, ...]:
    ranked = sorted(
        {
            symbol_id
            for symbol_id in symbol_ids
            if symbol_id in graph.nodes_by_id
        },
        key=lambda symbol_id: (
            _query_affinity(graph.nodes_by_id[symbol_id], query_tokens),
            graph.nodes_by_id[symbol_id].tail.lower(),
        ),
        reverse=True,
    )
    return tuple(ranked[:limit])


def _chunk_index_for_symbols(chunks: Sequence[Chunk]) -> tuple[dict[str, Chunk], dict[str, Chunk]]:
    by_symbol_id: dict[str, Chunk] = {}
    by_qualname: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.source_type != "generated":
            continue
        if chunk.symbol_id:
            by_symbol_id[_normalize_key(chunk.symbol_id)] = chunk
        if chunk.qualname:
            by_qualname[_normalize_key(chunk.qualname)] = chunk
    return by_symbol_id, by_qualname


def _chunk_index_for_narratives(chunks: Sequence[Chunk]) -> dict[str, Chunk]:
    """Index narrative chunks by their section_id (stored in symbol_id field)."""
    by_section_id: dict[str, Chunk] = {}
    for chunk in chunks:
        if chunk.source_type != "narrative":
            continue
        key = chunk.symbol_id or chunk.section_title
        if key:
            by_section_id[key] = chunk
    return by_section_id


def _symbol_id_for_result(graph: RetrievalKnowledgeGraph, result: RetrievalResult) -> str:
    if result.symbol_id and _normalize_key(result.symbol_id) in graph.nodes_by_id:
        return _normalize_key(result.symbol_id)
    if result.qualname:
        matches = resolve_symbol_ids(graph, result.qualname)
        if matches:
            return matches[0]
    return ""


def apply_graph_assisted_expansion(
    *,
    question: str,
    results: Sequence[RetrievalResult],
    workflow_selection: WorkflowSelection,
    chunks: Sequence[Chunk],
    graph: RetrievalKnowledgeGraph,
    max_graph_candidates: int = 18,
    seed_results_limit: int = 4,
) -> list[RetrievalResult]:
    result_list = list(results)
    if not result_list or not workflow_selection.has_workflow:
        return result_list

    query_tokens = _query_tokens(question)
    by_symbol_id, by_qualname = _chunk_index_for_symbols(chunks)
    preferred_seed_ids = list(graph.workflow_seed_symbols.get(workflow_selection.workflow_id, ()))
    if not preferred_seed_ids:
        for reference in workflow_selection.preferred_symbols:
            preferred_seed_ids.extend(resolve_symbol_ids(graph, reference))

    retrieved_seed_ids: list[str] = []
    for result in result_list:
        symbol_id = _symbol_id_for_result(graph, result)
        if not symbol_id:
            continue
        retrieved_seed_ids.append(symbol_id)
        if len(retrieved_seed_ids) >= seed_results_limit:
            break

    if not preferred_seed_ids and not retrieved_seed_ids:
        return result_list

    score_by_symbol: dict[str, float] = {}

    def add_score(symbol_id: str, value: float) -> None:
        if symbol_id not in graph.nodes_by_id:
            return
        score_by_symbol[symbol_id] = score_by_symbol.get(symbol_id, 0.0) + value

    for symbol_id in preferred_seed_ids:
        add_score(symbol_id, 3.0)
        node = graph.nodes_by_id.get(symbol_id)
        if node is None:
            continue
        if node.owner:
            for related in _best_related_symbols(
                graph,
                graph.owner_symbols.get(node.owner, ()),
                query_tokens=query_tokens,
                limit=6,
            ):
                if related != symbol_id:
                    add_score(related, 1.8)
            for owner_id in resolve_symbol_ids(graph, node.owner):
                if owner_id != symbol_id:
                    add_score(owner_id, 1.2)
        if node.module:
            for related in _best_related_symbols(
                graph,
                graph.module_symbols.get(node.module, ()),
                query_tokens=query_tokens,
                limit=5,
            ):
                if related != symbol_id:
                    add_score(related, 1.0)

    for symbol_id in retrieved_seed_ids:
        add_score(symbol_id, 0.5)
        node = graph.nodes_by_id.get(symbol_id)
        if node is None:
            continue
        if node.owner:
            for related in _best_related_symbols(
                graph,
                graph.owner_symbols.get(node.owner, ()),
                query_tokens=query_tokens,
                limit=4,
            ):
                if related != symbol_id:
                    add_score(related, 0.8)
        if node.module:
            for related in _best_related_symbols(
                graph,
                graph.module_symbols.get(node.module, ()),
                query_tokens=query_tokens,
                limit=3,
            ):
                if related != symbol_id:
                    add_score(related, 0.35)

    for symbol_id in list(score_by_symbol):
        node = graph.nodes_by_id[symbol_id]
        score_by_symbol[symbol_id] += 0.15 * _query_affinity(node, query_tokens)

    ranked_symbol_ids = [
        symbol_id
        for symbol_id, _ in sorted(
            score_by_symbol.items(),
            key=lambda item: (item[1], _query_affinity(graph.nodes_by_id[item[0]], query_tokens)),
            reverse=True,
        )
    ][:max_graph_candidates]

    merged_by_chunk: dict[int, RetrievalResult] = {result.chunk_id: result for result in result_list}
    for symbol_id in ranked_symbol_ids:
        node = graph.nodes_by_id[symbol_id]
        chunk = by_symbol_id.get(symbol_id) or by_qualname.get(_normalize_key(node.qualname))
        if chunk is None:
            continue
        bonus = score_by_symbol.get(symbol_id, 0.0)
        existing = merged_by_chunk.get(chunk.chunk_id)
        if existing is not None:
            merged_by_chunk[chunk.chunk_id] = replace(existing, score=existing.score + bonus)
        else:
            merged_by_chunk[chunk.chunk_id] = chunk_to_result(chunk, bonus)

    # --- Cross-source expansion: symbol ↔ narrative edges ---
    if graph.narrative_nodes:
        by_narrative_id = _chunk_index_for_narratives(chunks)

        # Collect narrative scores from symbols that scored well
        narrative_scores: dict[str, float] = {}
        for symbol_id in ranked_symbol_ids[:10]:
            sym_score = score_by_symbol.get(symbol_id, 0.0)
            for narrative_id in graph.symbol_referred_by_narratives.get(symbol_id, ()):
                narrative_scores[narrative_id] = (
                    narrative_scores.get(narrative_id, 0.0) + sym_score * 0.4
                )

        # Also boost narratives for retrieved results that are symbols
        for result in result_list[:seed_results_limit]:
            sid = _symbol_id_for_result(graph, result)
            if not sid:
                continue
            for narrative_id in graph.symbol_referred_by_narratives.get(sid, ()):
                narrative_scores[narrative_id] = (
                    narrative_scores.get(narrative_id, 0.0) + result.score * 0.3
                )

        # For already-retrieved narrative chunks, follow edges to pull in
        # the symbols they reference
        for result in result_list:
            if result.source_type != "narrative":
                continue
            section_id = result.symbol_id or result.section_title
            if not section_id:
                continue
            for sid in graph.narrative_refs_symbol.get(section_id, ()):
                add_score(sid, result.score * 0.3)
                sym_chunk = by_symbol_id.get(sid)
                if sym_chunk is None:
                    node = graph.nodes_by_id.get(sid)
                    if node:
                        sym_chunk = by_qualname.get(_normalize_key(node.qualname))
                if sym_chunk is not None:
                    existing = merged_by_chunk.get(sym_chunk.chunk_id)
                    if existing is not None:
                        merged_by_chunk[sym_chunk.chunk_id] = replace(
                            existing, score=existing.score + result.score * 0.3
                        )
                    else:
                        merged_by_chunk[sym_chunk.chunk_id] = chunk_to_result(
                            sym_chunk, result.score * 0.3
                        )

        # Add top narrative chunks from cross-source expansion
        ranked_narratives = sorted(
            narrative_scores.items(), key=lambda x: x[1], reverse=True
        )[:4]
        for narrative_id, bonus in ranked_narratives:
            chunk = by_narrative_id.get(narrative_id)
            if chunk is None:
                continue
            existing = merged_by_chunk.get(chunk.chunk_id)
            if existing is not None:
                merged_by_chunk[chunk.chunk_id] = replace(
                    existing, score=existing.score + bonus
                )
            else:
                merged_by_chunk[chunk.chunk_id] = chunk_to_result(chunk, bonus)

    merged = list(merged_by_chunk.values())
    merged.sort(key=lambda item: (item.score, item.source_type == "generated"), reverse=True)
    return merged
