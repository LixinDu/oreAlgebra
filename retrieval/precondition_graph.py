#!/usr/bin/env python3
"""Precondition graph artifact for ore_algebra code generation.

This module builds and loads a small derived artifact (default
``.rag/ore_algebra_precondition_graph.json``) that records the
information needed to fill prerequisites when generating Sage code:

- per-symbol records keyed by ``symbol_id`` (qualname, module, kind,
  owner class, best-effort top-level import target),
- a tail-name index mapping bare names like ``right_factors`` to the
  list of symbol ids that own them,
- a class-to-methods index used for "if a method is called, also
  surface its owning class",
- a curated list of symbol names that are importable directly from
  the top-level ``ore_algebra`` package, used by the import resolver.

The artifact is intentionally derived from ``generated/symbols.jsonl``
only. Workflow-specific setup templates live in
:mod:`code_plan` because they are curated, not derived.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence


# Symbols known to be exposed at the top of the ``ore_algebra`` package
# (i.e. ``from ore_algebra import OreAlgebra`` works). The dictionary
# value is the canonical import statement to inject. Anything not in
# this map falls back to "no import added" (the runtime already
# prepends ``from ore_algebra import *``).
TOPLEVEL_IMPORTABLE: dict[str, str] = {
    "OreAlgebra": "from ore_algebra import OreAlgebra",
    "DifferentialOperators": "from ore_algebra import DifferentialOperators",
    "guess": "from ore_algebra import guess",
    "guess_rec": "from ore_algebra.guessing import guess_rec",
    "guess_deq": "from ore_algebra.guessing import guess_deq",
    "guess_raw": "from ore_algebra.guessing import guess_raw",
}

# Names referenced from a code body that should never be treated as
# user identifiers needing constructor checks.
BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "print",
        "range",
        "len",
        "list",
        "dict",
        "set",
        "tuple",
        "int",
        "float",
        "str",
        "bool",
        "abs",
        "max",
        "min",
        "sum",
        "zip",
        "map",
        "filter",
        "sorted",
        "enumerate",
        "any",
        "all",
        # Sage globals injected by ``from sage.all import *``.
        "QQ",
        "ZZ",
        "RR",
        "CC",
        "QQbar",
        "ComplexField",
        "RealField",
        "PolynomialRing",
        "FractionField",
        "Matrix",
        "vector",
        "PowerSeriesRing",
        "Integer",
        "Rational",
        "I",
        "pi",
        "var",
    }
)

# Schema version for the on-disk artifact.
SCHEMA_VERSION = 1

DEFAULT_GRAPH_FILENAME = "ore_algebra_precondition_graph.json"

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass(frozen=True)
class SymbolNode:
    """One row in the precondition graph."""

    symbol_id: str
    qualname: str
    module: str
    kind: str  # "class" | "method" | "function"
    owner_class: str  # bare class name, e.g. "OreAlgebra"; empty for top-level
    owner_qualname: str  # full qualname of the owner class, empty for top-level
    tail: str  # last component of qualname
    import_statement: str  # canonical import line, "" if none known


@dataclass(frozen=True)
class PreconditionGraph:
    nodes: dict[str, SymbolNode]
    by_tail: dict[str, tuple[str, ...]]
    methods_by_class: dict[str, tuple[str, ...]]
    generated_at: str = ""
    source_path: str = ""

    def lookup_by_tail(self, tail: str) -> tuple[SymbolNode, ...]:
        ids = self.by_tail.get(_normalize_tail(tail), ())
        return tuple(self.nodes[i] for i in ids if i in self.nodes)

    def class_node(self, class_name: str) -> SymbolNode | None:
        normalized = (class_name or "").strip()
        if not normalized:
            return None
        for node in self.nodes.values():
            if node.kind == "class" and (
                node.qualname == normalized or node.qualname.endswith("." + normalized)
            ):
                return node
        return None

    def methods_of(self, class_name: str) -> tuple[SymbolNode, ...]:
        normalized = (class_name or "").strip()
        ids = self.methods_by_class.get(normalized, ())
        return tuple(self.nodes[i] for i in ids if i in self.nodes)

    def imports_for_names(self, names: Iterable[str]) -> tuple[str, ...]:
        """Return canonical imports needed for the bare names referenced.

        Only names matching :data:`TOPLEVEL_IMPORTABLE` are returned.
        Order matches first appearance in *names*.
        """

        seen: set[str] = set()
        ordered: list[str] = []
        for name in names:
            stmt = TOPLEVEL_IMPORTABLE.get(str(name).strip())
            if not stmt or stmt in seen:
                continue
            seen.add(stmt)
            ordered.append(stmt)
        return tuple(ordered)


def _normalize_tail(value: object) -> str:
    return str(value or "").strip()


def _split_qualname(qualname: str) -> tuple[str, str, str]:
    """Return ``(owner_qualname, owner_tail, tail)`` for a qualname.

    For top-level functions/classes both owner fields are empty.
    """

    text = str(qualname or "").strip()
    if not text:
        return "", "", ""
    if "." not in text:
        return "", "", text
    owner_qualname, tail = text.rsplit(".", 1)
    owner_tail = owner_qualname.rsplit(".", 1)[-1]
    return owner_qualname, owner_tail, tail


def _import_for(qualname: str, kind: str) -> str:
    """Return a canonical top-level import statement, or '' if unknown."""

    text = str(qualname or "").strip()
    if not text:
        return ""
    head = text.split(".", 1)[0]
    # For methods we want the import for the owning class, not the method
    # itself; methods are reached via instance access, not direct import.
    if kind == "method":
        return TOPLEVEL_IMPORTABLE.get(head, "")
    return TOPLEVEL_IMPORTABLE.get(head, TOPLEVEL_IMPORTABLE.get(text, ""))


def build_precondition_graph_from_records(
    records: Iterable[Mapping[str, object]],
    *,
    source_path: str = "",
) -> PreconditionGraph:
    """Build a :class:`PreconditionGraph` from raw symbol records."""

    nodes: dict[str, SymbolNode] = {}
    tail_index: dict[str, list[str]] = {}
    methods_by_class: dict[str, list[str]] = {}

    for raw in records:
        if not isinstance(raw, Mapping):
            continue
        symbol_id = str(raw.get("id", "")).strip()
        qualname = str(raw.get("qualname", "")).strip()
        kind = str(raw.get("kind", "")).strip()
        module = str(raw.get("module", "")).strip()
        if not symbol_id or not qualname or not kind:
            continue
        owner_qualname, owner_tail, tail = _split_qualname(qualname)
        node = SymbolNode(
            symbol_id=symbol_id,
            qualname=qualname,
            module=module,
            kind=kind,
            owner_class=owner_tail if kind == "method" else "",
            owner_qualname=owner_qualname if kind == "method" else "",
            tail=tail,
            import_statement=_import_for(qualname, kind),
        )
        nodes[symbol_id] = node
        tail_index.setdefault(tail, []).append(symbol_id)
        if kind == "method" and owner_tail:
            methods_by_class.setdefault(owner_tail, []).append(symbol_id)

    return PreconditionGraph(
        nodes=nodes,
        by_tail={k: tuple(v) for k, v in tail_index.items()},
        methods_by_class={k: tuple(v) for k, v in methods_by_class.items()},
        generated_at=_utc_now_iso(),
        source_path=source_path,
    )


def build_precondition_graph_from_file(symbols_path: str | Path) -> PreconditionGraph:
    path = Path(symbols_path).expanduser().resolve()
    records: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                records.append(payload)
    return build_precondition_graph_from_records(records, source_path=str(path))


def graph_to_payload(graph: PreconditionGraph) -> dict[str, object]:
    return {
        "version": SCHEMA_VERSION,
        "generated_at": graph.generated_at,
        "source_path": graph.source_path,
        "nodes": {sid: asdict(node) for sid, node in graph.nodes.items()},
        "by_tail": {tail: list(ids) for tail, ids in graph.by_tail.items()},
        "methods_by_class": {
            cls: list(ids) for cls, ids in graph.methods_by_class.items()
        },
    }


def payload_to_graph(payload: Mapping[str, object]) -> PreconditionGraph:
    raw_nodes = payload.get("nodes") or {}
    nodes: dict[str, SymbolNode] = {}
    if isinstance(raw_nodes, Mapping):
        for sid, item in raw_nodes.items():
            if not isinstance(item, Mapping):
                continue
            nodes[str(sid)] = SymbolNode(
                symbol_id=str(item.get("symbol_id", sid)),
                qualname=str(item.get("qualname", "")),
                module=str(item.get("module", "")),
                kind=str(item.get("kind", "")),
                owner_class=str(item.get("owner_class", "")),
                owner_qualname=str(item.get("owner_qualname", "")),
                tail=str(item.get("tail", "")),
                import_statement=str(item.get("import_statement", "")),
            )
    raw_tail = payload.get("by_tail") or {}
    by_tail: dict[str, tuple[str, ...]] = {}
    if isinstance(raw_tail, Mapping):
        for key, value in raw_tail.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                by_tail[str(key)] = tuple(str(v) for v in value)
    raw_methods = payload.get("methods_by_class") or {}
    methods_by_class: dict[str, tuple[str, ...]] = {}
    if isinstance(raw_methods, Mapping):
        for key, value in raw_methods.items():
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                methods_by_class[str(key)] = tuple(str(v) for v in value)
    return PreconditionGraph(
        nodes=nodes,
        by_tail=by_tail,
        methods_by_class=methods_by_class,
        generated_at=str(payload.get("generated_at", "")),
        source_path=str(payload.get("source_path", "")),
    )


def write_precondition_graph(path: str | Path, graph: PreconditionGraph) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = graph_to_payload(graph)
    target.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return target


def load_precondition_graph(path: str | Path) -> PreconditionGraph:
    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid precondition graph payload at {target}")
    return payload_to_graph(payload)


def extract_referenced_names(code: str) -> tuple[str, ...]:
    """Return the set of identifier-like tokens used in *code*.

    This is a deliberately shallow extractor (regex-based) used by the
    import resolver and the precondition validator. Returns names in
    insertion order.
    """

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _NAME_RE.finditer(code or ""):
        name = match.group(0)
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    return tuple(ordered)
