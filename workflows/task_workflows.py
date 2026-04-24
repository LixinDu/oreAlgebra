#!/usr/bin/env python3
"""Static task-workflow registry and lightweight selectors for ore_algebra."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Sequence

DEFAULT_TASK_WORKFLOWS_PATH = Path(__file__).resolve().parents[1] / "config" / "task_workflows.json"
SYMBOL_FOCUS_ALIASES = {
    "ct": ("creative telescoping", "telescoper", "certificate", "certificates"),
    "desingularize": ("apparent singularities", "remove apparent singularities"),
    "evaluate": ("evaluate", "function value", "value"),
    "eliminate": ("elimination", "eliminate"),
    "forward_matrix_bsplit": ("forward_matrix_bsplit", "large recurrence term", "large term", "large n"),
    "fglm": ("fglm", "groebner basis", "left ideal"),
    "guess_deq": ("differential equation", "infer an operator"),
    "guess_rec": ("guess recurrence", "infer a recurrence", "recover recurrence", "sample coefficients"),
    "groebner_basis": ("groebner basis", "buchberger", "left ideal"),
    "ideal": ("left ideal", "annihilating ideal", "multivariate ore algebra"),
    "numerical_solution": ("evaluate the solution", "numerically", "initial condition", "initial conditions", "f(0)"),
    "numerical_transition_matrix": ("transition matrix", "continue along a path", "path"),
    "orealgebra": ("ore algebra", "multivariate ore algebra", "several variables", "several generators"),
    "symmetric_power": ("symmetric square", "square of", "square"),
    "to_list": ("first terms", "first values", "sequence terms"),
    "to_s": ("taylor coefficients", "taylor-coefficient recurrence", "coefficient-side recurrence", "coefficient recurrence"),
}


@dataclass(frozen=True)
class CapabilityFamily:
    id: str
    name: str
    summary: str
    keywords: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskWorkflow:
    id: str
    family_id: str
    title: str
    summary: str
    keywords: tuple[str, ...] = ()
    preferred_symbols: tuple[str, ...] = ()
    prompt_rules: tuple[str, ...] = ()
    tuning_params: tuple[str, ...] = ()


@dataclass(frozen=True)
class WorkflowRegistry:
    families: tuple[CapabilityFamily, ...]
    workflows: tuple[TaskWorkflow, ...]


@dataclass(frozen=True)
class WorkflowSelection:
    family_id: str = ""
    family_name: str = ""
    workflow_id: str = ""
    workflow_title: str = ""
    confidence: float = 0.0
    preferred_symbols: tuple[str, ...] = ()
    prompt_rules: tuple[str, ...] = ()
    tuning_params: tuple[str, ...] = ()
    matched_on: tuple[str, ...] = field(default_factory=tuple)

    @property
    def has_family(self) -> bool:
        return bool(self.family_id)

    @property
    def has_workflow(self) -> bool:
        return bool(self.workflow_id)


def _coerce_string_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    out = []
    for item in value:
        text = str(item).strip()
        if text:
            out.append(text)
    return tuple(out)


@lru_cache(maxsize=1)
def load_workflow_registry(path: str | Path | None = None) -> WorkflowRegistry:
    config_path = Path(path).expanduser() if path else DEFAULT_TASK_WORKFLOWS_PATH
    payload = json.loads(config_path.read_text(encoding="utf-8"))

    families: list[CapabilityFamily] = []
    for item in payload.get("families", []):
        if not isinstance(item, dict):
            continue
        family_id = str(item.get("id", "")).strip().upper()
        name = str(item.get("name", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not family_id or not name:
            continue
        families.append(
            CapabilityFamily(
                id=family_id,
                name=name,
                summary=summary,
                keywords=_coerce_string_tuple(item.get("keywords")),
            )
        )

    workflows: list[TaskWorkflow] = []
    for item in payload.get("workflows", []):
        if not isinstance(item, dict):
            continue
        workflow_id = str(item.get("id", "")).strip()
        family_id = str(item.get("family_id", "")).strip().upper()
        title = str(item.get("title", "")).strip()
        summary = str(item.get("summary", "")).strip()
        if not workflow_id or not family_id or not title:
            continue
        workflows.append(
            TaskWorkflow(
                id=workflow_id,
                family_id=family_id,
                title=title,
                summary=summary,
                keywords=_coerce_string_tuple(item.get("keywords")),
                preferred_symbols=_coerce_string_tuple(item.get("preferred_symbols")),
                prompt_rules=_coerce_string_tuple(item.get("prompt_rules")),
                tuning_params=_coerce_string_tuple(item.get("tuning_params")),
            )
        )

    return WorkflowRegistry(families=tuple(families), workflows=tuple(workflows))


def normalize_family_id(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    registry = load_workflow_registry()
    upper = raw.upper()
    if upper in {family.id for family in registry.families}:
        return upper
    compact = upper.replace(".", " ").strip()
    head = compact.split()[0] if compact else ""
    if head in {family.id for family in registry.families}:
        return head
    lower = raw.lower()
    for family in registry.families:
        if lower == family.name.lower():
            return family.id
    return ""


def get_family(family_id: str) -> CapabilityFamily | None:
    family_id = normalize_family_id(family_id)
    if not family_id:
        return None
    for family in load_workflow_registry().families:
        if family.id == family_id:
            return family
    return None


def get_workflows_for_family(family_id: str) -> list[TaskWorkflow]:
    normalized = normalize_family_id(family_id)
    return [wf for wf in load_workflow_registry().workflows if wf.family_id == normalized]


def build_capability_family_prompt_block() -> str:
    registry = load_workflow_registry()
    family_ids = ", ".join(family.id for family in registry.families) or "none"
    lines = [f"Capability families (tag each subtask with exactly one family_id from: {family_ids}):"]
    for family in registry.families:
        lines.append(f"- {family.id}. {family.name} - {family.summary}")
    return "\n".join(lines)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    return " ".join(lowered.split())


def _focused_symbol_hint_score(question_haystack: str, symbol: str) -> int:
    tail = symbol.rsplit(".", 1)[-1].lower()
    score = 0
    if tail and tail in question_haystack:
        score += 4
    for phrase in SYMBOL_FOCUS_ALIASES.get(tail, ()):
        if _normalize_text(phrase) in question_haystack:
            score += 3
    for token in re.split(r"[^a-z0-9]+", tail):
        if len(token) <= 2:
            continue
        if token in question_haystack:
            score += 1
    return score


def _keyword_hits(haystack: str, keywords: Sequence[str], weight: int = 1) -> tuple[int, tuple[str, ...]]:
    hits: list[str] = []
    score = 0
    for keyword in keywords:
        normalized = _normalize_text(keyword)
        if not normalized:
            continue
        if normalized in haystack:
            hits.append(keyword)
            score += weight + (1 if " " in normalized else 0)
            continue
        if " " in normalized:
            tokens = [token for token in normalized.split() if token]
            if tokens and all(token in haystack for token in tokens):
                hits.append(keyword)
                score += weight
    return score, tuple(hits)


def _context_haystack(items: Iterable[object] | None) -> str:
    parts: list[str] = []
    for item in items or ():
        for attr in ("qualname", "symbol_id", "module", "title", "location", "text"):
            value = getattr(item, attr, "")
            text = str(value).strip()
            if text:
                parts.append(text)
    return _normalize_text("\n".join(parts))


def _looks_like_out_of_scope_task(question_haystack: str) -> bool:
    if not question_haystack:
        return False
    direct_markers = (
        "favorite color",
        "write a poem",
        "poem about",
        "what this app can do",
        "install ore_algebra",
        "install ore algebra",
    )
    if any(marker in question_haystack for marker in direct_markers):
        return True
    if "prove" in question_haystack and (
        "theorem" in question_haystack
        or "conjecture" in question_haystack
        or "new result" in question_haystack
    ):
        return True
    return False


def _looks_like_conceptual_meta_prompt(question_haystack: str) -> bool:
    if not question_haystack:
        return False
    conceptual_markers = (
        "conceptually",
        "at a high level",
        "high level",
    )
    if not any(marker in question_haystack for marker in conceptual_markers):
        return False
    action_markers = (
        "compute",
        "evaluate",
        "find",
        "guess",
        "construct",
        "create",
        "convert",
        "desingularize",
        "factor",
        "gcrd",
        "lclm",
        "transition matrix",
        "first terms",
        "annihilator",
        "numerical",
        "to_s",
        "to_d",
        "to_f",
        "to_t",
    )
    return not any(marker in question_haystack for marker in action_markers)


def _looks_like_family_meta_question(question_haystack: str) -> bool:
    if not question_haystack:
        return False
    family_markers = (
        "which capability family",
        "what capability family",
        "which workflow family",
        "what workflow family",
        "which family covers",
        "what family covers",
        "which category covers",
        "what category covers",
        "which branch covers",
        "what branch covers",
    )
    return any(marker in question_haystack for marker in family_markers)


def choose_workflow(
    *,
    question: str,
    context_items: Iterable[object] | None = None,
    family_hint: str = "",
) -> WorkflowSelection:
    registry = load_workflow_registry()
    question_haystack = _normalize_text(question)
    context_haystack = _context_haystack(context_items)
    forced_family = get_family(family_hint)
    if forced_family is None and _looks_like_out_of_scope_task(question_haystack):
        return WorkflowSelection()
    if forced_family is None and _looks_like_conceptual_meta_prompt(question_haystack):
        return WorkflowSelection()
    family_scores: list[tuple[int, CapabilityFamily, tuple[str, ...]]] = []
    for family in registry.families:
        score_q, hits_q = _keyword_hits(question_haystack, family.keywords, weight=3)
        score_ctx, hits_ctx = _keyword_hits(context_haystack, family.keywords, weight=1)
        score = score_q + score_ctx
        hits = hits_q + tuple(hit for hit in hits_ctx if hit not in hits_q)
        if forced_family and family.id == forced_family.id:
            score += 6
        family_scores.append((score, family, hits))
    family_scores.sort(key=lambda item: item[0], reverse=True)

    best_family: CapabilityFamily | None = forced_family
    best_family_hits: tuple[str, ...] = ()
    if best_family is None and family_scores and family_scores[0][0] > 0:
        _, best_family, best_family_hits = family_scores[0]
    elif forced_family is not None:
        for score, family, hits in family_scores:
            if family.id == forced_family.id:
                best_family_hits = hits
                break

    if forced_family is None and best_family is not None and _looks_like_family_meta_question(question_haystack):
        return WorkflowSelection(
            family_id=best_family.id,
            family_name=best_family.name,
            confidence=0.4,
            matched_on=best_family_hits,
        )

    candidates = [wf for wf in registry.workflows if wf.family_id == best_family.id] if forced_family else list(registry.workflows)
    scored_workflows: list[tuple[int, TaskWorkflow, tuple[str, ...]]] = []
    for workflow in candidates:
        score_q, hits_q = _keyword_hits(question_haystack, workflow.keywords, weight=4)
        score_ctx, hits_ctx = _keyword_hits(context_haystack, workflow.keywords, weight=2)
        score_syms, hits_syms = _keyword_hits(context_haystack, workflow.preferred_symbols, weight=3)
        score = score_q + score_ctx + score_syms
        hits = hits_q + tuple(hit for hit in hits_ctx + hits_syms if hit not in hits_q)
        if score > 0:
            scored_workflows.append((score, workflow, hits))
    scored_workflows.sort(key=lambda item: item[0], reverse=True)

    if not scored_workflows:
        if best_family is None:
            return WorkflowSelection()
        return WorkflowSelection(
            family_id=best_family.id,
            family_name=best_family.name,
            confidence=0.35,
            matched_on=best_family_hits,
        )

    top_score, top_workflow, top_hits = scored_workflows[0]
    best_family = get_family(top_workflow.family_id) or best_family
    runner_up = scored_workflows[1][0] if len(scored_workflows) > 1 else 0
    confidence = 0.55
    if top_score > 0:
        confidence = min(0.98, 0.55 + max(top_score - runner_up, 0) / max(top_score + 6, 1))

    return WorkflowSelection(
        family_id=best_family.id,
        family_name=best_family.name,
        workflow_id=top_workflow.id,
        workflow_title=top_workflow.title,
        confidence=confidence,
        preferred_symbols=top_workflow.preferred_symbols,
        prompt_rules=top_workflow.prompt_rules,
        tuning_params=top_workflow.tuning_params,
        matched_on=top_hits if top_hits else best_family_hits,
    )


def build_workflow_prompt_hint(selection: WorkflowSelection) -> str:
    if not selection.has_family:
        return ""

    lines = [
        "Task workflow guidance:",
        f"- capability_family: {selection.family_id}. {selection.family_name}",
    ]
    if selection.has_workflow:
        lines.append(f"- selected_workflow: {selection.workflow_id} - {selection.workflow_title}")
    if selection.preferred_symbols:
        lines.append(f"- preferred_symbols: {', '.join(selection.preferred_symbols)}")
    if selection.tuning_params:
        lines.append(f"- tuning_params: {', '.join(selection.tuning_params)}")
    if selection.prompt_rules:
        lines.append("- important_rules:")
        lines.extend(f"  - {rule}" for rule in selection.prompt_rules[:4])
    return "\n".join(lines)


def build_workflow_retrieval_queries(
    *,
    question: str,
    selection: WorkflowSelection | None = None,
    family_hint: str = "",
    max_symbol_hints: int = 3,
) -> tuple[str, ...]:
    base_query = str(question or "").strip()
    if not base_query:
        return ()

    active_selection = selection or choose_workflow(question=base_query, family_hint=family_hint)
    queries: list[str] = []
    seen: set[str] = set()

    def add_query(raw: str) -> None:
        text = str(raw or "").strip()
        if not text:
            return
        normalized = _normalize_text(text)
        if not normalized or normalized in seen:
            return
        seen.add(normalized)
        queries.append(text)

    add_query(base_query)

    if active_selection.has_workflow:
        symbol_hints = list(active_selection.preferred_symbols[: max(0, max_symbol_hints)])
        parts = [base_query, "ore_algebra"]
        if active_selection.workflow_title:
            parts.append(active_selection.workflow_title)
        parts.extend(symbol_hints)
        add_query(" ".join(parts))
        if symbol_hints:
            question_haystack = _normalize_text(base_query)
            focused_symbol = max(symbol_hints, key=lambda symbol: _focused_symbol_hint_score(question_haystack, symbol))
            if _focused_symbol_hint_score(question_haystack, focused_symbol) > 0:
                add_query(f"{base_query} ore_algebra {focused_symbol}")
    elif active_selection.has_family:
        add_query(f"{base_query} ore_algebra {active_selection.family_name}")

    return tuple(queries)
