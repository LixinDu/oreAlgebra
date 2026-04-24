#!/usr/bin/env python3
"""Shared runner for the full routing -> retrieval -> codegen -> exec LLM pipeline.

This module now follows the same core algorithmic flow as ``ui/streamlit_chat_app.py``:
- adaptive route selection (auto/fast/plan)
- workflow-aware retrieval
- code generation via deterministic executors/LLM
- preflight-aware code repair and wildcard import retry
- optional auto escalation from fast path to plan path on evidence gaps
- final answer synthesis (direct-from-Sage on success, LLM-backed otherwise)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def run_sage_code(code: str, timeout: int = 60) -> dict:
    """Execute code via the project's Sage runtime. Returns dict with status/stdout/stderr."""
    try:
        from core.sage_runtime import validate_and_run_sage

        result = validate_and_run_sage(code, timeout=timeout)
        return {
            "status": result.status,
            "stdout": result.stdout_full,
            "stderr": result.stderr,
            "validation_errors": list(result.validation_errors),
            "executed_code": getattr(result, "executed_code", ""),
        }
    except Exception as exc:
        return {
            "status": "error",
            "stdout": "",
            "stderr": str(exc),
            "validation_errors": [],
            "executed_code": "",
        }


DEFAULT_LLM_SETTINGS: dict[str, Any] = {
    "provider": "openai",
    "model": "gpt-4o-mini",
    "temperature": 0.1,
    "max_output_tokens": None,
    "run_mode": "auto",  # auto | fast | plan
}

_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _to_context_items(results: list, pdf_char_limit: int = 1600) -> list:
    from core.llm_service import ContextItem
    from core.ore_rag_assistant import location_label

    items: list[ContextItem] = []
    for idx, hit in enumerate(results, start=1):
        title = hit.qualname or hit.symbol_id or hit.section_title or hit.source
        text = hit.text if hit.source_type == "generated" else hit.text[:pdf_char_limit]
        items.append(
            ContextItem(
                context_id=f"ctx_{idx}",
                source_type=hit.source_type,
                title=title,
                location=location_label(hit),
                text=text,
                score=hit.score,
            )
        )
    return items


def _dedupe_results_by_chunk(results: list) -> list:
    seen = set()
    out = []
    for result in results:
        chunk_id = getattr(result, "chunk_id", None)
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        out.append(result)
    return out


def _normalize_module_for_auto_import(module: str) -> str:
    mod = (module or "").strip()
    if not mod:
        return ""
    if mod.startswith("ore_algebra."):
        return mod
    if mod == "ore_algebra":
        return mod
    if "." in mod or mod.isidentifier():
        return f"ore_algebra.{mod}"
    return ""


def _preferred_symbol_for_auto_import(result: Any) -> str:
    source = (getattr(result, "qualname", "") or getattr(result, "symbol_id", "") or "").strip()
    if not source:
        return ""
    parts = [part for part in source.split(".") if part]
    if not parts:
        return ""
    kind = (getattr(result, "kind", "") or "").strip().lower()
    candidate = parts[-2] if kind == "method" and len(parts) >= 2 else parts[-1]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate):
        return ""
    return candidate


def _has_explicit_import_for_name(code: str, name: str) -> bool:
    if not name:
        return False
    escaped = re.escape(name)
    return bool(
        re.search(rf"^\s*from\s+[A-Za-z0-9_\.]+\s+import\s+.*\b{escaped}\b", code, flags=re.MULTILINE)
        or re.search(rf"^\s*import\s+.*\b{escaped}\b", code, flags=re.MULTILINE)
    )


def _is_name_defined_in_code(code: str, name: str) -> bool:
    if not name:
        return False
    escaped = re.escape(name)
    return bool(
        re.search(rf"^\s*def\s+{escaped}\b", code, flags=re.MULTILINE)
        or re.search(rf"^\s*class\s+{escaped}\b", code, flags=re.MULTILINE)
        or re.search(rf"^\s*{escaped}\s*=", code, flags=re.MULTILINE)
    )


def _insert_import_lines(code: str, import_lines: list[str]) -> str:
    if not import_lines:
        return code
    lines = code.splitlines()
    insert_at = 0
    while insert_at < len(lines) and (
        not lines[insert_at].strip()
        or lines[insert_at].startswith("#!")
        or lines[insert_at].startswith("# -*-")
    ):
        insert_at += 1
    while insert_at < len(lines) and (
        lines[insert_at].startswith("from ") or lines[insert_at].startswith("import ")
    ):
        insert_at += 1
    out = lines[:insert_at] + import_lines + lines[insert_at:]
    return "\n".join(out).strip()


def _augment_code_with_retrieval_imports(
    code: str,
    retrieved_results: list,
    max_added_imports: int = 6,
) -> tuple[str, list[str]]:
    raw = code.strip()
    if not raw:
        return code, []
    added: list[str] = []
    seen = set()
    for result in retrieved_results:
        if getattr(result, "source_type", "") != "generated":
            continue
        module = _normalize_module_for_auto_import(getattr(result, "module", ""))
        symbol = _preferred_symbol_for_auto_import(result)
        if not module or not symbol:
            continue
        if module == "ore_algebra":
            continue
        if not re.search(rf"\b{re.escape(symbol)}\b", raw):
            continue
        if _has_explicit_import_for_name(raw, symbol):
            continue
        if _is_name_defined_in_code(raw, symbol):
            continue
        import_line = f"from {module} import {symbol}"
        if import_line in seen:
            continue
        seen.add(import_line)
        added.append(import_line)
        if len(added) >= max_added_imports:
            break
    if not added:
        return code, []
    return _insert_import_lines(raw, added), added


def _candidate_wildcard_modules(
    retrieved_results: list,
    max_modules: int = 2,
) -> list[str]:
    out: list[str] = []
    seen = set()
    for result in retrieved_results:
        if getattr(result, "source_type", "") != "generated":
            continue
        module = _normalize_module_for_auto_import(getattr(result, "module", ""))
        if not module or module == "ore_algebra":
            continue
        if module in seen:
            continue
        seen.add(module)
        out.append(module)
        if len(out) >= max_modules:
            break
    return out


def _augment_code_with_module_wildcards(code: str, modules: list[str]) -> tuple[str, list[str]]:
    raw = code.strip()
    if not raw or not modules:
        return code, []
    added: list[str] = []
    for module in modules:
        line = f"from {module} import *"
        if re.search(rf"^\s*from\s+{re.escape(module)}\s+import\s+\*", raw, flags=re.MULTILINE):
            continue
        added.append(line)
    if not added:
        return code, []
    return _insert_import_lines(raw, added), added


def _needs_wildcard_import_fallback(execution_result: Any | None) -> bool:
    if execution_result is None:
        return False
    if getattr(execution_result, "status", "") != "error":
        return False
    stderr = (getattr(execution_result, "stderr", "") or "").lower()
    markers = (
        "nameerror",
        "not defined",
        "importerror",
        "no module named",
        "cannot import name",
        "attributeerror",
    )
    return any(marker in stderr for marker in markers)


def _needs_generator_variable_repair(validation_errors: list[str]) -> bool:
    for item in validation_errors:
        item_lower = str(item).lower()
        if "generator/base-variable mismatch" in item_lower:
            return True
        if "generator binding/access issue" in item_lower:
            return True
        if "unbound generator issue" in item_lower:
            return True
    return False


def _expected_evidence_type(question: str) -> str:
    lower = (question or "").lower()
    pdf_markers = (
        "pdf",
        "paper",
        "user guide",
        "documentation",
        "docs",
        "manual",
        "section",
        "theorem",
        "proof",
        "algorithmic technique",
    )
    if any(marker in lower for marker in pdf_markers):
        return "symbol+pdf"
    return "symbol-only"


def _question_complexity_score(question: str) -> tuple[int, dict[str, int]]:
    q = (question or "").strip()
    lower = q.lower()
    words = re.findall(r"[A-Za-z0-9_]+", q)
    dotted_apis = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*\.[A-Za-z_][A-Za-z0-9_]*", q))
    known_api_hits = 0
    for marker in ("orealgebra", "differentialoperators", "annihilator", "telescoping", "recurrence"):
        if marker in lower:
            known_api_hits += 1
    multi_goal_markers = (
        " and ",
        " then ",
        " after ",
        " compare ",
        " both ",
        " first ",
        " second ",
        " finally ",
    )
    constraint_markers = (
        " with ",
        " without ",
        " subject to ",
        " verify ",
        " prove ",
        " show that ",
        " expected ",
        " must ",
    )
    goal_hits = sum(1 for marker in multi_goal_markers if marker in f" {lower} ")
    constraint_hits = sum(1 for marker in constraint_markers if marker in f" {lower} ")
    score = 0
    score += min(len(dotted_apis) + known_api_hits, 3)
    if len(words) > 20:
        score += 1
    if len(words) > 40:
        score += 1
    if goal_hits > 0:
        score += min(goal_hits, 2)
    if constraint_hits > 0:
        score += min(constraint_hits, 2)
    details = {
        "word_count": len(words),
        "api_refs": len(dotted_apis) + known_api_hits,
        "goal_markers": goal_hits,
        "constraint_markers": constraint_hits,
    }
    return score, details


def _normalized_retrieval_score(value: float) -> float:
    try:
        score = float(value)
    except Exception:
        return 0.0
    if score <= 0.0:
        return 0.0
    if score <= 1.0:
        return score
    return score / (score + 1.0)


def _retrieval_confidence(results: list) -> float:
    if not results:
        return 0.0
    top = results[: min(4, len(results))]
    scores = [_normalized_retrieval_score(getattr(r, "score", 0.0)) for r in top]
    confidence = sum(scores) / max(len(scores), 1)
    if sum(1 for r in top if getattr(r, "source_type", "") == "generated") >= 2:
        confidence = min(1.0, confidence + 0.03)
    return confidence


def _route_adaptive(
    *,
    complexity_score: int,
    confidence: float,
    expected_evidence_type: str,
    precheck_results: list,
) -> tuple[str, str]:
    top = precheck_results[: min(5, len(precheck_results))]
    has_pdf = any(getattr(r, "source_type", "") == "pdf" for r in top)
    if expected_evidence_type == "symbol+pdf" and not has_pdf:
        return "plan", "Question appears PDF/doc-dependent but first-pass retrieval lacks PDF evidence."
    if complexity_score <= 3 and confidence >= 0.60:
        return "auto_fast", "Low complexity and high first-pass retrieval confidence."
    if complexity_score <= 6 and confidence >= 0.30:
        return "auto", "Medium complexity or moderate confidence; allow one targeted retry."
    return "plan", "High complexity or low confidence; use plan flow."


def _top_symbol_hints(results: list, limit: int = 3) -> list[str]:
    out: list[str] = []
    for result in results:
        if getattr(result, "source_type", "") != "generated":
            continue
        hint = (getattr(result, "qualname", "") or getattr(result, "symbol_id", "") or "").strip()
        if not hint:
            continue
        if hint in out:
            continue
        out.append(hint)
        if len(out) >= limit:
            break
    return out


def _build_auto_retry_query(
    *,
    question: str,
    results: list,
    expected_evidence_type: str,
    missing_hints: list[str] | None = None,
) -> str:
    pieces: list[str] = [question]
    hints = [h.strip() for h in (missing_hints or []) if h and h.strip()]
    if hints:
        pieces.append("missing:")
        pieces.extend(hints[:2])
    symbol_hints = _top_symbol_hints(results, limit=2)
    if symbol_hints:
        pieces.append("related_apis:")
        pieces.extend(symbol_hints)
    if expected_evidence_type == "symbol+pdf":
        pieces.extend(["user guide", "documentation section"])
    else:
        pieces.extend(["runnable example", "ore_algebra"])
    return " ".join(pieces)


def _looks_like_evidence_gap(execution_result: Any | None) -> bool:
    if execution_result is None:
        return False
    if getattr(execution_result, "status", "") == "success":
        return False
    text = (
        "\n".join(getattr(execution_result, "validation_errors", []) or [])
        + "\n"
        + (getattr(execution_result, "stderr", "") or "")
    )
    lower = text.lower()
    non_gap_markers = (
        "blocked import detected",
        "blocked builtin",
        "execution timed out",
        "timed out after",
        "code is empty",
        "syntaxerror",
    )
    if any(marker in lower for marker in non_gap_markers):
        return False
    gap_markers = (
        "nameerror",
        "not defined",
        "attributeerror",
        "has no attribute",
        "importerror",
        "no module named",
        "unknown",
        "cannot import name",
    )
    return any(marker in lower for marker in gap_markers)


def _run_retrieval_for_query(
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
    workflow_selection_override: Any | None = None,
    strategy: str = "classic",
    graph_path: str = "",
) -> tuple[str, list, Any, tuple[str, ...]]:
    from retrieval.workflow_retrieval import run_workflow_retrieval

    return run_workflow_retrieval(
        query=query,
        payload=payload,
        chunks=chunks,
        k=k,
        mode=mode,
        index_path=index_path,
        hybrid_alpha=hybrid_alpha,
        source_priority=source_priority,
        symbols_ratio=symbols_ratio,
        max_pdf_extras=max_pdf_extras,
        family_hint=family_hint,
        workflow_selection_override=workflow_selection_override,
        strategy=strategy,
        graph_path=graph_path,
    )


def _collect_context_auto(
    *,
    question: str,
    payload: dict,
    chunks: list,
    index_path: Path,
    k: int,
    mode: str,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
    retrieval_strategy: str,
    graph_path: str,
    final_context_cap: int,
    retry_budget: int,
    expected_evidence_type: str,
    initial_results: list | None = None,
    workflow_selection_override: Any | None = None,
) -> tuple[list, dict[int, int]]:
    aggregated_results: list = []
    chunk_step_map: dict[int, int] = {}
    query = question
    cached_first = list(initial_results or [])
    for attempt in range(max(retry_budget, 0) + 1):
        if attempt == 0 and cached_first:
            results = cached_first
        else:
            _, results, _, _ = _run_retrieval_for_query(
                query=query,
                payload=payload,
                chunks=chunks,
                k=k,
                mode=mode,
                index_path=index_path,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                workflow_selection_override=workflow_selection_override,
                strategy=retrieval_strategy,
                graph_path=graph_path,
            )
        aggregated_results.extend(results)
        for result in results:
            chunk_step_map.setdefault(getattr(result, "chunk_id", -1), attempt + 1)
        if attempt >= retry_budget:
            continue
        retry_query = _build_auto_retry_query(
            question=question,
            results=results,
            expected_evidence_type=expected_evidence_type,
        ).strip()
        if not retry_query or retry_query == query.strip():
            break
        query = retry_query
    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_cap]
    return aggregated_results, chunk_step_map


def _collect_context_plan(
    *,
    question: str,
    planning_hint: str,
    payload: dict,
    chunks: list,
    index_path: Path,
    k: int,
    mode: str,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
    retrieval_strategy: str,
    graph_path: str,
    final_context_cap: int,
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    max_plan_steps: int,
    temperature: float,
    max_output_tokens: int | None,
) -> tuple[list, dict[int, int], str]:
    from core.llm_service import decide_next_action, plan_subtasks

    planning_question = question.strip()
    if planning_hint.strip():
        planning_question = (
            f"{planning_question}\n\nPrior attempt showed missing evidence:\n{planning_hint.strip()}"
        )
    try:
        plan = plan_subtasks(
            question=planning_question,
            provider=provider,
            model=model,
            api_key=api_key or None,
            base_url=base_url,
            max_steps=max_plan_steps,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
    except Exception as exc:
        return [], {}, f"Planning failed: {exc}"
    if not plan.subtasks:
        return [], {}, "Planner returned no subtasks."

    aggregated_results: list = []
    chunk_step_map: dict[int, int] = {}
    for subtask in plan.subtasks:
        step_query = subtask.retrieval_query or subtask.instruction or subtask.title
        _, results, _, _ = _run_retrieval_for_query(
            query=step_query,
            payload=payload,
            chunks=chunks,
            k=k,
            mode=mode,
            index_path=index_path,
            hybrid_alpha=hybrid_alpha,
            source_priority=source_priority,
            symbols_ratio=symbols_ratio,
            max_pdf_extras=max_pdf_extras,
            family_hint=subtask.family_id,
            strategy=retrieval_strategy,
            graph_path=graph_path,
        )
        context_items = _to_context_items(results=results, pdf_char_limit=1600)
        try:
            decision = decide_next_action(
                question=question,
                current_step=subtask,
                context_items=context_items,
                provider=provider,
                model=model,
                api_key=api_key or None,
                base_url=base_url,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
        except Exception:
            decision = None

        step_effective_results = results
        next_query = (getattr(decision, "next_query", "") or "").strip() if decision is not None else ""
        action = (getattr(decision, "action", "") or "").strip().lower() if decision is not None else ""
        if action == "refine_query" and next_query:
            _, refined_results, _, _ = _run_retrieval_for_query(
                query=next_query,
                payload=payload,
                chunks=chunks,
                k=k,
                mode=mode,
                index_path=index_path,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                family_hint=subtask.family_id,
                strategy=retrieval_strategy,
                graph_path=graph_path,
            )
            step_effective_results = refined_results

        aggregated_results.extend(step_effective_results)
        for result in step_effective_results:
            chunk_step_map.setdefault(getattr(result, "chunk_id", -1), int(getattr(subtask, "step_id", 0) or 0))

        if action == "stop":
            break

    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_cap]
    return aggregated_results, chunk_step_map, ""


def _order_contexts_cited_first(
    context_items: list,
    cited_context_ids: list[str],
) -> list:
    if not context_items or not cited_context_ids:
        return list(context_items)
    cited_ids = set(cited_context_ids)
    cited_items: list = []
    uncited_items: list = []
    for item in context_items:
        if item.context_id in cited_ids:
            cited_items.append(item)
        else:
            uncited_items.append(item)
    return cited_items + uncited_items


def _compact_nonempty_lines(text: str) -> list[str]:
    lines: list[str] = []
    for raw in str(text or "").splitlines():
        line = raw.rstrip()
        if line.strip():
            lines.append(line)
    return lines


def _direct_sage_answer(execution_result: Any | None, execution_skipped_reason: str) -> str:
    if execution_result is None:
        return execution_skipped_reason.strip() or "Execution was skipped, so there is no runtime result to display."
    if getattr(execution_result, "status", "") != "success":
        return f"Sage execution ended with status `{getattr(execution_result, 'status', '')}`."
    stdout = (
        getattr(execution_result, "stdout_summary", "")
        or getattr(execution_result, "stdout_full", "")
        or ""
    ).strip()
    if not stdout:
        return "Sage execution succeeded, but produced no visible stdout output."
    lines = _compact_nonempty_lines(stdout)
    if not lines:
        return "Sage execution succeeded, but output was empty after trimming whitespace."
    if len(lines) == 1:
        return f"Sage execution succeeded. Result: {lines[0]}"
    return f"Sage execution succeeded with {len(lines)} non-empty output lines."


def run_llm_pipeline(
    question: str,
    workflow_id: str = "",
    *,
    sage_timeout: int = 60,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
    run_mode: str = "auto",
    use_structured_fast_mode: bool = False,
    use_structured_auto_fast_mode: bool = False,
) -> dict[str, Any]:
    """Run a chat-parity retrieval->codegen->execution pipeline on a question.

    ``run_mode`` mirrors the chat app behavior:
      - ``"fast"``: fast retrieval flow with bounded retry.
      - ``"plan"``: plan/refine retrieval flow.
      - ``"auto"``: adaptive route selection between fast/plan.
    """
    from core.llm_service import (
        CodeGenerationRequest,
        ExecutionAwareAnswerRequest,
        answer_with_execution_llm,
        repair_code_with_llm,
    )
    from core.task_resolution import resolve_task, validate_request_satisfaction
    from core.task_understanding import analyze_question, merge_task_workflow_hints
    from core.ore_rag_assistant import load_index, parse_chunks
    from core.sage_runtime import validate_and_run_sage, validate_generated_code
    from retrieval.knowledge_base import default_index_path_for_mode
    from workflows.task_workflows import build_workflow_prompt_hint, choose_workflow
    from workflows.workflow_executors import generate_code_with_executors

    # Keep these defaults aligned with ui/streamlit_chat_app.py.
    retrieval_k = 6
    final_context_cap = 10
    max_auto_retries = 1
    max_plan_steps = 4
    retrieval_mode = "auto"
    hybrid_alpha = 0.7
    source_priority = "auto"
    symbols_ratio = 0.75
    max_pdf_extras = 2
    retrieval_strategy = "classic"
    graph_path = ""

    result: dict[str, Any] = {
        "routing_result": None,
        "retrieved_symbols": [],
        "generated_code": "",
        "generated_code_for_execution": "",
        "codegen_citations": [],
        "codegen_missing_info": [],
        "auto_imports": [],
        "sage_result": None,
        "final_answer": None,
        "final_answer_error": None,
        "route_mode": "",
        "route_reason": "",
        "escalated_to_plan": False,
        "error": None,
        "settings_used": {
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_output_tokens": max_output_tokens,
            "run_mode": run_mode,
            "workflow_id_hint": workflow_id,
            "use_structured_fast_mode": bool(use_structured_fast_mode),
            "use_structured_auto_fast_mode": bool(use_structured_auto_fast_mode),
        },
        "task_understanding": None,
        "resolved_task": None,
        "request_satisfaction": None,
    }

    try:
        mode = (run_mode or "auto").strip().lower()
        if mode not in {"auto", "fast", "plan"}:
            mode = "auto"
        structured_runtime_active = bool(use_structured_fast_mode) and mode == "fast"
        task_understanding = analyze_question(question) if structured_runtime_active else None
        resolved_task = resolve_task(task_understanding) if task_understanding is not None else None
        structured_selection_override = (
            task_understanding.workflow_selection_override()
            if task_understanding is not None
            else None
        )
        if task_understanding is not None:
            result["task_understanding"] = {
                "debug_lines": task_understanding.debug_lines(),
                "intent": task_understanding.intent.intent_id,
                "workflow_override": task_understanding.intent.workflow_id,
                "family_id": task_understanding.intent.family_id,
            }
        if resolved_task is not None:
            result["resolved_task"] = {
                "workflow_id": resolved_task.workflow_id,
                "intent": resolved_task.intent_id,
                "method_name": resolved_task.method_name,
                "debug_lines": resolved_task.debug_lines(),
            }

        index_path = Path(default_index_path_for_mode("both")).expanduser().resolve()
        payload = load_index(index_path)
        chunks = parse_chunks(payload)
        expected_evidence_type = _expected_evidence_type(question)

        route_mode = "fast"
        route_reason = "User selected Fast mode."
        precheck_results: list = []
        if mode == "plan":
            route_mode = "plan"
            route_reason = "User selected Plan mode."
        elif mode == "auto":
            precheck_k = min(max(4, retrieval_k), 8)
            _, precheck_results, _, _ = _run_retrieval_for_query(
                query=question,
                payload=payload,
                chunks=chunks,
                k=precheck_k,
                mode=retrieval_mode,
                index_path=index_path,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                strategy=retrieval_strategy,
                graph_path=graph_path,
            )
            complexity_score, complexity_details = _question_complexity_score(question)
            first_pass_conf = _retrieval_confidence(precheck_results)
            route_mode, route_reason = _route_adaptive(
                complexity_score=complexity_score,
                confidence=first_pass_conf,
                expected_evidence_type=expected_evidence_type,
                precheck_results=precheck_results,
            )
            result["route_debug"] = {
                "complexity_score": complexity_score,
                "complexity_details": complexity_details,
                "expected_evidence_type": expected_evidence_type,
                "first_pass_confidence": first_pass_conf,
            }

        result["route_mode"] = route_mode
        result["route_reason"] = route_reason

        if (
            mode == "auto"
            and route_mode != "plan"
            and bool(use_structured_auto_fast_mode)
        ):
            structured_runtime_active = True
            if task_understanding is None:
                task_understanding = analyze_question(question)
                resolved_task = resolve_task(task_understanding)
                structured_selection_override = task_understanding.workflow_selection_override()
                result["task_understanding"] = {
                    "debug_lines": task_understanding.debug_lines(),
                    "intent": task_understanding.intent.intent_id,
                    "workflow_override": task_understanding.intent.workflow_id,
                    "family_id": task_understanding.intent.family_id,
                }
                if resolved_task is not None:
                    result["resolved_task"] = {
                        "workflow_id": resolved_task.workflow_id,
                        "intent": resolved_task.intent_id,
                        "method_name": resolved_task.method_name,
                        "debug_lines": resolved_task.debug_lines(),
                    }

        if route_mode == "plan":
            aggregated_results, chunk_step_map, plan_error = _collect_context_plan(
                question=question,
                planning_hint="",
                payload=payload,
                chunks=chunks,
                index_path=index_path,
                k=retrieval_k,
                mode=retrieval_mode,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                retrieval_strategy=retrieval_strategy,
                graph_path=graph_path,
                final_context_cap=final_context_cap,
                provider=provider,
                model=model,
                api_key=None,
                base_url=None,
                max_plan_steps=max_plan_steps,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if plan_error:
                result["error"] = plan_error
        else:
            retry_budget = max_auto_retries if mode == "fast" else (0 if route_mode == "auto_fast" else min(max_auto_retries, 1))
            aggregated_results, chunk_step_map = _collect_context_auto(
                question=question,
                payload=payload,
                chunks=chunks,
                index_path=index_path,
                k=retrieval_k,
                mode=retrieval_mode,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                retrieval_strategy=retrieval_strategy,
                graph_path=graph_path,
                final_context_cap=final_context_cap,
                retry_budget=retry_budget,
                expected_evidence_type=expected_evidence_type,
                initial_results=(
                    precheck_results
                    if precheck_results and not structured_runtime_active
                    else None
                ),
                workflow_selection_override=(
                    structured_selection_override if structured_runtime_active else None
                ),
            )

        if not aggregated_results:
            if not result["error"]:
                result["error"] = "No aggregated context available for final synthesis."
            result["sage_result"] = {
                "status": "skipped",
                "stdout": "",
                "stderr": result["error"],
                "validation_errors": [],
            }
            return result

        current_aggregated_results = list(aggregated_results)
        current_chunk_step_map = dict(chunk_step_map)
        escalated_to_plan = False
        code_response = None
        code_for_execution = ""
        auto_imports: list[str] = []
        execution_result = None
        execution_skipped_reason = ""
        workflow_selection = None
        final_context_items: list = []

        while True:
            final_context_items = _to_context_items(
                results=current_aggregated_results,
                pdf_char_limit=1600,
            )
            if (
                structured_runtime_active
                and structured_selection_override
                and structured_selection_override.has_workflow
            ):
                workflow_selection = structured_selection_override
            else:
                workflow_selection = choose_workflow(
                    question=question,
                    context_items=current_aggregated_results,
                )
            workflow_hint = merge_task_workflow_hints(
                build_workflow_prompt_hint(workflow_selection),
                task_understanding.task_workflow_hint() if task_understanding is not None else "",
                resolved_task.prompt_hint() if resolved_task is not None else "",
            )
            code_request = CodeGenerationRequest(
                question=question,
                contexts=final_context_items,
                task_workflow_hint=workflow_hint,
                resolved_task=resolved_task,
                provider=provider,
                model=model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            code_response = generate_code_with_executors(
                request=code_request,
                workflow_selection=workflow_selection,
                api_key=None,
            )
            code_for_execution = code_response.code
            auto_imports = []
            if code_for_execution.strip():
                code_for_execution, auto_imports = _augment_code_with_retrieval_imports(
                    code=code_for_execution,
                    retrieved_results=current_aggregated_results,
                )
                preflight_errors = validate_generated_code(code_for_execution)
                if _needs_generator_variable_repair(preflight_errors):
                    repaired_response = repair_code_with_llm(
                        request=code_request,
                        original_code=code_for_execution,
                        validation_errors=preflight_errors,
                        api_key=None,
                    )
                    if repaired_response.code.strip():
                        repaired_exec = repaired_response.code
                        repaired_exec, repaired_imports = _augment_code_with_retrieval_imports(
                            code=repaired_exec,
                            retrieved_results=current_aggregated_results,
                        )
                        repaired_errors = validate_generated_code(repaired_exec)
                        if not _needs_generator_variable_repair(repaired_errors):
                            code_response = repaired_response
                            code_for_execution = repaired_exec
                            auto_imports = repaired_imports

            request_satisfaction = validate_request_satisfaction(
                resolved_task,
                generated_code=code_for_execution or (code_response.code if code_response is not None else ""),
                execution_result=None,
            )
            if code_response is not None and request_satisfaction.summary_messages():
                merged_missing = list(code_response.missing_info)
                for message in request_satisfaction.summary_messages():
                    if message not in merged_missing:
                        merged_missing.append(message)
                code_response.missing_info = merged_missing

            if code_for_execution.strip():
                execution_result = validate_and_run_sage(
                    code_for_execution,
                    timeout=sage_timeout,
                )
                executed_code = str(getattr(execution_result, "executed_code", "") or "").strip()
                if executed_code:
                    code_for_execution = executed_code
            else:
                execution_result = None
                execution_skipped_reason = (
                    "No code was generated from the retrieved context, so Sage execution was skipped."
                )

            request_satisfaction = validate_request_satisfaction(
                resolved_task,
                generated_code=code_for_execution or (code_response.code if code_response is not None else ""),
                execution_result=execution_result,
            )
            result["request_satisfaction"] = request_satisfaction.as_dict()

            if _needs_wildcard_import_fallback(execution_result) and code_for_execution.strip():
                modules = _candidate_wildcard_modules(current_aggregated_results, max_modules=2)
                fallback_code, _ = _augment_code_with_module_wildcards(code_for_execution, modules)
                if fallback_code != code_for_execution:
                    retry_result = validate_and_run_sage(
                        fallback_code,
                        timeout=sage_timeout,
                    )
                    if retry_result.status == "success":
                        code_for_execution = str(getattr(retry_result, "executed_code", "") or fallback_code).strip()
                        execution_result = retry_result

            request_satisfaction = validate_request_satisfaction(
                resolved_task,
                generated_code=code_for_execution or (code_response.code if code_response is not None else ""),
                execution_result=execution_result,
            )
            result["request_satisfaction"] = request_satisfaction.as_dict()

            needs_escalation = (
                route_mode in {"auto_fast", "auto"}
                and not escalated_to_plan
                and (
                    bool(code_response.missing_info)
                    or _looks_like_evidence_gap(execution_result)
                    or not request_satisfaction.passed
                )
            )
            if not needs_escalation:
                break

            escalated_results, escalated_map, plan_error = _collect_context_plan(
                question=question,
                planning_hint="",
                payload=payload,
                chunks=chunks,
                index_path=index_path,
                k=retrieval_k,
                mode=retrieval_mode,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                retrieval_strategy=retrieval_strategy,
                graph_path=graph_path,
                final_context_cap=final_context_cap,
                provider=provider,
                model=model,
                api_key=None,
                base_url=None,
                max_plan_steps=max_plan_steps,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )
            if not escalated_results:
                result["error"] = plan_error or "Auto route escalated to plan but no usable context was found."
                break
            current_aggregated_results = escalated_results
            current_chunk_step_map = escalated_map
            escalated_to_plan = True

        result["escalated_to_plan"] = escalated_to_plan
        if workflow_selection is not None:
            result["routing_result"] = {
                "workflow_id": workflow_selection.workflow_id,
                "family_id": workflow_selection.family_id,
                "confidence": workflow_selection.confidence,
            }

        symbols_found: list[str] = []
        for hit in current_aggregated_results:
            sym = getattr(hit, "symbol_id", "") or getattr(hit, "qualname", "") or ""
            if sym:
                symbols_found.append(sym)
        result["retrieved_symbols"] = symbols_found[:10]
        result["retrieval_context_count"] = len(current_aggregated_results)
        result["retrieval_chunk_step_map"] = {
            str(chunk_id): step for chunk_id, step in current_chunk_step_map.items() if chunk_id is not None
        }

        if code_response is not None:
            result["generated_code"] = code_response.code
            result["codegen_citations"] = list(code_response.citations_used)
            result["codegen_missing_info"] = list(code_response.missing_info)
        result["generated_code_for_execution"] = code_for_execution
        result["auto_imports"] = list(auto_imports)

        if execution_result is not None:
            result["sage_result"] = {
                "status": execution_result.status,
                "stdout": execution_result.stdout_full,
                "stderr": execution_result.stderr,
                "validation_errors": list(execution_result.validation_errors),
                "executed_code": str(getattr(execution_result, "executed_code", "") or ""),
            }
        else:
            result["sage_result"] = {
                "status": "skipped",
                "stdout": "",
                "stderr": execution_skipped_reason,
                "validation_errors": [],
                "executed_code": "",
            }

        if code_response is not None and code_response.code.strip():
            try:
                if execution_result is not None and execution_result.status == "success":
                    final_answer_text = _direct_sage_answer(execution_result, execution_skipped_reason)
                    result["final_answer"] = {
                        "answer": final_answer_text,
                        "citations_used": list(code_response.citations_used),
                        "missing_info": [],
                        "raw_response": "(direct-from-sage-output)",
                        "mode": "direct_sage",
                    }
                else:
                    final_response = answer_with_execution_llm(
                        request=ExecutionAwareAnswerRequest(
                            question=question,
                            contexts=_order_contexts_cited_first(
                                final_context_items,
                                list(code_response.citations_used),
                            ),
                            original_code=code_for_execution or code_response.code,
                            execution_result=execution_result,
                            execution_skipped_reason=execution_skipped_reason,
                            code_generation_citations=list(code_response.citations_used),
                            provider=provider,
                            model=model,
                            temperature=temperature,
                            max_output_tokens=max_output_tokens,
                        ),
                        api_key=None,
                    )
                    result["final_answer"] = {
                        "answer": final_response.answer,
                        "citations_used": list(final_response.citations_used),
                        "missing_info": list(final_response.missing_info),
                        "raw_response": final_response.raw_response,
                        "mode": "llm",
                    }
            except Exception as exc:
                result["final_answer_error"] = str(exc)

    except Exception as exc:
        result["error"] = str(exc)
        if result.get("sage_result") is None:
            result["sage_result"] = {
                "status": "error",
                "stdout": "",
                "stderr": str(exc),
                "validation_errors": [],
            }
    return result
