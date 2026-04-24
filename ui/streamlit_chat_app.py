#!/usr/bin/env python3
"""Streamlit agent app: plan -> retrieve per step -> decide -> final synthesis."""

from __future__ import annotations

try:
    from ._repo_path import ensure_repo_root_on_path
except ImportError:
    from _repo_path import ensure_repo_root_on_path

ensure_repo_root_on_path(__file__)

import base64
import json
import os
import re
from pathlib import Path
from typing import List

import streamlit as st

import time

from retrieval.knowledge_base import default_index_path_for_mode
from core.llm_service import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    ContextItem,
    ExecutionAwareAnswerRequest,
    FinalAnswerResponse,
    answer_with_execution_llm,
    Subtask,
    decide_next_action,
    list_ollama_models,
    plan_subtasks,
    repair_code_with_llm,
)
from core.task_resolution import resolve_task, validate_request_satisfaction
from core.task_understanding import analyze_question, merge_task_workflow_hints
from core.ore_rag_assistant import (
    RetrievalResult,
    load_index,
    location_label,
    parse_chunks,
)
from core.sage_runtime import (
    SageExecutionResult,
    prewarm_sage_session_async,
    validate_generated_code,
    validate_and_run_sage,
)
from workflows.task_workflows import (
    WorkflowSelection,
    build_workflow_prompt_hint,
    build_workflow_retrieval_queries,
    choose_workflow,
)
from retrieval.workflow_retrieval import merge_retrieval_result_sets as _shared_merge_retrieval_result_sets
from retrieval.workflow_retrieval import run_workflow_retrieval
from workflows.workflow_executors import generate_code_with_executors


def _result_title(result: RetrievalResult) -> str:
    if result.source_type == "generated":
        name = result.qualname or result.symbol_id or "unknown_symbol"
        return f"{name} [{result.score:.4f}]"
    pages = (
        f"{result.page_start}-{result.page_end}"
        if result.page_start is not None and result.page_end is not None
        else str(result.page_start or "?")
    )
    return f"{result.source} pp.{pages} [{result.score:.4f}]"


def _to_context_items(results: List[RetrievalResult], pdf_char_limit: int) -> List[ContextItem]:
    items: List[ContextItem] = []
    for i, r in enumerate(results, start=1):
        context_id = f"ctx_{i}"
        if r.source_type == "generated":
            title = r.qualname or r.symbol_id or "unknown_symbol"
            text = r.text
        else:
            title = r.section_title or r.source
            text = r.text[:pdf_char_limit]
        items.append(
            ContextItem(
                context_id=context_id,
                source_type=r.source_type,
                title=title,
                location=location_label(r),
                text=text,
                score=r.score,
            )
        )
    return items


def _dedupe_results_by_chunk(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    out: List[RetrievalResult] = []
    for r in results:
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
    return out


def _merge_retrieval_result_sets(result_sets: List[List[RetrievalResult]]) -> List[RetrievalResult]:
    return _shared_merge_retrieval_result_sets(result_sets)


def _citation_lines(citations_used: List[str], context_items: List[ContextItem]) -> str:
    ctx_by_id = {c.context_id: c for c in context_items}
    lines: List[str] = []
    for i, cid in enumerate(citations_used, start=1):
        item = ctx_by_id.get(cid)
        if not item:
            continue
        lines.append(f"- [{i}] {item.title} ({item.location})")
    if not lines:
        return "No citations returned."
    return "\n".join(lines)


def _executor_caption(raw_response: str) -> str:
    try:
        payload = json.loads(raw_response or "")
    except Exception:
        return ""
    if not isinstance(payload, dict):
        return ""
    executor_id = str(payload.get("executor", "")).strip()
    reason = str(payload.get("reason", "")).strip()
    if executor_id:
        if not reason:
            return f"Used deterministic executor: {executor_id}"
        return f"Used deterministic executor: {executor_id} ({reason})"
    # Detect a structured code-plan response by its schema keys.
    plan_keys = {"imports", "setup", "body", "prints"}
    if plan_keys.issubset(payload.keys()):
        return "Used structured code-plan codegen path"
    return ""


def _render_retrieved_context_pool(
    context_items: List[ContextItem],
    cited_ids: set,
    step_map: dict,
) -> None:
    sorted_items = sorted(context_items, key=lambda c: c.score, reverse=True)
    with st.expander(f"All retrieved contexts ({len(sorted_items)}, ranked by score)", expanded=False):
        for item in sorted_items:
            cited = item.context_id in cited_ids
            step_id = step_map.get(item.context_id)
            marker = " ★ cited" if cited else ""
            step_label = f"step {step_id}" if step_id is not None else "?"
            st.markdown(
                f"{'**' if cited else ''}{item.title}{'**' if cited else ''}"
                f" — {item.location}"
                f"  `score {item.score:.3f}` · `{step_label}`{marker}"
            )
            preview = item.text[:400] + ("..." if len(item.text) > 400 else "")
            st.caption(preview)


def _order_contexts_cited_first(
    context_items: List[ContextItem],
    cited_context_ids: List[str],
) -> List[ContextItem]:
    if not context_items or not cited_context_ids:
        return list(context_items)

    cited_ids = set(cited_context_ids)
    cited_items: List[ContextItem] = []
    uncited_items: List[ContextItem] = []
    for item in context_items:
        if item.context_id in cited_ids:
            cited_items.append(item)
        else:
            uncited_items.append(item)
    return cited_items + uncited_items


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


def _preferred_symbol_for_auto_import(result: RetrievalResult) -> str:
    source = (result.qualname or result.symbol_id or "").strip()
    if not source:
        return ""
    parts = [part for part in source.split(".") if part]
    if not parts:
        return ""

    kind = (result.kind or "").strip().lower()
    if kind == "method" and len(parts) >= 2:
        candidate = parts[-2]
    else:
        candidate = parts[-1]

    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", candidate or ""):
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


def _insert_import_lines(code: str, import_lines: List[str]) -> str:
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
    retrieved_results: List[RetrievalResult],
    max_added_imports: int = 6,
) -> tuple[str, List[str]]:
    raw = code.strip()
    if not raw:
        return code, []

    added: List[str] = []
    seen = set()
    for result in retrieved_results:
        if result.source_type != "generated":
            continue
        module = _normalize_module_for_auto_import(result.module)
        symbol = _preferred_symbol_for_auto_import(result)
        if not module or not symbol:
            continue
        if module == "ore_algebra":
            # Runtime already includes `from ore_algebra import *`.
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
    retrieved_results: List[RetrievalResult],
    max_modules: int = 2,
) -> List[str]:
    out: List[str] = []
    seen = set()
    for result in retrieved_results:
        if result.source_type != "generated":
            continue
        module = _normalize_module_for_auto_import(result.module)
        if not module or module == "ore_algebra":
            continue
        if module in seen:
            continue
        seen.add(module)
        out.append(module)
        if len(out) >= max_modules:
            break
    return out


def _augment_code_with_module_wildcards(code: str, modules: List[str]) -> tuple[str, List[str]]:
    raw = code.strip()
    if not raw or not modules:
        return code, []
    added: List[str] = []
    for module in modules:
        line = f"from {module} import *"
        if re.search(rf"^\s*from\s+{re.escape(module)}\s+import\s+\*", raw, flags=re.MULTILINE):
            continue
        added.append(line)
    if not added:
        return code, []
    return _insert_import_lines(raw, added), added


def _needs_wildcard_import_fallback(execution_result: SageExecutionResult | None) -> bool:
    if execution_result is None:
        return False
    if execution_result.status != "error":
        return False
    stderr = (execution_result.stderr or "").lower()
    markers = (
        "nameerror",
        "not defined",
        "importerror",
        "no module named",
        "cannot import name",
        "attributeerror",
    )
    return any(marker in stderr for marker in markers)


def _needs_generator_variable_repair(validation_errors: List[str]) -> bool:
    for item in validation_errors:
        item_lower = item.lower()
        if "generator/base-variable mismatch" in item_lower:
            return True
        if "generator binding/access issue" in item_lower:
            return True
        if "unbound generator issue" in item_lower:
            return True
    return False


def _download_link(data: str, filename: str, label: str = "Download") -> str:
    b64 = base64.b64encode(data.encode()).decode()
    return (
        f'<a href="data:text/plain;base64,{b64}" download="{filename}" '
        f'style="text-decoration:none;font-size:0.85em;">{label}</a>'
    )


def _wrap_latex_envs(text: str) -> str:
    # Normalize literal \n sequences to real newlines
    text = text.replace('\\n', '\n')
    # Wrap block LaTeX environments with $$
    text = re.sub(
        r'(\\begin\{[^}]+\}.*?\\end\{[^}]+\})',
        r'$$\1$$',
        text,
        flags=re.DOTALL,
    )
    return text


def _wrap_inline_latex(text: str) -> str:
    # Wrap inline LaTeX commands like \texttt{...}, \text{...} with $...$
    return re.sub(r'(\\[a-zA-Z]+\{[^}]*\})', r'$\1$', text)


def _render_answer(text: str, context_items: List[ContextItem] | None = None) -> None:
    # Replace \cite{ctx_N} with [N] before any LaTeX processing
    if context_items:
        ctx_index = {item.context_id: i for i, item in enumerate(context_items, start=1)}
        def _replace_cite(m: re.Match) -> str:
            cid = m.group(1)
            n = ctx_index.get(cid)
            return f"[{n}]" if n is not None else m.group(0)
        text = re.sub(r'\\cite\{(ctx_\d+)\}', _replace_cite, text)
    text = _wrap_latex_envs(text)
    parts = re.split(r'(\$\$.*?\$\$)', text, flags=re.DOTALL)
    for part in parts:
        if part.startswith('$$') and part.endswith('$$'):
            st.latex(part[2:-2].strip())
        elif part.strip():
            part = re.sub(r'\\+\s*$', '', part)
            st.markdown(_wrap_inline_latex(part))


def _compact_nonempty_lines(text: str) -> List[str]:
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.strip():
            lines.append(line)
    return lines


def _render_direct_sage_final_answer(
    execution_result: SageExecutionResult | None,
    execution_skipped_reason: str,
) -> str:
    if execution_result is None:
        msg = execution_skipped_reason.strip() or "Execution was skipped, so there is no runtime result to display."
        st.markdown(msg)
        return msg

    if execution_result.status != "success":
        msg = f"Sage execution ended with status `{execution_result.status}`."
        st.markdown(msg)
        detail = _execution_detail_text(execution_result, execution_skipped_reason)
        if detail.strip():
            st.code(detail, language="text")
        return msg

    stdout = (execution_result.stdout_summary or execution_result.stdout_full or "").strip()
    if not stdout:
        msg = "Sage execution succeeded, but produced no visible stdout output."
        st.markdown(msg)
        return msg

    lines = _compact_nonempty_lines(stdout)
    if not lines:
        msg = "Sage execution succeeded, but output was empty after trimming whitespace."
        st.markdown(msg)
        return msg

    st.markdown("Result from Sage execution:")
    preview_limit = 10
    preview = lines[:preview_limit]
    st.code("\n".join(preview), language="text")
    if len(lines) > preview_limit:
        st.caption(f"Showing {preview_limit} of {len(lines)} non-empty lines. Full output is available above.")

    if len(lines) == 1:
        return f"Sage execution succeeded. Result: {lines[0]}"
    return f"Sage execution succeeded with {len(lines)} non-empty output lines."


def _execution_status_label(execution_result: SageExecutionResult | None) -> str:
    if execution_result is None:
        return "skipped"
    return execution_result.status



def _execution_detail_text(
    execution_result: SageExecutionResult | None,
    skipped_reason: str,
) -> str:
    if execution_result is None:
        return skipped_reason.strip() or "Execution was skipped."

    detail_parts: List[str] = []
    if execution_result.validation_errors:
        detail_parts.append("Validation errors:")
        detail_parts.extend(f"- {item}" for item in execution_result.validation_errors)
    if execution_result.stderr.strip():
        if detail_parts:
            detail_parts.append("")
        detail_parts.append("stderr:")
        detail_parts.append(execution_result.stderr.rstrip())
    if not detail_parts:
        return "(No stderr or validation errors.)"
    return "\n".join(detail_parts)


def _render_retrieval_results(results: List[RetrievalResult], pdf_char_limit: int, key_prefix: str) -> None:
    st.subheader("Retrieved Context")
    for idx, r in enumerate(results):
        with st.expander(_result_title(r), expanded=False):
            st.write(f"source_type: `{r.source_type}`")
            st.write(f"source: `{r.source}`")
            if r.source_type == "generated":
                if r.signature:
                    st.write(f"signature: `{r.signature}`")
                if r.module:
                    st.write(f"module: `{r.module}`")
                st.write(f"location: `{location_label(r)}`")
                st.text_area(
                    "content",
                    value=r.text,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-gen",
                )
            else:
                section = r.section_title or "unknown"
                st.write(f"pages: `{r.page_start}-{r.page_end}`")
                st.write(f"section: `{section}`")
                preview = r.text[:pdf_char_limit] if len(r.text) > pdf_char_limit else r.text
                st.text_area(
                    "content",
                    value=preview,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-pdf",
                )


def _build_workflow_retrieval_debug_lines(
    *,
    workflow_selection: WorkflowSelection,
    retrieval_queries: tuple[str, ...],
    mode_used: str = "",
    results: List[RetrievalResult] | None = None,
) -> List[str]:
    lines: List[str] = []
    if getattr(workflow_selection, "has_family", False):
        label = f"{workflow_selection.family_id}. {workflow_selection.family_name}"
        if getattr(workflow_selection, "has_workflow", False):
            label = f"{label} -> {workflow_selection.workflow_title}"
        lines.append(f"workflow: {label}")
        lines.append(f"confidence: {workflow_selection.confidence:.2f}")
    if mode_used:
        lines.append(f"search_mode: {mode_used}")
    if getattr(workflow_selection, "matched_on", ()):
        lines.append(f"matched_on: {', '.join(workflow_selection.matched_on[:4])}")
    if len(retrieval_queries) > 1:
        lines.append("expanded_queries:")
        lines.extend(f"  - {query}" for query in retrieval_queries)
    elif retrieval_queries:
        lines.append(f"query: {retrieval_queries[0]}")
    if results:
        generated_count = sum(1 for item in results if item.source_type == "generated")
        pdf_count = sum(1 for item in results if item.source_type == "pdf")
        lines.append(f"merged_results: {len(results)}")
        lines.append(f"source_breakdown: generated={generated_count}, pdf={pdf_count}")
        top_hits = [_result_title(item) for item in list(results)[:3]]
        if top_hits:
            lines.append("top_hits:")
            lines.extend(f"  - {hit}" for hit in top_hits)
    return lines


def _render_workflow_retrieval_debug(
    *,
    workflow_selection: WorkflowSelection,
    retrieval_queries: tuple[str, ...],
    mode_used: str = "",
    results: List[RetrievalResult] | None = None,
) -> None:
    lines = _build_workflow_retrieval_debug_lines(
        workflow_selection=workflow_selection,
        retrieval_queries=retrieval_queries,
        mode_used=mode_used,
        results=results,
    )
    if not lines:
        return
    with st.expander("Workflow retrieval debug", expanded=False):
        for line in lines:
            st.write(line)


def _load_json_summary(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _workflow_benchmark_snapshot_lines(repo_root: Path) -> List[str]:
    generated_dir = repo_root / "generated"
    lines: List[str] = []

    workflow_payload = _load_json_summary(generated_dir / "workflow_eval_summary.json")
    if isinstance(workflow_payload, dict) and isinstance(workflow_payload.get("summary"), dict):
        summary = workflow_payload["summary"]
        lines.append(
            f"execution: {summary.get('passed', 0)}/{summary.get('total', 0)} passed"
        )

    routing_payload = _load_json_summary(generated_dir / "workflow_routing_summary.json")
    if isinstance(routing_payload, dict) and isinstance(routing_payload.get("summary"), dict):
        summary = routing_payload["summary"]
        lines.append(
            f"routing: {summary.get('passed', 0)}/{summary.get('total', 0)} passed"
        )

    retrieval_payload = _load_json_summary(generated_dir / "workflow_retrieval_summary.json")
    if isinstance(retrieval_payload, dict) and isinstance(retrieval_payload.get("summary"), dict):
        summary = retrieval_payload["summary"]
        lines.append(
            "retrieval lexical: "
            f"{summary.get('passed', 0)}/{summary.get('total', 0)} passed, "
            f"MRR={float(summary.get('mrr_workflow', 0.0)):.3f}"
        )

    hybrid_payload = _load_json_summary(generated_dir / "workflow_retrieval_summary_hybrid.json")
    if isinstance(hybrid_payload, dict) and isinstance(hybrid_payload.get("summary"), dict):
        summary = hybrid_payload["summary"]
        lines.append(
            "retrieval hybrid: "
            f"{summary.get('passed', 0)}/{summary.get('total', 0)} passed, "
            f"MRR={float(summary.get('mrr_workflow', 0.0)):.3f}"
        )
    else:
        lines.append("retrieval hybrid: not generated")

    return lines


def _render_workflow_benchmark_snapshot(repo_root: Path) -> None:
    lines = _workflow_benchmark_snapshot_lines(repo_root)
    if not lines:
        return
    with st.sidebar.expander("Workflow Metrics", expanded=False):
        for line in lines:
            st.write(line)


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
    workflow_selection_override: WorkflowSelection | None = None,
    strategy: str = "classic",
    graph_path: str = "",
) -> tuple[str, List[RetrievalResult], WorkflowSelection, tuple[str, ...]]:
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


def _render_step_header(step: Subtask, step_query: str) -> None:
    st.markdown(f"### Step {step.step_id}: {step.title}")
    st.write(f"instruction: {step.instruction}")
    st.write(f"query: `{step_query}`")


@st.cache_data(ttl=5)
def _cached_ollama_models(base_url: str) -> tuple[list[str], str]:
    return list_ollama_models(base_url=base_url)


def _question_complexity_score(question: str) -> tuple[int, dict[str, int]]:
    q = question.strip()
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
    score += min(goal_hits, 3)
    score += min(constraint_hits, 2)
    if q.count("?") > 1:
        score += 1

    details = {
        "word_count": len(words),
        "api_refs": len(dotted_apis) + known_api_hits,
        "goal_markers": goal_hits,
        "constraint_markers": constraint_hits,
    }
    return score, details


def _expected_evidence_type(question: str) -> str:
    lower = question.lower()
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


def _retrieval_confidence(results: List[RetrievalResult]) -> float:
    if not results:
        return 0.0
    top = results[: min(4, len(results))]
    scores = [_normalized_retrieval_score(r.score) for r in top]
    confidence = sum(scores) / max(len(scores), 1)
    if sum(1 for r in top if r.source_type == "generated") >= 2:
        confidence = min(1.0, confidence + 0.03)
    return confidence


def _route_adaptive(
    *,
    complexity_score: int,
    confidence: float,
    expected_evidence_type: str,
    precheck_results: List[RetrievalResult],
) -> tuple[str, str]:
    top = precheck_results[: min(5, len(precheck_results))]
    has_pdf = any(r.source_type == "pdf" for r in top)
    if expected_evidence_type == "symbol+pdf" and not has_pdf:
        return "plan", "Question appears PDF/doc-dependent but first-pass retrieval lacks PDF evidence."
    if complexity_score <= 3 and confidence >= 0.60:
        return "auto_fast", "Low complexity and high first-pass retrieval confidence."
    if complexity_score <= 6 and confidence >= 0.30:
        return "auto", "Medium complexity or moderate confidence; allow one targeted retry."
    return "plan", "High complexity or low confidence; use plan flow."


def _top_symbol_hints(results: List[RetrievalResult], limit: int = 3) -> List[str]:
    out: List[str] = []
    for r in results:
        if r.source_type != "generated":
            continue
        hint = (r.qualname or r.symbol_id or "").strip()
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
    results: List[RetrievalResult],
    expected_evidence_type: str,
    missing_hints: List[str] | None = None,
) -> str:
    pieces: List[str] = [question]
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


def _looks_like_evidence_gap(execution_result: SageExecutionResult | None) -> bool:
    if execution_result is None:
        return False
    if execution_result.status == "success":
        return False
    text = "\n".join(execution_result.validation_errors or []) + "\n" + (execution_result.stderr or "")
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


def _build_escalation_hint(
    code_response: CodeGenerationResponse,
    execution_result: SageExecutionResult | None,
) -> str:
    hints: List[str] = []
    if code_response.missing_info:
        hints.extend(code_response.missing_info[:3])
    if execution_result is not None and execution_result.stderr:
        first_line = next((line.strip() for line in execution_result.stderr.splitlines() if line.strip()), "")
        if first_line:
            hints.append(first_line[:200])
    return "; ".join(hints)


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
    pdf_char_limit: int,
    final_context_cap: int,
    provider: str,
    model: str,
    api_key: str | None,
    llm_base_url: str | None,
    max_plan_steps: int,
    temperature: float,
    llm_max_output_tokens: int | None,
    show_workflow_retrieval_debug: bool,
    section_title: str = "Solution Plan",
) -> tuple[List[RetrievalResult], dict[str, int]]:
    planning_question = question.strip()
    if planning_hint.strip():
        planning_question = (
            f"{planning_question}\n\nPrior attempt showed missing evidence:\n{planning_hint.strip()}"
        )

    try:
        with st.spinner("Planning subtasks..."):
            plan = plan_subtasks(
                question=planning_question,
                provider=provider,
                model=model,
                api_key=api_key or None,
                base_url=llm_base_url,
                max_steps=max_plan_steps,
                temperature=temperature,
                max_output_tokens=llm_max_output_tokens,
            )
    except Exception as exc:
        st.error(f"Planning failed: {exc}")
        return [], {}

    st.subheader(section_title)
    if not plan.subtasks:
        st.warning("Planner returned no subtasks.")
        return [], {}

    aggregated_results: List[RetrievalResult] = []
    chunk_step_map: dict[str, int] = {}

    for subtask in plan.subtasks:
        step_query = subtask.retrieval_query or subtask.instruction or subtask.title
        with st.status(f"Step {subtask.step_id} — {subtask.title}", expanded=False) as step_status:
            if subtask.family_id:
                st.write(f"family: `{subtask.family_id}`")
            st.write(f"instruction: {subtask.instruction}")
            st.write(f"query: {step_query}")

            mode_used, results, workflow_selection, retrieval_queries = _run_retrieval_for_query(
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
            st.write(f"Search mode: `{mode_used}`, results: `{len(results)}`")
            if show_workflow_retrieval_debug:
                _render_workflow_retrieval_debug(
                    workflow_selection=workflow_selection,
                    retrieval_queries=retrieval_queries,
                    mode_used=mode_used,
                    results=results,
                )
            _render_retrieval_results(
                results=results,
                pdf_char_limit=pdf_char_limit,
                key_prefix=f"plan-step-{subtask.step_id}-base",
            )
            context_items = _to_context_items(results=results, pdf_char_limit=pdf_char_limit)

            decision = decide_next_action(
                question=question,
                current_step=subtask,
                context_items=context_items,
                provider=provider,
                model=model,
                api_key=api_key or None,
                base_url=llm_base_url,
                temperature=temperature,
                max_output_tokens=llm_max_output_tokens,
            )

            st.write(f"Next step: `{decision.action}`")
            if decision.reason:
                st.write(f"reason: {decision.reason}")
            st.write(f"confidence: {decision.confidence:.2f}")

            step_effective_results = results
            if decision.action == "refine_query" and decision.next_query.strip():
                st.write(f"refine query: `{decision.next_query}`")
                mode_used2, refined_results, refined_selection, refined_queries = _run_retrieval_for_query(
                    query=decision.next_query.strip(),
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
                st.write(f"refined retrieval mode: `{mode_used2}`, results: `{len(refined_results)}`")
                if show_workflow_retrieval_debug:
                    _render_workflow_retrieval_debug(
                        workflow_selection=refined_selection,
                        retrieval_queries=refined_queries,
                        mode_used=mode_used2,
                        results=refined_results,
                    )
                _render_retrieval_results(
                    results=refined_results,
                    pdf_char_limit=pdf_char_limit,
                    key_prefix=f"plan-step-{subtask.step_id}-refined",
                )
                step_effective_results = refined_results

            aggregated_results.extend(step_effective_results)
            for r in step_effective_results:
                chunk_step_map.setdefault(r.chunk_id, subtask.step_id)

            if decision.action == "stop":
                st.info("Workflow stopped by agent decision.")
                break

            step_status.update(state="complete", expanded=False)

    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_cap]
    return aggregated_results, chunk_step_map


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
    pdf_char_limit: int,
    final_context_cap: int,
    retry_budget: int,
    expected_evidence_type: str,
    initial_results: List[RetrievalResult] | None = None,
    show_workflow_retrieval_debug: bool = False,
    workflow_selection_override: WorkflowSelection | None = None,
) -> tuple[List[RetrievalResult], dict[str, int]]:
    st.subheader("Fast Retrieval")
    aggregated_results: List[RetrievalResult] = []
    chunk_step_map: dict[str, int] = {}
    query = question
    cached_first = list(initial_results or [])

    for attempt in range(max(retry_budget, 0) + 1):
        with st.status(f"Fast step {attempt + 1}", expanded=False) as step_status:
            if attempt == 0 and cached_first:
                mode_used = "precheck-cache"
                results = cached_first
                if workflow_selection_override is not None and workflow_selection_override.has_workflow:
                    workflow_selection = workflow_selection_override
                else:
                    workflow_selection = choose_workflow(question=query)
                retrieval_queries = build_workflow_retrieval_queries(
                    question=query,
                    selection=workflow_selection,
                )
            else:
                mode_used, results, workflow_selection, retrieval_queries = _run_retrieval_for_query(
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
            st.write(f"query: {query}")
            st.write(f"Search mode: `{mode_used}`, results: `{len(results)}`")
            if show_workflow_retrieval_debug:
                _render_workflow_retrieval_debug(
                    workflow_selection=workflow_selection,
                    retrieval_queries=retrieval_queries,
                    mode_used=mode_used,
                    results=results,
                )
            _render_retrieval_results(
                results=results,
                pdf_char_limit=pdf_char_limit,
                key_prefix=f"auto-step-{attempt + 1}",
            )

            aggregated_results.extend(results)
            for r in results:
                chunk_step_map.setdefault(r.chunk_id, attempt + 1)

            if attempt >= retry_budget:
                step_status.update(state="complete", expanded=False)
                continue

            retry_query = _build_auto_retry_query(
                question=question,
                results=results,
                expected_evidence_type=expected_evidence_type,
            ).strip()
            if not retry_query or retry_query == query.strip():
                step_status.update(state="complete", expanded=False)
                break
            query = retry_query
            step_status.update(state="complete", expanded=False)

    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_cap]
    return aggregated_results, chunk_step_map


def main() -> None:
    st.set_page_config(page_title="ore_algebra Assistant", layout="wide", initial_sidebar_state="collapsed",)
    st.title("ore_algebra Assistant")

    with st.sidebar:
        st.header("Knowledge Base")
        index_path = st.text_input("Index path", default_index_path_for_mode("both"))
        workflow_mode = st.selectbox(
            "Workflow mode",
            ["Auto (Recommended)", "Fast", "Plan"],
            index=0,
            help="Auto routes between Fast and Plan using a quick pre-check.",
        )
        max_plan_steps = st.slider("Max planning steps", min_value=1, max_value=10, value=4, help="Maximum number of retrieval steps the planner will run.")
        col1, col2 = st.columns(2)
        with col1:
            k = st.slider("Top-k per step", min_value=1, max_value=20, value=6, help="Retrieved results per planning step.")
        with col2:
            final_context_limit = st.slider("Final Top-k", min_value=2, max_value=30, value=10, help="Context chunks passed to the final answer LLM.")
        with st.expander("Cost guardrails"):
            max_auto_retries = st.slider(
                "Max fast retries",
                min_value=0,
                max_value=2,
                value=1,
                help="Upper bound for Fast retries. Auto mode uses at most one retry when it routes to Fast.",
            )
            hard_max_contexts = st.slider(
                "Hard max contexts",
                min_value=2,
                max_value=30,
                value=10,
                help="Absolute cap on contexts passed into code/final LLM calls.",
            )
            llm_max_output_tokens = st.slider(
                "Max output tokens / LLM call",
                min_value=0,
                max_value=4096,
                value=1200,
                step=64,
                help="Set 0 to let the provider choose token limits.",
            )

        with st.expander("Advanced search settings"):
            mode = st.selectbox("Search mode", ["auto", "hybrid", "dense", "lexical"], index=0)
            retrieval_strategy = "classic"
            graph_path = ""
            st.caption("Retrieval strategy: classic")
            hybrid_alpha = st.slider("Hybrid weight", min_value=0.0, max_value=1.0, value=0.7, help="1.0 = dense only, 0.0 = lexical only.")
            source_priority = st.selectbox("Source preference", ["auto", "symbols-first", "flat"], index=0)
            symbols_ratio = st.slider("Symbol vs text ratio", min_value=0.0, max_value=1.0, value=0.75)
            max_pdf_extras = st.slider("Max PDF results", min_value=0, max_value=10, value=2)
            pdf_char_limit = st.slider("PDF chunk length", min_value=400, max_value=3000, value=1600)
            show_workflow_retrieval_debug = st.checkbox(
                "Show workflow retrieval debug",
                value=False,
                help="Show the selected workflow and the workflow-expanded retrieval queries for each search step.",
            )
            show_workflow_benchmark_metrics = st.checkbox(
                "Show workflow benchmark snapshot",
                value=False,
                help="Show the latest workflow benchmark summaries from generated report files.",
            )
        with st.expander("Experimental task understanding"):
            use_structured_fast_paths = st.checkbox(
                "Use structured task understanding for Fast paths",
                value=False,
                help=(
                    "Experimental: adds a ParsedRequest + intent/workflow guidance layer "
                    "before retrieval/code generation when Fast is used directly or selected by Auto routing."
                ),
            )
            show_structured_fast_debug = st.checkbox(
                "Show structured task debug",
                value=False,
                help=(
                    "Show the parsed request, intent, and workflow override when the experimental "
                    "structured Fast path is active."
                ),
            )
            st.caption("Plan mode is unchanged in this first rollout.")

        st.header("Language Model")
        provider = st.selectbox("Provider", ["openai", "gemini", "ollama"], index=0)
        temperature = st.slider(
            "Temperature",
            min_value=0.0, max_value=1.0, value=0.1, step=0.05,
            help="Lower = more deterministic, higher = more creative.",
        )
        api_key = ""
        llm_base_url: str | None = None

        if provider == "openai":
            openai_models = [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4-turbo",
                "gpt-4.1",
                "gpt-4.1-mini",
                "gpt-5.4-mini",
                "o1-mini",
                "o3-mini",
                "(custom)",
            ]
            picked_model = st.selectbox("Model", openai_models, index=0)
            if picked_model == "(custom)":
                model = st.text_input(
                    "Custom OpenAI model id",
                    value="",
                    placeholder="e.g. gpt-4o-2024-11-20",
                ).strip() or "gpt-4o-mini"
            else:
                model = picked_model
            api_key = st.text_input("API key", type="password")
            has_default_key = bool(os.getenv("OPENAI_API_KEY"))
            st.caption("Using key from environment." if (not api_key and has_default_key) else "" if api_key else "No API key found.")
        elif provider == "gemini":
            model = st.text_input("Model", "gemini-2.5-flash")
            api_key = st.text_input("API key", type="password")
            has_default_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
            st.caption("Using key from environment." if (not api_key and has_default_key) else "" if api_key else "No API key found.")
        else:
            llm_base_url = st.text_input(
                "Ollama base URL",
                value=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ).strip()
            detected_models, detect_error = _cached_ollama_models(llm_base_url or "http://localhost:11434")
            if detected_models:
                default_model = "qwen3-coder:30b"
                default_index = detected_models.index(default_model) if default_model in detected_models else 0
                selected_model = st.selectbox("Model", detected_models, index=default_index)
                custom_model = st.text_input("Override model (optional)", "")
                model = custom_model.strip() or selected_model
                st.caption(f"{len(detected_models)} model(s) detected locally.")
            else:
                model = st.text_input("Model", os.getenv("OLLAMA_MODEL", "llama3.1"))
                st.caption(f"No models detected. {detect_error}" if detect_error else "No models detected. Run `ollama pull <model>`.")

        st.header("Sage Execution")
        sage_bin = st.text_input("Sage path", os.getenv("SAGE_BIN", "sage"))
        execution_timeout = st.slider("Timeout (s)", min_value=1, max_value=180, value=60)
        use_warm_sage_session = st.checkbox(
            "Warm Sage session",
            value=True,
            help="Start Sage in background when a question arrives and reuse it for faster execution.",
        )
        warm_idle_minutes = st.slider(
            "Warm idle timeout (min)",
            min_value=1,
            max_value=60,
            value=15,
            help="Close warm Sage session after this idle time.",
        )

    if show_workflow_benchmark_metrics:
        _render_workflow_benchmark_snapshot(Path(__file__).resolve().parents[1])

    question = st.chat_input("Ask a question about ore_algebra...")
    if question:
        st.session_state["last_question"] = question

    if not question:
        last_question = st.session_state.get("last_question", "")
        if last_question:
            st.subheader("Last Question")
            st.markdown(last_question)
        st.info("You can ask about symbolic computations in ore_algebra.")
        return
    
    start_time = time.time()
    with st.chat_message("user"):
        st.markdown(question)

    if use_warm_sage_session:
        prewarm_sage_session_async(
            sage_bin=sage_bin,
            warm_ttl_seconds=warm_idle_minutes * 60,
            startup_timeout=min(execution_timeout, 20),
        )

    idx = Path(index_path).expanduser().resolve()
    if not idx.exists():
        st.error(f"Index file not found: {idx}")
        return

    try:
        with st.spinner("Loading index..."):
            payload = load_index(idx)
            chunks = parse_chunks(payload)
    except Exception as exc:
        st.error(f"Index load failed: {exc}")
        return

    final_context_cap = min(final_context_limit, hard_max_contexts)
    expected_evidence_type = _expected_evidence_type(question)
    route_mode = "auto"
    route_reason = "User selected Fast mode."
    precheck_results: List[RetrievalResult] = []
    structured_runtime_active = bool(use_structured_fast_paths) and workflow_mode == "Fast"
    task_understanding = analyze_question(question) if structured_runtime_active else None
    resolved_task = resolve_task(task_understanding) if task_understanding is not None else None
    structured_selection_override = (
        task_understanding.workflow_selection_override()
        if task_understanding is not None
        else None
    )

    if workflow_mode == "Plan":
        route_mode = "plan"
        route_reason = "User selected Plan mode."
    elif workflow_mode == "Auto (Recommended)":
        with st.status("Adaptive pre-check", expanded=False) as pre_status:
            complexity_score, complexity_details = _question_complexity_score(question)
            precheck_k = min(max(4, k), 8)
            precheck_mode, precheck_results, precheck_selection, precheck_queries = _run_retrieval_for_query(
                query=question,
                payload=payload,
                chunks=chunks,
                k=precheck_k,
                mode=mode,
                index_path=idx,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                strategy=retrieval_strategy,
                graph_path=graph_path,
            )
            if show_workflow_retrieval_debug:
                _render_workflow_retrieval_debug(
                    workflow_selection=precheck_selection,
                    retrieval_queries=precheck_queries,
                    mode_used=precheck_mode,
                    results=precheck_results,
                )
            first_pass_conf = _retrieval_confidence(precheck_results)
            route_mode, route_reason = _route_adaptive(
                complexity_score=complexity_score,
                confidence=first_pass_conf,
                expected_evidence_type=expected_evidence_type,
                precheck_results=precheck_results,
            )
            st.write(
                "complexity="
                f"{complexity_score} (words={complexity_details['word_count']}, "
                f"api_refs={complexity_details['api_refs']}, "
                f"goals={complexity_details['goal_markers']}, "
                f"constraints={complexity_details['constraint_markers']})"
            )
            st.write(
                f"expected_evidence: `{expected_evidence_type}`  |  "
                f"first_pass_confidence: `{first_pass_conf:.2f}`  |  "
                f"precheck_mode: `{precheck_mode}`"
            )
            st.write(f"route: `{route_mode}`")
            pre_status.update(state="complete", expanded=False)

    if (
        workflow_mode == "Auto (Recommended)"
        and route_mode != "plan"
        and bool(use_structured_fast_paths)
    ):
        structured_runtime_active = True
        if task_understanding is None:
            task_understanding = analyze_question(question)
            resolved_task = resolve_task(task_understanding)
            structured_selection_override = task_understanding.workflow_selection_override()

    st.caption(f"Routing decision: `{route_mode}` — {route_reason}")
    if structured_runtime_active:
        st.caption("Structured task understanding is active for this run.")
        if show_structured_fast_debug and task_understanding is not None:
            st.markdown("\n".join(f"- {line}" for line in task_understanding.debug_lines()))
        if show_structured_fast_debug and resolved_task is not None:
            st.markdown("\n".join(f"- {line}" for line in resolved_task.debug_lines()))

    if route_mode == "plan":
        aggregated_results, chunk_step_map = _collect_context_plan(
            question=question,
            planning_hint="",
            payload=payload,
            chunks=chunks,
            index_path=idx,
            k=k,
            mode=mode,
            hybrid_alpha=hybrid_alpha,
            source_priority=source_priority,
            symbols_ratio=symbols_ratio,
            max_pdf_extras=max_pdf_extras,
            retrieval_strategy=retrieval_strategy,
            graph_path=graph_path,
            pdf_char_limit=pdf_char_limit,
            final_context_cap=final_context_cap,
            provider=provider,
            model=model,
            api_key=api_key or None,
            llm_base_url=llm_base_url,
            max_plan_steps=max_plan_steps,
            temperature=temperature,
            llm_max_output_tokens=llm_max_output_tokens,
            show_workflow_retrieval_debug=show_workflow_retrieval_debug,
            section_title="Solution Plan",
        )
    else:
        if workflow_mode == "Auto (Recommended)":
            retry_budget = 0 if route_mode == "auto_fast" else min(max_auto_retries, 1)
        else:
            retry_budget = max_auto_retries
        aggregated_results, chunk_step_map = _collect_context_auto(
            question=question,
            payload=payload,
            chunks=chunks,
            index_path=idx,
            k=k,
            mode=mode,
            hybrid_alpha=hybrid_alpha,
            source_priority=source_priority,
            symbols_ratio=symbols_ratio,
            max_pdf_extras=max_pdf_extras,
            retrieval_strategy=retrieval_strategy,
            graph_path=graph_path,
            pdf_char_limit=pdf_char_limit,
            final_context_cap=final_context_cap,
            retry_budget=retry_budget,
            expected_evidence_type=expected_evidence_type,
            initial_results=(
                precheck_results
                if precheck_results and not structured_runtime_active
                else None
            ),
            show_workflow_retrieval_debug=show_workflow_retrieval_debug,
            workflow_selection_override=(
                structured_selection_override if structured_runtime_active else None
            ),
        )

    if not aggregated_results:
        st.warning("No aggregated context available for final synthesis.")
        return

    current_aggregated_results = list(aggregated_results)
    current_chunk_step_map = dict(chunk_step_map)
    final_context_items: List[ContextItem] = []
    ctx_step_map: dict[str, int] = {}

    code_response: CodeGenerationResponse
    code_for_execution = ""
    execution_result: SageExecutionResult | None = None
    execution_skipped_reason = ""
    final_response: FinalAnswerResponse

    try:
        st.divider()
        st.subheader("Computation")
        escalated_to_plan = False
        while True:
            execution_skipped_reason = ""
            final_context_items = _to_context_items(
                results=current_aggregated_results,
                pdf_char_limit=pdf_char_limit,
            )
            ctx_step_map = {
                f"ctx_{i}": current_chunk_step_map.get(r.chunk_id, 0)
                for i, r in enumerate(current_aggregated_results, start=1)
            }
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
            if workflow_selection.has_family:
                label = f"{workflow_selection.family_id}. {workflow_selection.family_name}"
                if workflow_selection.has_workflow:
                    label = f"{label} -> {workflow_selection.workflow_title}"
                st.caption(
                    f"Task workflow: {label} "
                    f"(confidence {workflow_selection.confidence:.2f})"
                )
            code_request = CodeGenerationRequest(
                question=question,
                contexts=final_context_items,
                task_workflow_hint=workflow_hint,
                resolved_task=resolved_task,
                provider=provider,
                model=model,
                temperature=temperature,
                base_url=llm_base_url,
                max_output_tokens=llm_max_output_tokens,
            )

            code_label = "Sage Code" if not escalated_to_plan else "Sage Code (After Plan Escalation)"
            with st.status(code_label, expanded=False) as code_status:
                code_response = generate_code_with_executors(
                    request=code_request,
                    workflow_selection=workflow_selection,
                    api_key=api_key or None,
                )
                code_for_execution = code_response.code
                auto_imports: List[str] = []
                if code_for_execution.strip():
                    code_for_execution, auto_imports = _augment_code_with_retrieval_imports(
                        code=code_for_execution,
                        retrieved_results=current_aggregated_results,
                    )
                    preflight_errors = validate_generated_code(code_for_execution, sage_bin=sage_bin)
                    if _needs_generator_variable_repair(preflight_errors):
                        st.caption("Pre-execution validation found a generator binding/access issue. Attempting automatic repair.")
                        repaired_response = repair_code_with_llm(
                            request=code_request,
                            original_code=code_for_execution,
                            validation_errors=preflight_errors,
                            api_key=api_key or None,
                        )
                        if repaired_response.code.strip():
                            repaired_exec = repaired_response.code
                            repaired_exec, repaired_imports = _augment_code_with_retrieval_imports(
                                code=repaired_exec,
                                retrieved_results=current_aggregated_results,
                            )
                            repaired_errors = validate_generated_code(repaired_exec, sage_bin=sage_bin)
                            if not _needs_generator_variable_repair(repaired_errors):
                                code_response = repaired_response
                                code_for_execution = repaired_exec
                                auto_imports = repaired_imports
                                st.success("Automatic repair resolved the generator binding/access issue.")
                            else:
                                st.warning("Automatic repair still has a generator binding/access issue; keeping original code.")
                        else:
                            st.warning("Automatic repair returned empty code; keeping original code.")

                request_satisfaction = validate_request_satisfaction(
                    resolved_task,
                    generated_code=code_for_execution or code_response.code,
                    execution_result=None,
                )
                if request_satisfaction.summary_messages():
                    merged_missing = list(code_response.missing_info)
                    for message in request_satisfaction.summary_messages():
                        if message not in merged_missing:
                            merged_missing.append(message)
                    code_response.missing_info = merged_missing

                if code_response.code.strip():
                    st.code(code_response.code, language="python")
                else:
                    st.caption("No code generated.")

                executor_caption = _executor_caption(code_response.raw_response)
                if executor_caption:
                    st.caption(executor_caption)

                if auto_imports:
                    st.caption("Auto-added imports from retrieved symbols")
                    st.markdown("\n".join(f"- `{line}`" for line in auto_imports))
                    with st.expander("Execution code (after import fixes)", expanded=False):
                        st.code(code_for_execution, language="python")

                citations_text = _citation_lines(code_response.citations_used, final_context_items)
                st.caption("Citations")
                st.markdown(citations_text)
                _render_retrieved_context_pool(
                    final_context_items,
                    cited_ids=set(code_response.citations_used),
                    step_map=ctx_step_map,
                )

                if code_response.missing_info:
                    st.caption("Missing info")
                    st.markdown("\n".join(f"- {item}" for item in code_response.missing_info))

                with st.expander("Raw output", expanded=False):
                    st.code(code_response.raw_response or "(No raw response captured)", language="json")

                code_status.update(label=code_label, state="complete", expanded=False)

            with st.status("Sage Execution", expanded=False) as exec_status:
                if code_for_execution.strip():
                    execution_result = validate_and_run_sage(
                        code_for_execution,
                        sage_bin=sage_bin,
                        timeout=execution_timeout,
                        use_warm_session=use_warm_sage_session,
                        warm_ttl_seconds=warm_idle_minutes * 60,
                        warm_startup_timeout=min(execution_timeout, 20),
                    )
                else:
                    execution_skipped_reason = (
                        "No code was generated from the retrieved context, so Sage execution was skipped."
                    )

                status_label = _execution_status_label(execution_result)
                if status_label == "success":
                    st.success("success")
                elif status_label == "skipped":
                    st.warning("skipped")
                else:
                    st.error(status_label)

                if execution_result is not None:
                    meta = [
                        f"returncode: {execution_result.returncode}",
                        f"preflight: {'ok' if execution_result.preflight_ok else 'failed'}",
                    ]
                    if execution_result.is_truncated:
                        meta.append("output truncated by Sage")
                    st.caption("  |  ".join(meta))

                has_errors = execution_result is not None and bool(
                    execution_result.stderr.strip() or execution_result.validation_errors
                )
                tab_output, tab_errors = st.tabs(["Output", "Errors ⚠" if has_errors else "Errors"])

                with tab_output:
                    if execution_result is None:
                        st.caption(execution_skipped_reason or "Execution was skipped.")
                    else:
                        stdout_full = execution_result.stdout_full or ""
                        display_text = (execution_result.stdout_summary or "").strip() or stdout_full

                        if not display_text.strip():
                            if execution_result.status == "success":
                                st.caption("Execution succeeded but produced no output. Add `print(...)` in generated code.")
                            else:
                                st.caption("(No output available.)")
                        else:
                            line_limit = 40
                            lines = display_text.splitlines()
                            is_long = len(lines) > line_limit

                            st.code("\n".join(lines[:line_limit]) if is_long else display_text, language="text")

                            info = f"Showing {line_limit} of {len(lines)} lines." if is_long else ""
                            dl = _download_link(stdout_full, "sage_output.txt", "↓ Download full output") if stdout_full.strip() else ""
                            st.markdown(
                                f'<div style="display:flex;justify-content:space-between;align-items:center">'
                                f'<span style="font-size:0.8em;color:gray">{info}</span>{dl}</div>',
                                unsafe_allow_html=True,
                            )

                            if is_long:
                                with st.expander("Show full output"):
                                    st.code(display_text, language="text")

                with tab_errors:
                    if has_errors:
                        st.code(_execution_detail_text(execution_result, execution_skipped_reason), language="text")
                    else:
                        st.caption("No errors.")

                request_satisfaction = validate_request_satisfaction(
                    resolved_task,
                    generated_code=code_for_execution or code_response.code,
                    execution_result=execution_result,
                )
                if request_satisfaction.blocking_issues or request_satisfaction.advisory_issues:
                    st.caption("Request satisfaction checks")
                    for item in request_satisfaction.blocking_issues:
                        st.write(f"- blocking: {item}")
                    for item in request_satisfaction.advisory_issues:
                        st.write(f"- advisory: {item}")

                exec_status.update(state="complete", expanded=False)

            should_retry_with_wildcard = _needs_wildcard_import_fallback(execution_result)
            if should_retry_with_wildcard and code_for_execution.strip():
                modules = _candidate_wildcard_modules(current_aggregated_results, max_modules=2)
                fallback_code, fallback_imports = _augment_code_with_module_wildcards(
                    code_for_execution,
                    modules,
                )
                if fallback_imports and fallback_code != code_for_execution:
                    st.info("Retrying Sage execution with fallback wildcard imports from retrieved modules.")
                    st.markdown("\n".join(f"- `{line}`" for line in fallback_imports))
                    with st.expander("Execution code (wildcard fallback)", expanded=False):
                        st.code(fallback_code, language="python")
                    retry_result = validate_and_run_sage(
                        fallback_code,
                        sage_bin=sage_bin,
                        timeout=execution_timeout,
                        use_warm_session=use_warm_sage_session,
                        warm_ttl_seconds=warm_idle_minutes * 60,
                        warm_startup_timeout=min(execution_timeout, 20),
                    )
                    if retry_result.status == "success":
                        st.success("Fallback execution succeeded with wildcard imports.")
                        code_for_execution = fallback_code
                        execution_result = retry_result
                    else:
                        st.warning("Fallback wildcard import retry did not succeed; keeping original execution result.")

            request_satisfaction = validate_request_satisfaction(
                resolved_task,
                generated_code=code_for_execution or code_response.code,
                execution_result=execution_result,
            )

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

            st.info("Auto route escalated to Refresh Plan retrieval (fresh start).")
            route_mode = "plan"
            route_reason = "Auto route switched to Refresh Plan mode after evidence gap (fresh start)."
            escalated_results, escalated_map = _collect_context_plan(
                question=question,
                planning_hint="",
                payload=payload,
                chunks=chunks,
                index_path=idx,
                k=k,
                mode=mode,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
                retrieval_strategy=retrieval_strategy,
                graph_path=graph_path,
                pdf_char_limit=pdf_char_limit,
                final_context_cap=final_context_cap,
                provider=provider,
                model=model,
                api_key=api_key or None,
                llm_base_url=llm_base_url,
                max_plan_steps=max_plan_steps,
                temperature=temperature,
                llm_max_output_tokens=llm_max_output_tokens,
                show_workflow_retrieval_debug=show_workflow_retrieval_debug,
                section_title="Refresh Plan Retrieval",
            )
            if not escalated_results:
                st.error("Refresh Plan retrieval found no usable context; stopping instead of using stale fast context.")
                return
            current_aggregated_results = escalated_results
            current_chunk_step_map = escalated_map
            escalated_to_plan = True

        st.divider()
        st.subheader("Final Answer")

        use_direct_final = execution_result is not None and execution_result.status == "success"
        if use_direct_final:
            with st.chat_message("assistant"):
                direct_answer_text = _render_direct_sage_final_answer(
                    execution_result=execution_result,
                    execution_skipped_reason=execution_skipped_reason,
                )
            final_response = FinalAnswerResponse(
                answer=direct_answer_text,
                citations_used=code_response.citations_used,
                missing_info=[],
                raw_response="(direct-from-sage-output)",
            )
        else:
            live_placeholder = st.empty()

            def _on_chunk(_piece: str, _acc_text: str) -> None:
                live_placeholder.status("Composing answer...", expanded=False)

            answer_request = ExecutionAwareAnswerRequest(
                question=question,
                contexts=_order_contexts_cited_first(
                    final_context_items,
                    code_response.citations_used,
                ),
                original_code=code_for_execution or code_response.code,
                execution_result=execution_result,
                execution_skipped_reason=execution_skipped_reason,
                code_generation_citations=code_response.citations_used,
                provider=provider,
                model=model,
                temperature=temperature,
                base_url=llm_base_url,
                max_output_tokens=llm_max_output_tokens,
            )

            final_response = answer_with_execution_llm(
                request=answer_request,
                api_key=api_key or None,
                stream=True,
                on_chunk=_on_chunk,
            )
            live_placeholder.empty()

            with st.chat_message("assistant"):
                _render_answer(final_response.answer or "(No answer returned)", context_items=final_context_items)

        with st.expander("Details", expanded=False):
            st.caption("Citations used")
            cited_set = set(final_response.citations_used)
            st.markdown(_citation_lines(final_response.citations_used, final_context_items))
            cited_items = [c for c in final_context_items if c.context_id in cited_set]
            if cited_items:
                with st.expander(f"Cited sources ({len(cited_items)})", expanded=False):
                    for i, cid in enumerate(final_response.citations_used, start=1):
                        item = next((c for c in final_context_items if c.context_id == cid), None)
                        if not item:
                            continue
                        step_id = ctx_step_map.get(cid)
                        step_label = f"step {step_id}" if step_id else "?"
                        st.markdown(f"**[{i}] {item.title}** — {item.location}  `score {item.score:.3f}` · `{step_label}`")
                        preview = item.text[:400] + ("..." if len(item.text) > 400 else "")
                        st.caption(preview)

            if final_response.missing_info:
                st.caption("Missing info")
                st.markdown("\n".join(f"- {item}" for item in final_response.missing_info))

            with st.expander("Raw output", expanded=False):
                st.code(final_response.raw_response or "(No raw response captured)", language="json")

        #stage_placeholder.write=st.empty()
    except Exception as exc:
        st.error(f"Code generation / Sage execution / final answer failed: {exc}")
        return
    
    end_time = time.time()
    runtime = end_time - start_time
    print(f"Runtime: {runtime:.2f} seconds")

if __name__ == "__main__":
    main()
