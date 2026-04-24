#!/usr/bin/env python3
"""Deterministic workflow-specific Sage code builders."""

from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Sequence

from core.llm_service import CodeGenerationRequest, CodeGenerationResponse, ContextItem, generate_code_with_llm
from core.operator_normalization import normalize_operator_expression
from workflows.task_workflows import WorkflowSelection

# Env flag controlling the structured code-plan codegen path. The flag
# is opt-in for now: it adds an extra LLM call and we want runtime
# parity with the existing free-form path until we have collected
# enough end-to-end signal.
CODE_PLAN_ENV_VAR = "ORE_ASSISTANT_USE_CODE_PLAN"


def _code_plan_enabled() -> bool:
    raw = (os.getenv(CODE_PLAN_ENV_VAR, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def _load_precondition_graph_cached():
    """Load the precondition graph artifact, returning ``None`` on failure.

    Cached so we don't re-read the artifact on every code generation.
    Failures are silent: the code-plan path still works without a
    graph (it just skips graph-driven import suggestions).
    """

    try:
        from retrieval.knowledge_base import default_precondition_graph_path  # local import: optional
        from retrieval.precondition_graph import load_precondition_graph
    except Exception:
        return None
    try:
        path = Path(default_precondition_graph_path()).expanduser()
        if not path.exists():
            return None
        return load_precondition_graph(path)
    except Exception:
        return None

SAFE_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s]+$")
SAFE_ATOM_RE = re.compile(r"^[A-Za-z0-9_+\-*/^.]+$")
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SUPPORTED_BASE_RINGS = {"QQ", "ZZ", "RR", "CC", "QQbar"}


def _normalize_text(text: str) -> str:
    lowered = str(text or "").lower()
    return " ".join(lowered.split())


def _safe_expr(text: str) -> str | None:
    value = str(text or "").strip()
    if not value or not SAFE_EXPR_RE.fullmatch(value):
        return None
    return value


def _safe_atom(text: str) -> str | None:
    value = str(text or "").strip()
    if not value or not SAFE_ATOM_RE.fullmatch(value):
        return None
    return value


def _safe_identifier(text: str) -> str | None:
    value = str(text or "").strip()
    if not value or not SAFE_IDENTIFIER_RE.fullmatch(value):
        return None
    return value


def _normalize_operator_expr(
    text: str,
    *,
    known_variables: Sequence[str] = (),
    known_generators: Sequence[str] = (),
) -> str:
    return normalize_operator_expression(
        text,
        known_variables=tuple(known_variables),
        known_generators=tuple(known_generators),
    )


def _cite_contexts(contexts: Sequence[ContextItem], *needles: str) -> list[str]:
    tokens = [_normalize_text(token) for token in needles if str(token).strip()]
    matched: list[str] = []
    for context in contexts:
        haystack = _normalize_text(
            "\n".join((context.title, context.location, context.text))
        )
        if any(token in haystack for token in tokens):
            matched.append(context.context_id)
    return matched


def _build_response(
    *,
    executor_id: str,
    reason: str,
    code: str,
    citations_used: Sequence[str] = (),
    missing_info: Sequence[str] = (),
) -> CodeGenerationResponse:
    return CodeGenerationResponse(
        code=code.strip() + "\n",
        citations_used=list(citations_used),
        missing_info=list(missing_info),
        raw_response=json.dumps(
            {
                "executor": executor_id,
                "reason": reason,
                "used_llm": False,
            },
            ensure_ascii=True,
            indent=2,
        ),
    )


def _build_resolved_task_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    resolved_task = getattr(request, "resolved_task", None)
    if resolved_task is None:
        return None
    resolved_workflow = str(getattr(resolved_task, "workflow_id", "") or "").strip()
    if not resolved_workflow:
        return None
    if workflow_selection.workflow_id and resolved_workflow != workflow_selection.workflow_id:
        return None

    code = ""
    if hasattr(resolved_task, "code"):
        try:
            code = str(resolved_task.code() or "")
        except Exception:
            code = ""
    if not code.strip():
        code_lines = getattr(resolved_task, "code_lines", ()) or ()
        code = "\n".join(str(line) for line in code_lines if str(line).strip())
    if not code.strip():
        return None

    method_name = str(getattr(resolved_task, "method_name", "") or "").strip()
    intent_id = str(getattr(resolved_task, "intent_id", "") or "").strip()
    citation_needles = tuple(getattr(resolved_task, "citation_needles", ()) or ())
    reason_bits = [bit for bit in (method_name, intent_id) if bit]
    reason = (
        "Used the structured resolved task plan."
        if not reason_bits
        else "Used the structured resolved task plan for " + " / ".join(reason_bits) + "."
    )
    return _build_response(
        executor_id=f"resolved_{resolved_workflow.lower()}",
        reason=reason,
        code=code,
        citations_used=_cite_contexts(request.contexts, *citation_needles),
    )


def _build_right_factor_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    question = request.question.strip()
    normalized = _normalize_text(question)
    if workflow_selection.workflow_id != "B2_factor_right_factor" and "right factor" not in normalized:
        return None

    match = re.search(
        r"\boperator\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?P<expr>.+?)\s+in\s+"
        r"(?P<ring>[A-Za-z_][A-Za-z0-9_]*)\[(?P<base_var>[A-Za-z_][A-Za-z0-9_]*)\]"
        r"\[(?P<generator>[A-Za-z_][A-Za-z0-9_]*)\]",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    coefficient_ring = _safe_identifier(match.group("ring"))
    base_var = _safe_identifier(match.group("base_var"))
    generator = _safe_identifier(match.group("generator"))
    operator_expr = _safe_expr(
        _normalize_operator_expr(
            match.group("expr"),
            known_variables=(base_var,) if base_var else (),
            known_generators=(generator,) if generator else (),
        )
    )
    if (
        operator_expr is None
        or coefficient_ring not in SUPPORTED_BASE_RINGS
        or base_var is None
        or generator is None
    ):
        return None

    code = "\n".join(
        [
            "from ore_algebra import OreAlgebra",
            f"R.<{base_var}> = {coefficient_ring}['{base_var}']",
            f"A.<{generator}> = OreAlgebra(R)",
            f"L = {operator_expr}",
            "factors = L.right_factors()",
            "print(factors)",
        ]
    )
    return _build_response(
        executor_id="b2_right_factor",
        reason="Matched a direct right-factor question with explicit shift-algebra notation.",
        code=code,
        citations_used=_cite_contexts(request.contexts, "right_factors"),
    )


def _build_indicial_polynomial_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    question = request.question.strip()
    normalized = _normalize_text(question)
    if workflow_selection.workflow_id != "B3_local_singularity_analysis" and "indicial polynomial" not in normalized:
        return None

    match = re.search(
        r"\bof\s+(?P<expr>.+?)\s+at\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^,\s)]+"
        r"(?:\s+in\s+(?P<ring>[^.?!]+))?",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return None

    singular_var = _safe_identifier(match.group("var"))
    ring_text = str(match.group("ring") or "").strip()
    if singular_var is None or not ring_text:
        return None

    ring_compact = re.sub(r"\s+", "", ring_text)

    setup_lines: list[str] = ["from ore_algebra import OreAlgebra"]
    known_variables: list[str] = []
    known_generators: list[str] = []

    qqxyy_match = re.fullmatch(
        r"(?i:QQ)\((?P<x>[A-Za-z_][A-Za-z0-9_]*)\)\[(?P<y>[A-Za-z_][A-Za-z0-9_]*)\]\[(?P<gen>[A-Za-z_][A-Za-z0-9_]*)\]",
        ring_compact,
    )
    qqxy_match = re.fullmatch(
        r"(?i:QQ)\((?P<vars>[A-Za-z_][A-Za-z0-9_]*(?:,[A-Za-z_][A-Za-z0-9_]*)+)\)\[(?P<gen>[A-Za-z_][A-Za-z0-9_]*)\]",
        ring_compact,
    )

    if qqxyy_match is not None:
        x_var = _safe_identifier(qqxyy_match.group("x"))
        y_var = _safe_identifier(qqxyy_match.group("y"))
        generator = _safe_identifier(qqxyy_match.group("gen"))
        if x_var is None or y_var is None or generator is None:
            return None
        known_variables.extend((x_var, y_var))
        known_generators.append(generator)
        setup_lines.extend(
            [
                f"R.<{x_var}> = QQ['{x_var}']",
                f"Q.<{y_var}> = Frac(R)[]",
                f"A.<{generator}> = OreAlgebra(Q)",
            ]
        )
    elif qqxy_match is not None:
        raw_vars = qqxy_match.group("vars").split(",")
        vars_list = []
        for item in raw_vars:
            value = _safe_identifier(item)
            if value is None:
                return None
            vars_list.append(value)
        generator = _safe_identifier(qqxy_match.group("gen"))
        if not vars_list or generator is None:
            return None
        known_variables.extend(vars_list)
        known_generators.append(generator)
        setup_lines.extend(
            [
                f"R.<{','.join(vars_list)}> = QQ[]",
                "Q = R.fraction_field()",
                f"A.<{generator}> = OreAlgebra(Q)",
            ]
        )
    else:
        return None

    operator_expr = _safe_expr(
        _normalize_operator_expr(
            match.group("expr"),
            known_variables=tuple(known_variables),
            known_generators=tuple(known_generators),
        )
    )
    if operator_expr is None:
        return None

    code = "\n".join(
        [
            *setup_lines,
            f"L = {operator_expr}",
            f"print(L.indicial_polynomial({singular_var}).factor())",
        ]
    )
    return _build_response(
        executor_id="b3_indicial_polynomial",
        reason="Matched an indicial-polynomial request with explicit coefficient-field notation.",
        code=code,
        citations_used=_cite_contexts(request.contexts, "indicial_polynomial", "OreAlgebra", "fraction_field"),
    )


def _build_taylor_recurrence_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    question = request.question.strip()
    normalized = _normalize_text(question)
    if workflow_selection.workflow_id not in {"C2_algebra_conversion", "D2_compute_sequence_terms"} and (
        "taylor" not in normalized or "coefficient recurrence" not in normalized
    ):
        return None

    operator_match = re.search(
        r"\bfor\s+\((?P<expr>.+?)\)\s*f\s*=\s*0",
        question,
        flags=re.IGNORECASE,
    )
    count_match = re.search(
        r"\bfirst\s+(?P<count>\d+)\s+(coefficients|terms|values)\b",
        normalized,
    )
    if not operator_match or not count_match:
        return None

    raw_operator_expr = operator_match.group("expr")
    normalized_operator_expr = _normalize_operator_expr(raw_operator_expr)
    operator_expr = _safe_expr(normalized_operator_expr)
    count = int(count_match.group("count"))
    generator_match = re.search(r"\bD([a-z])\b", operator_expr or "")
    if operator_expr is None or generator_match is None or count <= 0:
        return None

    diff_var = generator_match.group(1)
    diff_generator = f"D{diff_var}"
    code = "\n".join(
        [
            "from ore_algebra import DifferentialOperators, OreAlgebra",
            f"Dops, {diff_var}, {diff_generator} = DifferentialOperators(QQ, '{diff_var}')",
            "R2.<n> = QQ['n']",
            "A2.<Sn> = OreAlgebra(R2)",
            f"L = {operator_expr}",
            "rec = L.to_S(A2)",
            "print(rec)",
            f"print(rec.to_list([1], {count}))",
        ]
    )
    return _build_response(
        executor_id="c2_taylor_to_list",
        reason="Matched a differential-to-recurrence question that also asked for coefficient terms.",
        code=code,
        citations_used=_cite_contexts(request.contexts, "to_s", "to_list"),
        missing_info=("No initial coefficient was specified, so the Taylor series was normalized with a_0 = 1.",),
    )


def _build_numerical_solution_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    question = request.question.strip()
    normalized = _normalize_text(question)
    if workflow_selection.workflow_id != "E1_numerical_eval_differential_solution" and "evaluate the solution" not in normalized:
        return None

    operator_match = re.search(
        r"\bsolution\s+of\s+\((?P<expr>.+?)\)\s*f\s*=\s*0",
        question,
        flags=re.IGNORECASE,
    )
    initial_match = re.search(
        r"\bwith\s+f\(\s*(?P<initial_point>[^)]+?)\s*\)\s*=\s*(?P<initial_value>[^\s,.;]+)",
        question,
        flags=re.IGNORECASE,
    )
    target_match = re.search(
        r"\bat\s+[A-Za-z_][A-Za-z0-9_]*\s*=\s*(?P<target>[^\s,.;?!]+)",
        question,
        flags=re.IGNORECASE,
    )
    if not operator_match or not initial_match or not target_match:
        return None

    raw_operator_expr = operator_match.group("expr")
    normalized_operator_expr = _normalize_operator_expr(raw_operator_expr)
    operator_expr = _safe_expr(normalized_operator_expr)
    initial_point = _safe_atom(initial_match.group("initial_point"))
    initial_value = _safe_atom(initial_match.group("initial_value"))
    target_point = _safe_atom(target_match.group("target"))
    generator_match = re.search(r"\bD([a-z])\b", operator_expr or "")
    if (
        operator_expr is None
        or initial_point is None
        or initial_value is None
        or target_point is None
        or generator_match is None
    ):
        return None

    diff_var = generator_match.group(1)
    diff_generator = f"D{diff_var}"
    code = "\n".join(
        [
            "from ore_algebra import DifferentialOperators",
            f"Dops, {diff_var}, {diff_generator} = DifferentialOperators(QQ, '{diff_var}')",
            f"L = {operator_expr}",
            f"value = L.numerical_solution([{initial_value}], [{initial_point}, {target_point}])",
            "print(value)",
        ]
    )
    return _build_response(
        executor_id="e1_numerical_solution",
        reason="Matched a first-order differential IVP with an explicit evaluation point.",
        code=code,
        citations_used=_cite_contexts(request.contexts, "numerical_solution", "differentialoperators"),
    )


def _build_guess_then_terms_code(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    question = request.question.strip()
    normalized = _normalize_text(question)
    if workflow_selection.workflow_id not in {"F1_guess_from_data", "D2_compute_sequence_terms"} and "infer a recurrence" not in normalized:
        return None

    sequence_match = re.search(
        r"\bterms\s+(?P<data>.+?)\s+and\s+print\s+the\s+first\s+(?P<count>\d+)\s+(values|terms)\b",
        question,
        flags=re.IGNORECASE,
    )
    if not sequence_match:
        return None

    count = int(sequence_match.group("count"))
    raw_items = [item.strip() for item in sequence_match.group("data").split(",")]
    data_items = [_safe_atom(item) for item in raw_items if item]
    if count <= 0 or not data_items or any(item is None for item in data_items):
        return None

    data_literal = ", ".join(item for item in data_items if item is not None)
    code = "\n".join(
        [
            "from ore_algebra import OreAlgebra",
            "from ore_algebra.guessing import guess_rec",
            "R.<n> = QQ['n']",
            "A.<Sn> = OreAlgebra(R)",
            f"data = [{data_literal}]",
            "rec = guess_rec(data, n, Sn)",
            "print(rec)",
            "init = data[:max(1, rec.order())]",
            f"print(rec.to_list(init, {count}))",
        ]
    )
    return _build_response(
        executor_id="f1_guess_then_terms",
        reason="Matched a guess-from-data question that immediately asked for continued sequence terms.",
        code=code,
        citations_used=_cite_contexts(request.contexts, "guess_rec", "to_list"),
    )


EXECUTOR_BUILDERS = (
    _build_resolved_task_code,
    _build_right_factor_code,
    _build_indicial_polynomial_code,
    _build_taylor_recurrence_code,
    _build_numerical_solution_code,
    _build_guess_then_terms_code,
)


def try_generate_code_with_executor(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
) -> CodeGenerationResponse | None:
    for builder in EXECUTOR_BUILDERS:
        response = builder(request=request, workflow_selection=workflow_selection)
        if response is not None:
            return response
    return None


def generate_code_with_executors(
    request: CodeGenerationRequest,
    workflow_selection: WorkflowSelection,
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
) -> CodeGenerationResponse:
    response = try_generate_code_with_executor(request=request, workflow_selection=workflow_selection)
    if response is not None:
        return response

    # Optional structured code-plan path. Falls back to free-form
    # codegen on any LLM/parse failure so this stays drop-in safe.
    if _code_plan_enabled():
        try:
            from core.code_plan import generate_code_with_plan  # local import: optional
        except Exception:
            generate_code_with_plan = None  # type: ignore[assignment]
        if generate_code_with_plan is not None:
            try:
                graph = _load_precondition_graph_cached()
                plan_response = generate_code_with_plan(
                    request=request,
                    graph=graph,
                    api_key=api_key,
                )
                if plan_response.code.strip():
                    return plan_response
            except Exception:
                # Fall through to the free-form path; we never want a
                # code-plan failure to block the chat run.
                pass

    return generate_code_with_llm(
        request=request,
        api_key=api_key,
        parse_repair_attempts=parse_repair_attempts,
    )
