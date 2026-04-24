#!/usr/bin/env python3
"""Deterministic task resolution and request-satisfaction checks.

This module takes a structured ``TaskUnderstanding`` object and resolves
it into a narrower executable plan for workflows where broad workflow
labels are still too coarse.  The first rollout covers the failure-heavy
families:

- B2 right-factor search
- B3 local basis expansions / indicial polynomial
- C1 annihilator_of_polynomial / annihilator_of_associate

The same resolved object is then reused by deterministic executors and
by lightweight request-satisfaction validation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from core.operator_normalization import (
    clean_operator_expression,
    extract_question_symbols,
    normalize_operator_expression,
)
from core.sage_runtime import SageExecutionResult
from core.task_understanding import ParsedRequest, TaskUnderstanding

SAFE_EXPR_RE = re.compile(r"^[A-Za-z0-9_+\-*/^().,\s=\[\]]+$")
SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
SAFE_Q_RE = re.compile(r"^[A-Za-z0-9_+\-*/^.]+$")
SUPPORTED_SCALAR_RINGS = {"QQ", "ZZ", "RR", "CC", "QQbar"}


@dataclass(frozen=True)
class RingSpec:
    scalar_ring: str
    base_var: str
    generator: str
    use_fraction_field: bool = False
    q_value: str = ""

    def build_setup_lines(self, *, import_ore_algebra: bool = True) -> tuple[tuple[str, ...], str]:
        lines: list[str] = []
        if import_ore_algebra:
            lines.append("from ore_algebra import OreAlgebra")

        coeff_binding = "R"
        if self.use_fraction_field:
            coeff_binding = "K"
            lines.append(f"K.<{self.base_var}> = {self.scalar_ring}['{self.base_var}']")
            lines.append("K = K.fraction_field()")
        else:
            lines.append(f"R.<{self.base_var}> = {self.scalar_ring}['{self.base_var}']")

        if self.q_value and self.generator.startswith("Q"):
            lines.append(f"A.<{self.generator}> = OreAlgebra({coeff_binding}, q={self.q_value})")
        else:
            lines.append(f"A.<{self.generator}> = OreAlgebra({coeff_binding}, '{self.generator}')")
        return tuple(lines), coeff_binding


@dataclass(frozen=True)
class ResolvedTask:
    workflow_id: str
    intent_id: str
    method_name: str
    code_lines: tuple[str, ...]
    helper_actions: tuple[str, ...] = ()
    obligations: tuple[str, ...] = ()
    citation_needles: tuple[str, ...] = ()
    operator_text: str = ""
    point_source: str = ""
    point_selection: str = ""
    resolved_point_expr: str = ""
    needs_base_ring: bool = False
    wants_boolean_answer: bool = False
    wants_first_element_only: bool = False
    polynomial_expression: str = ""
    associate_expression: str = ""
    receiver_binding: str = ""
    argument_binding: str = ""
    receiver_kind: str = ""

    def code(self) -> str:
        return "\n".join(line for line in self.code_lines if str(line).strip()).strip() + "\n"

    def debug_lines(self) -> list[str]:
        lines = [
            f"workflow={self.workflow_id}",
            f"intent={self.intent_id}",
            f"method={self.method_name}",
        ]
        if self.operator_text:
            lines.append(f"operator={self.operator_text}")
        if self.point_source:
            point_bits = [self.point_source]
            if self.point_selection:
                point_bits.append(self.point_selection)
            if self.resolved_point_expr:
                point_bits.append(self.resolved_point_expr)
            lines.append("point=" + " | ".join(point_bits))
        if self.needs_base_ring:
            lines.append("needs_base_ring=true")
        if self.wants_boolean_answer:
            lines.append("wants_boolean_answer=true")
        if self.wants_first_element_only:
            lines.append("wants_first_element_only=true")
        if self.receiver_binding or self.argument_binding:
            role_bits = []
            if self.receiver_binding:
                role_bits.append(f"receiver={self.receiver_binding}")
            if self.argument_binding:
                role_bits.append(f"argument={self.argument_binding}")
            if self.receiver_kind:
                role_bits.append(f"receiver_kind={self.receiver_kind}")
            lines.append("roles=" + " | ".join(role_bits))
        if self.helper_actions:
            lines.append("helper_actions=" + ", ".join(self.helper_actions))
        if self.obligations:
            lines.append("obligations=" + ", ".join(self.obligations))
        return lines

    def prompt_hint(self) -> str:
        lines = [
            "Resolved execution plan:",
            f"- method: {self.method_name}",
        ]
        if self.point_source:
            lines.append(f"- point_source: {self.point_source}")
        if self.point_selection:
            lines.append(f"- point_selection: {self.point_selection}")
        if self.resolved_point_expr:
            lines.append(f"- resolved_point_expr: {self.resolved_point_expr}")
        if self.needs_base_ring:
            lines.append("- must_show_base_ring: true")
        if self.wants_boolean_answer:
            lines.append("- must_answer_existence_question: true")
        if self.wants_first_element_only:
            lines.append("- must_print_only_first_requested_element: true")
        if self.receiver_binding:
            lines.append(f"- receiver_binding: {self.receiver_binding}")
        if self.argument_binding:
            lines.append(f"- argument_binding: {self.argument_binding}")
        if self.receiver_kind:
            lines.append(f"- receiver_kind: {self.receiver_kind}")
        if self.helper_actions:
            lines.append("- helper_actions:")
            lines.extend(f"  - {item}" for item in self.helper_actions)
        return "\n".join(lines)

    def as_payload(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "intent_id": self.intent_id,
            "method_name": self.method_name,
            "code_lines": list(self.code_lines),
            "helper_actions": list(self.helper_actions),
            "obligations": list(self.obligations),
            "citation_needles": list(self.citation_needles),
            "operator_text": self.operator_text,
            "point_source": self.point_source,
            "point_selection": self.point_selection,
            "resolved_point_expr": self.resolved_point_expr,
            "needs_base_ring": self.needs_base_ring,
            "wants_boolean_answer": self.wants_boolean_answer,
            "wants_first_element_only": self.wants_first_element_only,
            "polynomial_expression": self.polynomial_expression,
            "associate_expression": self.associate_expression,
            "receiver_binding": self.receiver_binding,
            "argument_binding": self.argument_binding,
            "receiver_kind": self.receiver_kind,
        }


@dataclass(frozen=True)
class RequestSatisfactionReport:
    passed: bool
    blocking_issues: tuple[str, ...] = ()
    advisory_issues: tuple[str, ...] = ()
    obligations_checked: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "blocking_issues": list(self.blocking_issues),
            "advisory_issues": list(self.advisory_issues),
            "obligations_checked": list(self.obligations_checked),
        }

    def summary_messages(self) -> list[str]:
        return list(self.blocking_issues) + list(self.advisory_issues)


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _safe_expr(text: str) -> str | None:
    value = str(text or "").strip().strip(" .,:;")
    if not value or not SAFE_EXPR_RE.fullmatch(value):
        return None
    return value


def _safe_identifier(text: str) -> str | None:
    value = str(text or "").strip()
    if not value or not SAFE_IDENTIFIER_RE.fullmatch(value):
        return None
    return value


def _safe_q_value(text: str) -> str | None:
    value = str(text or "").strip().strip(" .,:;")
    if not value or not SAFE_Q_RE.fullmatch(value):
        return None
    return value


def _clean_operator_expr(text: str) -> str:
    return clean_operator_expression(text)


def _normalization_symbols(
    question: str,
    parsed: ParsedRequest,
    expr_text: str = "",
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    variables, generators = extract_question_symbols(question)
    fallback_generator = _safe_identifier(_extract_generator(expr_text or parsed.operator_text, parsed))
    if fallback_generator:
        generators = tuple(dict.fromkeys((*generators, fallback_generator)))
        suffix = _safe_identifier(fallback_generator[1:])
        if suffix:
            variables = tuple(dict.fromkeys((*variables, suffix)))
    return variables, generators


def _normalize_operator_expr(text: str, parsed: ParsedRequest, question: str = "") -> str:
    variables, generators = _normalization_symbols(question, parsed, str(text or ""))
    return normalize_operator_expression(
        text,
        known_variables=variables,
        known_generators=generators,
    )


def _extract_generator(expr: str, parsed: ParsedRequest) -> str:
    match = re.search(r"\b([DSFTQJ][A-Za-z_][A-Za-z0-9_]*)\b", expr)
    if match:
        return match.group(1)
    if parsed.operator_kind == "differential":
        return "Dx"
    if parsed.operator_kind == "q_recurrence":
        return "Qn"
    if parsed.operator_kind == "recurrence":
        return "Sn"
    return ""


def _default_scalar_ring(expr: str) -> str:
    if "/" in expr:
        return "QQ"
    return "ZZ"


def _parse_q_value(question: str) -> str:
    match = re.search(r"\bq\s*=\s*(?P<q>[-+]?[A-Za-z0-9_./]+)", question, flags=re.IGNORECASE)
    value = _safe_q_value(match.group("q")) if match else None
    return value or ""


def _parse_ring_spec(question: str, parsed: ParsedRequest, operator_expr: str) -> RingSpec | None:
    match = re.search(
        r"\bin\s+(?P<scalar>QQ|ZZ|RR|CC|QQbar)\s*"
        r"(?:\(\s*(?P<frac_var>[A-Za-z_][A-Za-z0-9_]*)\s*\)|\[\s*(?P<poly_var>[A-Za-z_][A-Za-z0-9_]*)\s*\])"
        r"\s*\[\s*(?P<generator>[A-Za-z_][A-Za-z0-9_]*)\s*\]",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        scalar_ring = str(match.group("scalar")).strip()
        base_var = _safe_identifier(match.group("frac_var") or match.group("poly_var") or "")
        generator = _safe_identifier(match.group("generator") or "")
        if scalar_ring not in SUPPORTED_SCALAR_RINGS or base_var is None or generator is None:
            return None
        return RingSpec(
            scalar_ring=scalar_ring,
            base_var=base_var,
            generator=generator,
            use_fraction_field=bool(match.group("frac_var")),
            q_value=_parse_q_value(question),
        )

    generator = _safe_identifier(_extract_generator(operator_expr, parsed))
    if generator is None:
        return None
    base_var = _safe_identifier(generator[1:] or "")
    if base_var is None:
        return None
    return RingSpec(
        scalar_ring=_default_scalar_ring(operator_expr),
        base_var=base_var,
        generator=generator,
        use_fraction_field=False,
        q_value=_parse_q_value(question),
    )


def _clean_point_expr(parsed: ParsedRequest) -> str:
    point = str(parsed.point_text or "").strip().strip(" .,:;")
    point = re.sub(r"^[A-Za-z_][A-Za-z0-9_]*\s*=\s*", "", point)
    if point.lower() == "the origin":
        return "0"
    if parsed.point_kind == "algebraic" and "QQbar" not in point and any(
        token in point for token in ("sqrt", "I")
    ):
        return f"QQbar({point})"
    return point


def _resolve_point_code(parsed: ParsedRequest, normalized_question: str) -> tuple[tuple[str, ...], str]:
    if parsed.point_source == "leading_coefficient_root":
        lines = [
            "lc = L.leading_coefficient()",
            "roots = [rt for rt, _ in lc.roots(AA)]",
        ]
        index = 0
        if parsed.point_selection == "second_real":
            index = 1
        lines.append(f"alpha = roots[{index}]")
        return tuple(lines), "alpha"

    if parsed.point_source == "explicit_point":
        point_expr = _clean_point_expr(parsed)
        if point_expr:
            return (), point_expr

    if "origin" in normalized_question:
        return (), "0"

    return (), ""


def _extract_b3_indicial_var(question: str) -> str:
    match = re.search(
        r"\bat\s+(?P<var>[A-Za-z_][A-Za-z0-9_]*)\s*=",
        question,
        flags=re.IGNORECASE,
    )
    return _safe_identifier(match.group("var")) if match else ""


def _extract_c1_polynomial_expression(question: str) -> str:
    patterns = (
        r"\bpolynomial expression\s+(?P<expr>.+?)\s+in\s+",
        r"\bannihilating operator of the polynomial expression\s+(?P<expr>.+?)\s+in\s+",
        r"\bannihilating operator of\s+(?P<expr>.+?)\s+in\s+",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        expr = _safe_expr(match.group("expr"))
        if expr:
            return expr
    return ""


def _extract_c1_operator_expression(question: str, parsed: ParsedRequest) -> str:
    patterns = (
        r"\bwhere\s+y0\s+is\s+any\s+solution\s+of\s+the\s+operator\s+(?P<expr>.+?)(?:[.?!]|$)",
        r"\bsolution\s+of\s+the\s+operator\s+(?P<expr>.+?)(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        expr = _safe_expr(_normalize_operator_expr(match.group("expr"), parsed, question))
        if expr:
            return expr
    expr = _safe_expr(_normalize_operator_expr(parsed.operator_text, parsed, question))
    return expr or ""


def _extract_association_labels(question: str) -> tuple[str, str]:
    match = re.search(
        r"\bassociation\s+(?P<receiver>[A-Za-z_][A-Za-z0-9_]*)\((?P<solution>[A-Za-z_][A-Za-z0-9_]*)\)",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return "L", "f"
    receiver = _safe_identifier(match.group("receiver")) or "L"
    solution = _safe_identifier(match.group("solution")) or "f"
    return receiver, solution


def _extract_c1_associate_receiver_expression(question: str, receiver_label: str) -> str:
    escaped = re.escape(receiver_label)
    patterns = (
        rf"\b{escaped}\s*=\s*(?P<expr>.+?)(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        expr = _safe_expr(_clean_operator_expr(match.group("expr")))
        if expr:
            return expr
    return ""


def _extract_c1_associate_argument_expression(
    question: str,
    receiver_label: str,
    solution_label: str,
) -> str:
    receiver_escaped = re.escape(receiver_label)
    solution_escaped = re.escape(solution_label)
    patterns = (
        rf"\bwhere\s+{solution_escaped}\s+is\s+any\s+solution\s+of\s+(?:the\s+operator\s+)?"
        rf"(?P<expr>.+?)\s+and\s+{receiver_escaped}\s*(?:=|is\b)",
        rf"\bwhere\s+{solution_escaped}\s+is\s+any\s+solution\s+of\s+(?:the\s+operator\s+)?"
        rf"(?P<expr>.+?)(?:[.?!]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        expr = _safe_expr(_clean_operator_expr(match.group("expr")))
        if expr:
            return expr
    return ""


def _extract_b2_operator_expression(question: str, parsed: ParsedRequest) -> str:
    expr = _safe_expr(_normalize_operator_expr(parsed.operator_text, parsed, question))
    if expr:
        return expr

    patterns = (
        r"^(?:does|do|is|are|has|have)\s+(?P<expr>.+?)\s+have\s+(?:first-order\s+|first order\s+)?right factors?\s+in\b",
        r"\boperator\s+(?P<expr>.+?)\s+in\b",
    )
    for pattern in patterns:
        match = re.search(pattern, question, flags=re.IGNORECASE)
        if not match:
            continue
        expr = _safe_expr(_normalize_operator_expr(match.group("expr"), parsed, question))
        if expr:
            return expr
    return ""


def _polynomial_symbols(expr: str) -> tuple[str, ...]:
    indices = [int(match.group(1)) for match in re.finditer(r"\by(\d+)\b", expr)]
    if not indices:
        return ("y0", "y1", "y2")
    max_idx = max(indices)
    return tuple(f"y{i}" for i in range(max_idx + 1))


def _resolve_b2(understanding: TaskUnderstanding) -> ResolvedTask | None:
    parsed = understanding.parsed_request
    operator_expr = _extract_b2_operator_expression(parsed.raw_question, parsed)
    if not operator_expr:
        return None
    ring_spec = _parse_ring_spec(parsed.raw_question, parsed, operator_expr)
    if ring_spec is None:
        return None
    setup_lines, _ = ring_spec.build_setup_lines()
    first_order_requested = "first-order" in parsed.normalized_question or "first order" in parsed.normalized_question
    factor_call = "factors = L.right_factors(order=1)" if first_order_requested else "factors = L.right_factors()"
    print_lines = ["print(factors)"]
    obligations = ["use:right_factors", "show:right_factors"]
    if parsed.wants_existence_check:
        print_lines.append("print(len(factors) > 0)")
        obligations.append("answer:existence_check")

    return ResolvedTask(
        workflow_id="B2_factor_right_factor",
        intent_id=understanding.intent.intent_id or "right_factor_search",
        method_name="right_factors",
        code_lines=(
            *setup_lines,
            f"L = {operator_expr}",
            factor_call,
            *print_lines,
        ),
        helper_actions=understanding.intent.helper_actions,
        obligations=tuple(obligations),
        citation_needles=("right_factors", "OreAlgebra"),
        operator_text=operator_expr,
        wants_boolean_answer=parsed.wants_existence_check,
    )


def _resolve_b3_local_basis(understanding: TaskUnderstanding) -> ResolvedTask | None:
    parsed = understanding.parsed_request
    operator_expr = _safe_expr(_normalize_operator_expr(parsed.operator_text, parsed, parsed.raw_question))
    if operator_expr is None:
        return None
    ring_spec = _parse_ring_spec(parsed.raw_question, parsed, operator_expr)
    if ring_spec is None:
        return None
    point_lines, point_expr = _resolve_point_code(parsed, parsed.normalized_question)
    if not point_expr:
        return None

    setup_lines, _ = ring_spec.build_setup_lines()
    if parsed.precision is not None:
        basis_call = f"basis = L.local_basis_expansions({point_expr}, order={parsed.precision})"
    else:
        basis_call = f"basis = L.local_basis_expansions({point_expr})"

    print_lines = ["print(basis[0])" if parsed.wants_first_element_only else "print(basis)"]
    obligations = ["use:local_basis_expansions", "show:local_basis"]
    if parsed.point_source == "leading_coefficient_root":
        obligations.append("resolve:leading_coefficient_root")
    if parsed.needs_base_ring:
        print_lines.append("print(basis[0].base_ring())")
        obligations.append("show:base_ring")
    if parsed.wants_first_element_only:
        obligations.append("show:first_element_only")

    return ResolvedTask(
        workflow_id="B3_local_singularity_analysis",
        intent_id=understanding.intent.intent_id or "local_basis_expansions",
        method_name="local_basis_expansions",
        code_lines=(
            *setup_lines,
            f"L = {operator_expr}",
            *point_lines,
            basis_call,
            *print_lines,
        ),
        helper_actions=understanding.intent.helper_actions,
        obligations=tuple(obligations),
        citation_needles=("local_basis_expansions", "OreAlgebra", "leading_coefficient", "base_ring"),
        operator_text=operator_expr,
        point_source=parsed.point_source,
        point_selection=parsed.point_selection,
        resolved_point_expr=point_expr,
        needs_base_ring=parsed.needs_base_ring,
        wants_first_element_only=parsed.wants_first_element_only,
    )


def _resolve_b3_indicial(understanding: TaskUnderstanding) -> ResolvedTask | None:
    parsed = understanding.parsed_request
    operator_expr = _safe_expr(_normalize_operator_expr(parsed.operator_text, parsed, parsed.raw_question))
    singular_var = _extract_b3_indicial_var(parsed.raw_question)
    if operator_expr is None or not singular_var:
        return None
    ring_spec = _parse_ring_spec(parsed.raw_question, parsed, operator_expr)
    if ring_spec is None:
        return None
    setup_lines, _ = ring_spec.build_setup_lines()
    return ResolvedTask(
        workflow_id="B3_local_singularity_analysis",
        intent_id="indicial_polynomial",
        method_name="indicial_polynomial",
        code_lines=(
            *setup_lines,
            f"L = {operator_expr}",
            f"print(L.indicial_polynomial({singular_var}).factor())",
        ),
        obligations=("use:indicial_polynomial", "show:indicial_polynomial"),
        citation_needles=("indicial_polynomial", "OreAlgebra"),
        operator_text=operator_expr,
    )


def _resolve_c1_annihilator_of_polynomial(understanding: TaskUnderstanding) -> ResolvedTask | None:
    parsed = understanding.parsed_request
    polynomial_expr = _extract_c1_polynomial_expression(parsed.raw_question)
    operator_expr = _extract_c1_operator_expression(parsed.raw_question, parsed)
    if not polynomial_expr or not operator_expr:
        return None
    ring_spec = _parse_ring_spec(parsed.raw_question, parsed, operator_expr)
    if ring_spec is None:
        return None
    setup_lines, coeff_binding = ring_spec.build_setup_lines()
    poly_symbols = _polynomial_symbols(polynomial_expr)
    poly_ring_line = f"P.<{','.join(poly_symbols)}> = {coeff_binding}[]"
    return ResolvedTask(
        workflow_id="C1_closure_combinatorics",
        intent_id="annihilator_of_polynomial",
        method_name="annihilator_of_polynomial",
        code_lines=(
            *setup_lines,
            poly_ring_line,
            f"L = {operator_expr}",
            f"p = {polynomial_expr}",
            "print(L.annihilator_of_polynomial(p))",
        ),
        obligations=("use:annihilator_of_polynomial", "show:annihilator"),
        citation_needles=("annihilator_of_polynomial", "OreAlgebra"),
        operator_text=operator_expr,
        polynomial_expression=polynomial_expr,
    )


def _resolve_c1_annihilator_of_associate(understanding: TaskUnderstanding) -> ResolvedTask | None:
    parsed = understanding.parsed_request
    receiver_label, solution_label = _extract_association_labels(parsed.raw_question)
    receiver_expr = _extract_c1_associate_receiver_expression(parsed.raw_question, receiver_label)
    associate_expr = _extract_c1_associate_argument_expression(
        parsed.raw_question,
        receiver_label,
        solution_label,
    )
    receiver_expr = _safe_expr(_normalize_operator_expr(receiver_expr, parsed, parsed.raw_question)) or ""
    associate_expr = _safe_expr(_normalize_operator_expr(associate_expr, parsed, parsed.raw_question)) or ""
    if not receiver_expr or not associate_expr:
        return None

    ring_spec = _parse_ring_spec(parsed.raw_question, parsed, receiver_expr or associate_expr)
    if ring_spec is None:
        return None

    setup_lines, _ = ring_spec.build_setup_lines()
    return ResolvedTask(
        workflow_id="C1_closure_combinatorics",
        intent_id="annihilator_of_associate",
        method_name="annihilator_of_associate",
        code_lines=(
            *setup_lines,
            f"F = {associate_expr}",
            f"L = {receiver_expr}",
            "print(L.annihilator_of_associate(F))",
        ),
        helper_actions=understanding.intent.helper_actions,
        obligations=(
            "use:annihilator_of_associate",
            "show:annihilator",
            "bind:receiver_operator",
            "bind:associate_operator",
        ),
        citation_needles=("annihilator_of_associate", "OreAlgebra"),
        operator_text=receiver_expr,
        associate_expression=associate_expr,
        receiver_binding="L",
        argument_binding="F",
        receiver_kind="operator",
    )


def resolve_task(understanding: TaskUnderstanding | None) -> ResolvedTask | None:
    if understanding is None or not understanding.is_actionable:
        return None
    workflow_id = understanding.intent.workflow_id
    intent_id = understanding.intent.intent_id
    if workflow_id == "B2_factor_right_factor":
        return _resolve_b2(understanding)
    if workflow_id == "B3_local_singularity_analysis":
        if intent_id == "indicial_polynomial":
            return _resolve_b3_indicial(understanding)
        if intent_id == "local_basis_expansions":
            return _resolve_b3_local_basis(understanding)
        return None
    if workflow_id == "C1_closure_combinatorics" and intent_id == "annihilator_of_polynomial":
        return _resolve_c1_annihilator_of_polynomial(understanding)
    if workflow_id == "C1_closure_combinatorics" and intent_id == "annihilator_of_associate":
        return _resolve_c1_annihilator_of_associate(understanding)
    return None


def validate_request_satisfaction(
    resolved_task: ResolvedTask | None,
    *,
    generated_code: str,
    execution_result: SageExecutionResult | None = None,
) -> RequestSatisfactionReport:
    if resolved_task is None:
        return RequestSatisfactionReport(passed=True)

    code = str(generated_code or "").strip()
    blocking: list[str] = []
    advisory: list[str] = []
    obligations = list(resolved_task.obligations)

    def require(predicate: bool, message: str) -> None:
        if not predicate and message not in blocking:
            blocking.append(message)

    if resolved_task.method_name == "right_factors":
        require(".right_factors(" in code, "Generated code does not call `right_factors`.")
        if resolved_task.wants_boolean_answer:
            require(
                "len(factors) > 0" in code or "bool(" in code,
                "Generated code does not answer the requested existence check for right factors.",
            )
    elif resolved_task.method_name == "local_basis_expansions":
        require(
            ".local_basis_expansions(" in code,
            "Generated code does not call `local_basis_expansions`.",
        )
        if resolved_task.point_source == "leading_coefficient_root":
            require(
                "leading_coefficient" in code and ".roots(" in code,
                "Generated code does not resolve the requested leading-coefficient root before expansion.",
            )
        if resolved_task.needs_base_ring:
            require(
                "base_ring()" in code,
                "Generated code does not print the requested base ring.",
            )
        if resolved_task.wants_first_element_only:
            require(
                "[0]" in code,
                "Generated code does not restrict output to the requested first local basis element.",
            )
    elif resolved_task.method_name == "indicial_polynomial":
        require(
            ".indicial_polynomial(" in code,
            "Generated code does not call `indicial_polynomial`.",
        )
    elif resolved_task.method_name == "annihilator_of_polynomial":
        require(
            ".annihilator_of_polynomial(" in code,
            "Generated code does not call `annihilator_of_polynomial`.",
        )
        require(
            "p =" in code or resolved_task.polynomial_expression in code,
            "Generated code does not define the requested polynomial expression.",
        )
    elif resolved_task.method_name == "annihilator_of_associate":
        require(
            ".annihilator_of_associate(" in code,
            "Generated code does not call `annihilator_of_associate`.",
        )
        normalized_code = re.sub(r"\s+", "", code)
        receiver_binding = resolved_task.receiver_binding or "L"
        argument_binding = resolved_task.argument_binding or "F"
        require(
            f"{receiver_binding}.annihilator_of_associate({argument_binding})" in normalized_code,
            "Generated code does not use the resolved `receiver.annihilator_of_associate(argument)` call shape.",
        )
        require(
            f"{receiver_binding}=" in normalized_code,
            "Generated code does not define the resolved receiver operator.",
        )
        require(
            f"{argument_binding}=" in normalized_code,
            "Generated code does not define the resolved associate operator.",
        )
        require(
            "A.annihilator_of_associate(" not in normalized_code,
            "Generated code calls `annihilator_of_associate` on the Ore algebra instead of on the resolved operator.",
        )

    if execution_result is not None:
        if execution_result.status != "success":
            blocking.append(
                f"Execution status was `{execution_result.status}`, so the request was not fully satisfied."
            )
        else:
            stdout = str(execution_result.stdout_summary or execution_result.stdout_full or "")
            lines = [line.strip() for line in stdout.splitlines() if line.strip()]
            if resolved_task.needs_base_ring and len(lines) < 2:
                blocking.append("Execution output does not show both the local basis result and the requested base ring.")
            if resolved_task.wants_boolean_answer and not any(line in {"True", "False"} for line in lines):
                advisory.append("Execution output does not visibly include a boolean existence answer.")
            if resolved_task.method_name == "annihilator_of_polynomial" and not lines:
                blocking.append("Execution output is empty, so no annihilating operator was shown.")
            if resolved_task.method_name == "annihilator_of_associate" and not lines:
                blocking.append("Execution output is empty, so no annihilating operator was shown.")

    return RequestSatisfactionReport(
        passed=not blocking,
        blocking_issues=tuple(blocking),
        advisory_issues=tuple(advisory),
        obligations_checked=tuple(obligations),
    )
