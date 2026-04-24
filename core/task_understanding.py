#!/usr/bin/env python3
"""Structured task-understanding helpers for ore_algebra questions.

This module is the first rollout slice of a richer task-understanding
layer.  It intentionally stays deterministic:

- parse a small, typed request object from the raw question,
- classify a more precise operational intent than the current broad
  workflow buckets,
- expose helper/action hints for downstream code generation,
- optionally override workflow selection in Fast mode so retrieval and
  codegen see a narrower task frame.

The goal of this first cut is not to solve every ore_algebra task.  It
is to reduce the most common "wrong workflow / wrong method family"
mistakes while keeping the current planner mode unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from workflows.task_workflows import WorkflowSelection, get_family, load_workflow_registry


_TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_POINT_AT_RE = re.compile(
    r"\b(?:at|around)\s+(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)?(?P<point>[^,.;]+)",
    flags=re.IGNORECASE,
)
_ORDER_RE = re.compile(r"\border(?:\s*=|\s+of\s+)(?P<n>\d+)\b", flags=re.IGNORECASE)
_FIRST_N_RE = re.compile(
    r"\bfirst\s+(?P<n>\d+)\s+(?:items|terms|coefficients|values|elements)\b",
    flags=re.IGNORECASE,
)
_BITS_RE = re.compile(
    r"\bprecision(?:\s+of)?\s+(?P<n>\d+)\s+bits?\b",
    flags=re.IGNORECASE,
)
_METHOD_QUERY_LIMIT = 4
_ALGEBRA_PROPERTY_METHODS = {
    "is_c": "is_C",
    "is_d": "is_D",
    "is_delta": "is_Delta",
    "is_e": "is_E",
    "is_f": "is_F",
    "is_j": "is_J",
    "is_q": "is_Q",
    "is_s": "is_S",
}
_A_FEATURE_METHOD_HINTS = {
    "associated_commutative_algebra": ("associated_commutative_algebra", "OreAlgebra"),
    "change_constant_ring": ("change_constant_ring", "OreAlgebra"),
    "change_ring": ("change_ring", "OreAlgebra"),
    "delta": ("delta", "OreAlgebra"),
    "sigma": ("sigma", "OreAlgebra"),
    "DifferentialOperators": ("DifferentialOperators", "OreAlgebra"),
    "gens": ("OreAlgebra_generic.gens", "gens", "OreAlgebra"),
    "is_C": ("is_C", "OreAlgebra"),
    "is_D": ("is_D", "OreAlgebra"),
    "is_Delta": ("is_Delta", "OreAlgebra"),
    "is_E": ("is_E", "OreAlgebra"),
    "is_F": ("is_F", "OreAlgebra"),
    "is_J": ("is_J", "OreAlgebra"),
    "is_Q": ("is_Q", "OreAlgebra"),
    "is_S": ("is_S", "OreAlgebra"),
    "OreAlgebra": ("OreAlgebra",),
}
_B1_METHOD_HINTS = {
    "gcrd": ("UnivariateOreOperator.gcrd", "gcrd"),
    "lclm": ("UnivariateOreOperator.lclm", "lclm"),
    "gcrd_lclm": (
        "UnivariateOreOperator.gcrd",
        "gcrd",
        "UnivariateOreOperator.lclm",
        "lclm",
    ),
}


@dataclass(frozen=True)
class ParsedRequest:
    raw_question: str
    normalized_question: str
    operator_text: str = ""
    operator_kind: str = ""
    algebra_scope: str = ""
    algebra_feature: str = ""
    common_operator_task: str = ""
    point_source: str = ""
    point_text: str = ""
    point_selection: str = ""
    point_kind: str = ""
    solution_kind: str = ""
    precision: int | None = None
    wants_algebra_setup: bool = False
    needs_basis: bool = False
    needs_base_ring: bool = False
    needs_code: bool = True
    wants_series: bool = False
    wants_generalized: bool = False
    wants_first_element_only: bool = False
    wants_existence_check: bool = False
    closure_variant: str = ""


@dataclass(frozen=True)
class IntentDecision:
    family_id: str = ""
    workflow_id: str = ""
    intent_id: str = ""
    confidence: float = 0.0
    preferred_methods: tuple[str, ...] = ()
    helper_actions: tuple[str, ...] = ()
    matched_on: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskUnderstanding:
    parsed_request: ParsedRequest
    intent: IntentDecision

    @property
    def is_actionable(self) -> bool:
        return bool(self.intent.workflow_id)

    def workflow_selection_override(self) -> WorkflowSelection:
        if not self.intent.workflow_id or not self.intent.family_id:
            return WorkflowSelection()

        registry = load_workflow_registry()
        workflow = next(
            (item for item in registry.workflows if item.id == self.intent.workflow_id),
            None,
        )
        family = get_family(self.intent.family_id)
        if workflow is None or family is None:
            return WorkflowSelection()

        preferred: list[str] = []
        for method in self.intent.preferred_methods:
            if method and method not in preferred:
                preferred.append(method)
        for symbol in workflow.preferred_symbols:
            if symbol and symbol not in preferred:
                preferred.append(symbol)

        prompt_rules: list[str] = list(workflow.prompt_rules)
        if self.parsed_request.needs_base_ring:
            prompt_rules.append(
                "The user explicitly asked to show the base ring, so generated code must print it."
            )
        if self.parsed_request.point_source == "leading_coefficient_root":
            prompt_rules.append(
                "Resolve the requested root of the leading coefficient before calling the local analysis method."
            )
        if self.parsed_request.wants_first_element_only:
            prompt_rules.append(
                "Only print the first requested element rather than the full returned collection."
            )

        return WorkflowSelection(
            family_id=family.id,
            family_name=family.name,
            workflow_id=workflow.id,
            workflow_title=workflow.title,
            confidence=self.intent.confidence,
            preferred_symbols=tuple(preferred),
            prompt_rules=tuple(prompt_rules),
            tuning_params=workflow.tuning_params,
            matched_on=self.intent.matched_on,
        )

    def task_workflow_hint(self) -> str:
        if not self.is_actionable:
            return ""

        lines = [
            "Structured task understanding:",
            f"- intent: {self.intent.intent_id}",
            f"- workflow_override: {self.intent.workflow_id}",
        ]

        if self.parsed_request.operator_kind:
            lines.append(f"- operator_kind: {self.parsed_request.operator_kind}")
        if self.parsed_request.algebra_scope:
            lines.append(f"- algebra_scope: {self.parsed_request.algebra_scope}")
        if self.parsed_request.algebra_feature:
            lines.append(f"- algebra_feature: {self.parsed_request.algebra_feature}")
        if self.parsed_request.common_operator_task:
            lines.append(f"- common_operator_task: {self.parsed_request.common_operator_task}")
        if self.parsed_request.solution_kind:
            lines.append(f"- solution_kind: {self.parsed_request.solution_kind}")
        if self.parsed_request.point_source:
            lines.append(f"- point_source: {self.parsed_request.point_source}")
        if self.parsed_request.point_selection:
            lines.append(f"- point_selection: {self.parsed_request.point_selection}")
        if self.parsed_request.point_kind:
            lines.append(f"- point_kind: {self.parsed_request.point_kind}")
        if self.parsed_request.point_text:
            lines.append(f"- point_text: {self.parsed_request.point_text}")
        if self.parsed_request.precision is not None:
            lines.append(f"- precision_hint: {self.parsed_request.precision}")
        if self.parsed_request.needs_base_ring:
            lines.append("- must_show_base_ring: true")
        if self.parsed_request.wants_first_element_only:
            lines.append("- output_scope: first_element_only")
        if self.parsed_request.wants_existence_check:
            lines.append("- answer_scope: existence_check")
        if self.intent.preferred_methods:
            lines.append(f"- preferred_methods: {', '.join(self.intent.preferred_methods)}")
        if self.intent.helper_actions:
            lines.append("- helper_actions:")
            lines.extend(f"  - {item}" for item in self.intent.helper_actions)
        return "\n".join(lines)

    def debug_lines(self) -> list[str]:
        lines = []
        if self.intent.family_id:
            lines.append(
                f"family={self.intent.family_id} intent={self.intent.intent_id or '?'} "
                f"workflow={self.intent.workflow_id or '?'} conf={self.intent.confidence:.2f}"
            )
        else:
            lines.append("family=? intent=? workflow=?")
        if self.parsed_request.operator_kind:
            lines.append(f"operator_kind={self.parsed_request.operator_kind}")
        if self.parsed_request.algebra_scope:
            lines.append(f"algebra_scope={self.parsed_request.algebra_scope}")
        if self.parsed_request.algebra_feature:
            lines.append(f"algebra_feature={self.parsed_request.algebra_feature}")
        if self.parsed_request.common_operator_task:
            lines.append(f"common_operator_task={self.parsed_request.common_operator_task}")
        if self.parsed_request.solution_kind:
            lines.append(f"solution_kind={self.parsed_request.solution_kind}")
        if self.parsed_request.point_source:
            point_bits = [self.parsed_request.point_source]
            if self.parsed_request.point_selection:
                point_bits.append(self.parsed_request.point_selection)
            if self.parsed_request.point_text:
                point_bits.append(self.parsed_request.point_text)
            lines.append("point=" + " | ".join(point_bits))
        if self.parsed_request.precision is not None:
            lines.append(f"precision={self.parsed_request.precision}")
        if self.parsed_request.needs_base_ring:
            lines.append("needs_base_ring=true")
        if self.parsed_request.wants_existence_check:
            lines.append("wants_existence_check=true")
        if self.intent.preferred_methods:
            lines.append("preferred_methods=" + ", ".join(self.intent.preferred_methods))
        return lines


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _extract_operator_text(question: str) -> str:
    text = str(question or "").strip()
    if not text:
        return ""
    patterns = (
        r"\b(?:of|for)\s+(?P<expr>.+?)\s+(?:at|around|with|where|in|using|and show|but print|, and show)\b",
        r"\boperator\s+(?P<expr>.+?)\s+(?:at|around|with|where|in|using|and show|but print|, and show)\b",
        r"\b(?:equation|operator)\s+(?P<expr>.+)$",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        expr = str(match.group("expr") or "").strip(" .,:;")
        if expr.lower().startswith("algebra "):
            continue
        if (
            "[" in expr
            and "]" in expr
            and not any(ch in expr for ch in "+-*^/()")
            and not expr.strip().startswith(("Dx", "Dy", "Dt", "Dz", "Sn", "Sk", "Sx", "Qx", "Fx", "Tx", "Jx"))
        ):
            continue
        if any(token in expr for token in ("Dx", "Dy", "Dt", "Dz", "Sn", "Sk", "Sx", "Qx", "Fx", "Tx", "Jx")):
            return expr
    return ""


def _operator_kind(normalized: str) -> str:
    tokens = {token.lower() for token in _TOKEN_RE.findall(normalized)}
    has_q_recurrence = "q-shift" in normalized or "q_recurrence" in normalized or "qx" in tokens or "jx" in tokens
    has_recurrence = "recurrence" in normalized or "shift operator" in normalized or any(
        token in tokens for token in ("sn", "sk", "sx")
    )
    has_differential = "differential operator" in normalized or "differential equation" in normalized or any(
        token in tokens for token in ("dx", "dy", "dt", "dz", "du", "dv")
    )
    if "mixed generators" in normalized or (has_differential and (has_recurrence or has_q_recurrence)):
        return "mixed"
    if has_q_recurrence:
        return "q_recurrence"
    if has_recurrence:
        return "recurrence"
    if has_differential:
        return "differential"
    return ""


def _solution_kind(normalized: str) -> str:
    if "indicial polynomial" in normalized:
        return "indicial_polynomial"
    if "leading monomial" in normalized or "leading monomials" in normalized:
        return "local_basis_monomials"
    if "leading logarithmic monomial" in normalized or "leading logarithmic monomials" in normalized:
        return "local_basis_monomials"
    if "generalized series" in normalized:
        return "generalized_series"
    if "series solution" in normalized or "series solutions" in normalized or "local basis" in normalized:
        return "local_basis_expansions"
    if "groebner basis" in normalized or "gröbner basis" in normalized:
        return "groebner_basis"
    return ""


def _algebra_scope(normalized: str, question: str) -> str:
    if any(
        marker in normalized
        for marker in (
            "multivariate ore algebra",
            "several generators",
            "several variables",
            "mixed generators",
            "dx dy",
            "dx and dy",
            "du dv",
            "sn sk",
        )
    ):
        return "multivariate"
    if re.search(
        r"\[[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\]\s*\[[A-Za-z_][A-Za-z0-9_]*\s*,\s*[A-Za-z_][A-Za-z0-9_]*\]",
        question,
    ):
        return "multivariate"
    return "univariate"


def _algebra_feature(normalized: str) -> str:
    if "associated commutative algebra" in normalized or "associated_commutative_algebra" in normalized:
        return "associated_commutative_algebra"
    if "change_constant_ring" in normalized or "change constant ring" in normalized:
        return "change_constant_ring"
    if "change_ring" in normalized or "change ring" in normalized:
        return "change_ring"
    if (
        "delta callable" in normalized
        or " ore algebra delta" in normalized
        or ("delta" in normalized and "algebra" in normalized)
    ):
        return "delta"
    if (
        "sigma callable" in normalized
        or " ore algebra sigma" in normalized
        or ("sigma" in normalized and "algebra" in normalized)
    ):
        return "sigma"
    if "differentialoperators" in normalized or "differential operators" in normalized:
        return "DifferentialOperators"
    property_match = re.search(r"\b(is_(?:c|d|delta|e|f|j|q|s))\b", normalized)
    if property_match:
        return _ALGEBRA_PROPERTY_METHODS.get(property_match.group(1), property_match.group(1))
    if (
        "gens" in normalized
        or "generators of this algebra" in normalized
        or "generator names" in normalized
        or "what are the generators" in normalized
        or "list generators" in normalized
        or "generators of the" in normalized
    ):
        return "gens"
    if (
        "orealgebra" in normalized
        or "ore algebra" in normalized
        or "differential operator algebra" in normalized
        or "recurrence operator algebra" in normalized
        or "create algebra" in normalized
        or "construct algebra" in normalized
        or "which algebra" in normalized
    ):
        return "OreAlgebra"
    return ""


def _common_operator_task(normalized: str) -> str:
    asks_gcrd = "gcrd" in normalized or "greatest common right divisor" in normalized or "common divisor" in normalized
    asks_lclm = "lclm" in normalized or "least common left multiple" in normalized or "common multiple" in normalized
    if asks_gcrd and asks_lclm:
        return "gcrd_lclm"
    if asks_gcrd:
        return "gcrd"
    if asks_lclm:
        return "lclm"
    return ""


def _wants_algebra_setup(normalized: str) -> bool:
    if _algebra_feature(normalized):
        return True
    return any(
        marker in normalized
        for marker in (
            "orealgebra",
            "ore algebra",
            "differential operator algebra",
            "recurrence operator algebra",
            "differentialoperators",
            "create algebra",
            "construct algebra",
            "which algebra",
            "multivariate ore algebra",
            "several generators",
            "several variables",
            "mixed generators",
        )
    )


def _closure_variant(normalized: str) -> str:
    if "annihilator_of_polynomial" in normalized:
        return "annihilator_of_polynomial"
    if "annihilating operator of the polynomial expression" in normalized:
        return "annihilator_of_polynomial"
    if "annihilating operator of p(" in normalized:
        return "annihilator_of_polynomial"
    if "annihilator_of_associate" in normalized or "association " in normalized:
        return "annihilator_of_associate"
    if "symmetric product" in normalized:
        return "symmetric_product"
    if "symmetric power" in normalized or "symmetric square" in normalized:
        return "symmetric_power"
    return ""


def _point_fields(normalized: str) -> tuple[str, str, str, str]:
    point_source = ""
    point_text = ""
    point_selection = ""
    point_kind = ""

    if "leading coefficient" in normalized and "root" in normalized:
        point_source = "leading_coefficient_root"
        point_kind = "algebraic"
    else:
        match = _POINT_AT_RE.search(normalized)
        if match:
            point_text = str(match.group("point") or "").strip()
            point_source = "explicit_point"
            if point_text in {"0", "x=0", "y=0", "t=0", "u=0", "z=0"}:
                point_kind = "zero"
            elif any(token in point_text for token in ("sqrt", "qqbar", "i", "root", "qatar(sort(")):
                point_kind = "algebraic"
            elif re.fullmatch(r"[-+]?\d+(?:/\d+)?", point_text):
                point_kind = "rational"

    if "second real root" in normalized:
        point_selection = "second_real"
        point_kind = point_kind or "algebraic"
    elif "first real root" in normalized:
        point_selection = "first_real"
        point_kind = point_kind or "algebraic"
    elif "one real root" in normalized or "a real root" in normalized:
        point_selection = "one_real"
        point_kind = point_kind or "algebraic"

    if point_source == "leading_coefficient_root" and not point_selection:
        point_selection = "unspecified_root"

    return point_source, point_text, point_selection, point_kind


def _extract_precision(normalized: str) -> int | None:
    for pattern in (_ORDER_RE, _BITS_RE, _FIRST_N_RE):
        match = pattern.search(normalized)
        if not match:
            continue
        try:
            return int(match.group("n"))
        except Exception:
            return None
    return None


def parse_request(question: str) -> ParsedRequest:
    normalized = _normalize_text(question)
    point_source, point_text, point_selection, point_kind = _point_fields(normalized)
    solution_kind = _solution_kind(normalized)
    closure_variant = _closure_variant(normalized)
    algebra_feature = _algebra_feature(normalized)
    common_operator_task = _common_operator_task(normalized)
    wants_algebra_setup = _wants_algebra_setup(normalized)
    algebra_scope = _algebra_scope(normalized, question) if wants_algebra_setup else ""
    wants_series = "series" in normalized or "frobenius" in normalized
    wants_generalized = "generalized series" in normalized or "generalised series" in normalized
    needs_basis = "basis" in normalized
    needs_base_ring = "base ring" in normalized
    wants_first_element_only = (
        "only the first element" in normalized
        or "print only the first element" in normalized
        or "show only the first element" in normalized
        or "first local basis element" in normalized
        or "1st local basis element" in normalized
    )
    wants_existence_check = bool(re.match(r"^(does|do|is|are|has|have)\b", normalized))

    return ParsedRequest(
        raw_question=str(question or ""),
        normalized_question=normalized,
        operator_text=_extract_operator_text(question),
        operator_kind=_operator_kind(normalized),
        algebra_scope=algebra_scope,
        algebra_feature=algebra_feature,
        common_operator_task=common_operator_task,
        point_source=point_source,
        point_text=point_text,
        point_selection=point_selection,
        point_kind=point_kind,
        solution_kind=solution_kind,
        precision=_extract_precision(normalized),
        wants_algebra_setup=wants_algebra_setup,
        needs_basis=needs_basis,
        needs_base_ring=needs_base_ring,
        needs_code=True,
        wants_series=wants_series,
        wants_generalized=wants_generalized,
        wants_first_element_only=wants_first_element_only,
        wants_existence_check=wants_existence_check,
        closure_variant=closure_variant,
    )


def classify_intent(parsed: ParsedRequest) -> IntentDecision:
    q = parsed.normalized_question
    matched_on: list[str] = []

    if parsed.closure_variant:
        matched_on.append(parsed.closure_variant)
        helper_actions: list[str] = []
        if parsed.wants_first_element_only:
            helper_actions.append("restrict output to the first requested result")
        return IntentDecision(
            family_id="C",
            workflow_id="C1_closure_combinatorics",
            intent_id=parsed.closure_variant,
            confidence=0.95,
            preferred_methods=(parsed.closure_variant,),
            helper_actions=tuple(helper_actions),
            matched_on=tuple(matched_on),
        )

    if any(token in q for token in ("annihilator_of_integral", "annihilator_of_composition", " to_s", " to_d", " to_f", " to_t", "convert to recurrence", "taylor-coefficient recurrence", "coefficient-side recurrence")):
        matched_on.append("conversion")
        return IntentDecision(
            family_id="C",
            workflow_id="C2_algebra_conversion",
            intent_id="algebra_conversion",
            confidence=0.9,
            preferred_methods=tuple(
                method
                for method in (
                    "annihilator_of_integral" if "annihilator_of_integral" in q else "",
                    "annihilator_of_composition" if "annihilator_of_composition" in q else "",
                    "to_S" if "to_s" in q or "convert to recurrence" in q else "",
                    "to_D" if "to_d" in q else "",
                    "to_F" if "to_f" in q else "",
                    "to_T" if "to_t" in q else "",
                )
                if method
            ),
            matched_on=tuple(matched_on),
        )

    if parsed.common_operator_task:
        matched_on.append(parsed.common_operator_task)
        return IntentDecision(
            family_id="B",
            workflow_id="B1_gcrd_lclm",
            intent_id=parsed.common_operator_task,
            confidence=0.94,
            preferred_methods=_B1_METHOD_HINTS.get(parsed.common_operator_task, ("gcrd", "lclm"))[
                :_METHOD_QUERY_LIMIT
            ],
            matched_on=tuple(matched_on),
        )

    if "right factor" in q or "right factors" in q:
        matched_on.append("right_factors")
        helper_actions: list[str] = []
        if "first-order" in q or "first order" in q:
            helper_actions.append("restrict attention to first-order right factors")
        if parsed.operator_kind == "q_recurrence":
            helper_actions.append("preserve q-shift algebra setup and avoid generic factorization APIs")
        if parsed.wants_existence_check:
            helper_actions.append("report whether any right factor satisfying the request exists")
        return IntentDecision(
            family_id="B",
            workflow_id="B2_factor_right_factor",
            intent_id="right_factor_search",
            confidence=0.93,
            preferred_methods=("right_factors",),
            helper_actions=tuple(helper_actions),
            matched_on=tuple(matched_on),
        )

    if parsed.solution_kind == "indicial_polynomial":
        matched_on.append("indicial_polynomial")
        return IntentDecision(
            family_id="B",
            workflow_id="B3_local_singularity_analysis",
            intent_id="indicial_polynomial",
            confidence=0.96,
            preferred_methods=("indicial_polynomial",),
            matched_on=tuple(matched_on),
        )

    if parsed.solution_kind == "local_basis_monomials":
        matched_on.append("local_basis_monomials")
        helper_actions = []
        if parsed.point_source == "leading_coefficient_root":
            helper_actions.extend(
                (
                    "extract the leading coefficient",
                    "compute and sort the relevant real roots",
                    "select the requested root before local analysis",
                )
            )
        return IntentDecision(
            family_id="B",
            workflow_id="B3_local_singularity_analysis",
            intent_id="local_basis_monomials",
            confidence=0.93,
            preferred_methods=("local_basis_monomials",),
            helper_actions=tuple(helper_actions),
            matched_on=tuple(matched_on),
        )

    if parsed.wants_series or parsed.solution_kind in {"generalized_series", "local_basis_expansions"}:
        matched_on.append("local_series")
        helper_actions = []
        preferred_methods: list[str] = []
        if parsed.wants_generalized and parsed.point_text in {"0", ""}:
            preferred_methods.append("generalized_series_solutions")
        preferred_methods.append("local_basis_expansions")
        if parsed.point_source == "leading_coefficient_root":
            helper_actions.extend(
                (
                    "extract the leading coefficient",
                    "compute real roots of the leading coefficient",
                    "sort real roots and select the requested one",
                )
            )
        if parsed.needs_base_ring:
            helper_actions.append("inspect the base ring of a returned series object")
        return IntentDecision(
            family_id="B",
            workflow_id="B3_local_singularity_analysis",
            intent_id="local_basis_expansions",
            confidence=0.88,
            preferred_methods=tuple(dict.fromkeys(preferred_methods))[:_METHOD_QUERY_LIMIT],
            helper_actions=tuple(helper_actions),
            matched_on=tuple(matched_on),
        )

    if any(token in q for token in ("groebner basis", "gröbner basis", "fglm", "left ideal", "annihilating ideal", "eliminate", "elimination", "intersection of the left ideal")):
        matched_on.append("ideal_groebner")
        preferred: list[str] = []
        if "groebner basis" in q or "gröbner basis" in q:
            preferred.append("OreLeftIdeal.groebner_basis")
        if "fglm" in q:
            preferred.append("fglm")
        if "eliminate" in q or "elimination" in q:
            preferred.append("OreLeftIdeal.eliminate")
        preferred.append("OreAlgebra_generic.ideal")
        return IntentDecision(
            family_id="B",
            workflow_id="B4_multivariate_ideal_groebner_basis",
            intent_id="groebner_basis",
            confidence=0.92,
            preferred_methods=tuple(dict.fromkeys(preferred))[:_METHOD_QUERY_LIMIT],
            matched_on=tuple(matched_on),
        )

    if parsed.wants_algebra_setup:
        if parsed.algebra_scope == "multivariate":
            matched_on.extend(filter(None, ("algebra_setup", parsed.algebra_scope, parsed.algebra_feature)))
            preferred = ["OreAlgebra", "OreAlgebra_generic.gens", "OreAlgebra_generic.ideal"]
            if parsed.algebra_feature == "gens":
                preferred.insert(0, "OreAlgebra_generic.gens")
            return IntentDecision(
                family_id="A",
                workflow_id="A2_create_or_choose_multivariate_algebra",
                intent_id="create_or_choose_multivariate_algebra",
                confidence=0.9,
                preferred_methods=tuple(dict.fromkeys(preferred))[:_METHOD_QUERY_LIMIT],
                matched_on=tuple(matched_on),
            )

        matched_on.extend(filter(None, ("algebra_setup", parsed.algebra_feature)))
        feature = parsed.algebra_feature or "OreAlgebra"
        preferred = list(_A_FEATURE_METHOD_HINTS.get(feature, ("OreAlgebra",)))
        return IntentDecision(
            family_id="A",
            workflow_id="A1_create_or_choose_algebra",
            intent_id="create_or_choose_algebra",
            confidence=0.88,
            preferred_methods=tuple(dict.fromkeys(preferred))[:_METHOD_QUERY_LIMIT],
            matched_on=tuple(matched_on),
        )

    return IntentDecision()


def analyze_question(question: str) -> TaskUnderstanding:
    parsed = parse_request(question)
    intent = classify_intent(parsed)
    return TaskUnderstanding(parsed_request=parsed, intent=intent)


def merge_task_workflow_hints(*blocks: str) -> str:
    parts = [str(block).strip() for block in blocks if str(block or "").strip()]
    return "\n\n".join(parts)


def summarize_task_understanding(question: str) -> list[str]:
    return analyze_question(question).debug_lines()


def contains_operator_markers(text: str) -> bool:
    tokens = {token.lower() for token in _TOKEN_RE.findall(text or "")}
    return any(token in tokens for token in {"dx", "dy", "dt", "dz", "sn", "sk", "sx", "qx", "fx", "tx", "jx"})
