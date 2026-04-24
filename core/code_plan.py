#!/usr/bin/env python3
"""Structured code-plan codegen path for ore_algebra.

The free-form ``generate_code_with_llm`` path asks the model for a
whole Sage script in one shot. That works most of the time but fails
on a recurring class of small mistakes (missing imports, missing
generator binding, missing ``print(...)`` of the requested quantity).

This module replaces that single shot with a two-stage process:

1. Ask the LLM for a *structured* JSON ``CodePlan``: imports, ring/
   algebra setup, body lines, and explicit print targets.
2. Run deterministic post-processing:
   - inject canonical ore_algebra imports for any referenced top-level
     name that is missing one,
   - guarantee that at least one ``print(...)`` line is present,
   - assemble the final script in a fixed order.

The output is a :class:`llm_service.CodeGenerationResponse` so this
path is a drop-in replacement for ``generate_code_with_llm`` from the
chat app's point of view.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import List, Sequence

from core.llm_service import (
    CodeGenerationRequest,
    CodeGenerationResponse,
    ContextItem,
    ORE_ALGEBRA_RULES,
    _call_llm,
    _coerce_string_list,
    _context_block,
    _loads_json_object,
)
from retrieval.precondition_graph import (
    PreconditionGraph,
    TOPLEVEL_IMPORTABLE,
    extract_referenced_names,
)


_PRINT_RE = re.compile(r"\bprint\s*\(")
_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")


@dataclass
class CodePlan:
    """Structured representation the LLM is asked to fill in."""

    imports: List[str] = field(default_factory=list)
    setup: List[str] = field(default_factory=list)
    body: List[str] = field(default_factory=list)
    prints: List[str] = field(default_factory=list)
    citations_used: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


def _strip_lines(values: Sequence[object]) -> List[str]:
    out: List[str] = []
    for value in values:
        text = str(value or "").rstrip()
        if not text.strip():
            continue
        out.append(text)
    return out


def parse_code_plan(raw_text: str, allowed_context_ids: Sequence[str]) -> CodePlan:
    """Parse a JSON code-plan response from the LLM."""

    payload = _loads_json_object(raw_text)
    allowed = set(allowed_context_ids)
    plan = CodePlan(
        imports=_strip_lines(_coerce_string_list(payload.get("imports"))),
        setup=_strip_lines(_coerce_string_list(payload.get("setup"))),
        body=_strip_lines(_coerce_string_list(payload.get("body"))),
        prints=_strip_lines(_coerce_string_list(payload.get("prints"))),
        citations_used=[
            cid
            for cid in _coerce_string_list(payload.get("citations_used"))
            if cid in allowed
        ],
        missing_info=_coerce_string_list(payload.get("missing_info")),
        notes=_coerce_string_list(payload.get("notes")),
    )
    return plan


def assemble_code(plan: CodePlan) -> str:
    """Concatenate plan sections into a runnable Sage script."""

    parts: List[str] = []
    if plan.imports:
        parts.extend(plan.imports)
    if plan.setup:
        if parts:
            parts.append("")
        parts.extend(plan.setup)
    if plan.body:
        if parts:
            parts.append("")
        parts.extend(plan.body)
    if plan.prints:
        if parts:
            parts.append("")
        parts.extend(plan.prints)
    if not parts:
        return ""
    return "\n".join(parts) + "\n"


def _existing_imported_names(plan: CodePlan) -> set[str]:
    names: set[str] = set()
    for line in plan.imports:
        for chunk in re.split(r"[,\s]+", line):
            chunk = chunk.strip().strip(",")
            if chunk and chunk not in {"from", "import", "as"}:
                names.add(chunk)
    return names


def _last_assignment_target(plan: CodePlan) -> str:
    for line in reversed(plan.body):
        match = _ASSIGN_RE.match(line)
        if match:
            return match.group(1)
    return ""


def validate_and_fix_plan(
    plan: CodePlan,
    graph: PreconditionGraph | None,
) -> CodePlan:
    """Apply deterministic repairs to a parsed code plan.

    Repairs performed (each one appends a note to ``plan.notes``):

    - inject canonical top-level imports for any name referenced in
      ``setup``/``body``/``prints`` that has a known import in the
      precondition graph and is not already imported,
    - guarantee at least one ``print(...)`` line is present,
    - drop empty sections.
    """

    referenced_text = "\n".join((*plan.setup, *plan.body, *plan.prints))
    referenced = extract_referenced_names(referenced_text)
    already = _existing_imported_names(plan)

    added_imports: List[str] = []
    seen_added: set[str] = set()
    for name in referenced:
        stmt = TOPLEVEL_IMPORTABLE.get(name)
        if not stmt or stmt in seen_added:
            continue
        if name in already:
            continue
        # Avoid duplicating an existing line that already imports it.
        if any(stmt == existing.strip() for existing in plan.imports):
            continue
        added_imports.append(stmt)
        seen_added.add(stmt)

    if added_imports:
        plan.imports = added_imports + list(plan.imports)
        plan.notes.append(
            "auto-added imports: " + ", ".join(added_imports)
        )

    # Guarantee at least one explicit print(...) so the runtime captures
    # something in stdout.
    has_print = any(_PRINT_RE.search(line) for line in (*plan.body, *plan.prints))
    if not has_print:
        target = _last_assignment_target(plan)
        if target:
            plan.prints.append(f"print({target})")
            plan.notes.append(
                f"auto-added print({target}) to surface the final result"
            )
        elif plan.body:
            # Wrap the last body expression as a print as a last resort.
            last = plan.body[-1]
            plan.body = plan.body[:-1]
            plan.prints.append(f"print({last})")
            plan.notes.append(
                "auto-wrapped final body line in print(...) to surface the result"
            )

    # Surface the auto-fix notes via missing_info so the chat UI shows
    # them next to the generated code, but keep them deduped.
    for note in plan.notes:
        if note not in plan.missing_info:
            plan.missing_info.append(note)

    return plan


def code_plan_to_response(
    plan: CodePlan,
    *,
    raw_response: str,
) -> CodeGenerationResponse:
    return CodeGenerationResponse(
        code=assemble_code(plan),
        citations_used=list(plan.citations_used),
        missing_info=list(plan.missing_info),
        raw_response=raw_response,
    )


def build_code_plan_prompt(
    request: CodeGenerationRequest,
    graph: PreconditionGraph | None = None,
) -> str:
    """Prompt the LLM to fill the structured CodePlan schema."""

    schema = {
        "imports": ["from ore_algebra import OreAlgebra"],
        "setup": [
            "R.<x> = QQ['x']",
            "A.<Dx> = OreAlgebra(R)",
        ],
        "body": [
            "L = (Dx + 1) * (Dx + 2)",
            "factors = L.right_factors()",
        ],
        "prints": ["print(factors)"],
        "citations_used": ["ctx_1"],
        "missing_info": ["string"],
    }
    workflow_hint = ""
    if request.task_workflow_hint.strip():
        workflow_hint = f"\n{request.task_workflow_hint.strip()}\n"
    return f"""You generate SageMath code for ore_algebra by filling a structured plan.

Return JSON only with this exact schema. Do NOT return a free-form script.

Section meanings:
- imports: explicit ore_algebra import lines you need (e.g. "from ore_algebra import OreAlgebra"). Do NOT include `from sage.all import *`; the runtime adds it.
- setup: ring/base-variable/algebra/generator construction lines, in order. Every named object used in `body` or `prints` must be created here unless it is a Sage builtin (QQ, ZZ, RR, ...).
- body: the main computation lines (assignments and intermediate calls). One statement per list entry. No print(...) here.
- prints: explicit print(...) lines. Always include at least one print of the final answer.
- citations_used: only context IDs from the provided context.
- missing_info: list anything important that the context did not provide.

Hard rules:
- Use only APIs and syntax supported by the provided context.
- Cite only the provided context IDs.
- Do not include markdown fences.
- Do not put print(...) inside `body`; put them in `prints`.
- Sage shorthand (R.<x> = QQ['x'], A.<Dx> = OreAlgebra(R)) is allowed; the runtime preparses before execution.
- If converting between operator types, build the source algebra in `setup` and the target algebra in `setup` too.
- If the question asks for a multi-step result (convert-then-compute, guess-then-continue), the plan must complete the full chain rather than stopping after the intermediate.

{ORE_ALGEBRA_RULES}
{workflow_hint}

JSON schema (shape only, fill with real values):
{json.dumps(schema, ensure_ascii=True)}

Question:
{request.question}

Context:
{_context_block(request.contexts)}
"""


def generate_code_with_plan(
    request: CodeGenerationRequest,
    *,
    graph: PreconditionGraph | None,
    api_key: str | None = None,
) -> CodeGenerationResponse:
    """Run the structured code-plan codegen path.

    On any LLM/parse failure the caller should fall back to the
    free-form ``generate_code_with_llm`` path.
    """

    prompt = build_code_plan_prompt(request, graph=graph)
    raw = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        api_key=api_key,
        base_url=request.base_url,
        max_output_tokens=request.max_output_tokens,
    )
    allowed_ids = [c.context_id for c in request.contexts]
    plan = parse_code_plan(raw, allowed_context_ids=allowed_ids)
    plan = validate_and_fix_plan(plan, graph)
    response = code_plan_to_response(plan, raw_response=raw)
    return response
