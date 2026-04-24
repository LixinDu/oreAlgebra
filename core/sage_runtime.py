#!/usr/bin/env python3
"""Hardened SageMath execution helpers for ore_algebra code."""

from __future__ import annotations

import ast
import json
import os
import re
import select
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4


OUTPUT_SUMMARY_MAX_CHARS = 2000
OUTPUT_TRUNCATION_HINT = (
    "[Output Truncated] ... Total length: {length} chars. "
    "Suggest querying specific properties like .degree() or .coefficient()."
)
BLOCKED_IMPORT_ROOTS = {"os", "subprocess", "shutil", "socket", "requests", "urllib"}
BLOCKED_CALLS = {"eval", "exec", "__import__"}
ORE_ALGEBRA_MARKERS = ("OreAlgebra", "ore_algebra")
BLOCKING_VALIDATION_PREFIXES = (
    "Blocked import detected:",
    "Blocked builtin call detected:",
    "Code is empty.",
)
UNBOUND_BASE_VAR_PREFIX = "Unbound base-ring variable issue:"
SCRIPT_LOG_DIR = Path(__file__).resolve().parents[1] / "scripts_log"
AUTO_IMPORTS = "from sage.all import *\nfrom ore_algebra import *\n"
WARM_DEFAULT_IDLE_SECONDS = 15 * 60
WARM_DEFAULT_STARTUP_TIMEOUT = 20
WARM_DEFAULT_MAX_RUNS = 25
WARM_REQUEST_MARGIN_SECONDS = 2
SEMANTIC_COMPARE_DEFAULT_TIMEOUT = 10

SEMANTIC_COMPARE_WORKER_CODE = r"""
import json
import keyword
import re
import sys

from sage.all import SR, var

IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SAFE_EXPR_RE = re.compile(r"^[0-9A-Za-z_\s\[\]\(\),.+\-*/^]*$")
RESERVED_NAMES = {
    "True",
    "False",
    "None",
    "pi",
    "e",
    "I",
    "oo",
    "Infinity",
    "log",
    "ln",
    "exp",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "abs",
}


def _split_top_level_csv(text):
    items = []
    current = []
    depth = 0
    for ch in str(text or ""):
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            token = "".join(current).strip()
            if token:
                items.append(token)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        items.append(tail)
    return items


def _is_list_like(text):
    token = str(text or "").strip()
    return token.startswith("[") and token.endswith("]")


def _declare_symbols(text):
    for name in sorted(set(IDENT_RE.findall(text))):
        if keyword.iskeyword(name) or name in RESERVED_NAMES:
            continue
        try:
            var(name)
        except Exception:
            continue


def _parse_scalar(token):
    atom = str(token or "").strip()
    if not atom:
        raise ValueError("empty atom")
    _declare_symbols(atom)
    return SR(atom)


def _parse_value(text):
    token = str(text or "").strip()
    if not token:
        raise ValueError("empty expression")
    if len(token) > 20000:
        raise ValueError("expression too large")
    if not SAFE_EXPR_RE.match(token):
        raise ValueError("unsupported characters in expression")
    if _is_list_like(token):
        return [_parse_scalar(item) for item in _split_top_level_csv(token[1:-1])]
    return _parse_scalar(token)


def _is_zero_like(expr):
    try:
        if bool(expr == 0):
            return True
    except Exception:
        pass
    for meth in ("simplify_full", "expand"):
        try:
            reduced = getattr(expr, meth)()
            if bool(reduced == 0):
                return True
        except Exception:
            continue
    try:
        return bool(expr.is_zero())
    except Exception:
        return False


def _equal_scalars(left, right):
    try:
        if bool(left == right):
            return True
    except Exception:
        pass
    try:
        return _is_zero_like(left - right)
    except Exception:
        return False


def _equal_values(left, right):
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        return all(_equal_values(lv, rv) for lv, rv in zip(left, right))
    if isinstance(left, list) or isinstance(right, list):
        return False
    return _equal_scalars(left, right)


def main():
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except Exception as exc:
        print(json.dumps({"equal": False, "reason": f"invalid_json: {exc}"}))
        return

    observed = str(payload.get("observed", ""))
    expected = str(payload.get("expected", ""))
    try:
        left = _parse_value(observed)
        right = _parse_value(expected)
    except Exception as exc:
        print(json.dumps({"equal": False, "reason": f"parse_error: {exc}"}))
        return

    equal = _equal_values(left, right)
    reason = "semantic_equal" if equal else "semantic_not_equal"
    print(json.dumps({"equal": bool(equal), "reason": reason}))


if __name__ == "__main__":
    main()
"""

WARM_WORKER_CODE = r"""
import contextlib
import io
import json
import sys
import traceback

from sage.all import preparse

BASE_SCOPE = {}
exec("from sage.all import *\nfrom ore_algebra import *", BASE_SCOPE, BASE_SCOPE)


def _emit(payload):
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + "\n")
    sys.stdout.flush()


while True:
    line = sys.stdin.readline()
    if not line:
        break
    try:
        req = json.loads(line)
    except Exception as exc:
        _emit({"ok": False, "error": f"invalid request: {exc}"})
        continue

    cmd = str(req.get("cmd", "")).strip().lower()
    if cmd == "ping":
        _emit({"ok": True, "status": "ready"})
        continue
    if cmd == "shutdown":
        _emit({"ok": True, "status": "bye"})
        break
    if cmd == "preparse":
        code = str(req.get("code", ""))
        try:
            _emit({"ok": True, "preparsed": preparse(code)})
        except Exception:
            err_buf = io.StringIO()
            traceback.print_exc(file=err_buf)
            _emit({"ok": False, "error": err_buf.getvalue()})
        continue
    if cmd != "exec":
        _emit({"ok": False, "error": f"unsupported cmd: {cmd}"})
        continue

    code = str(req.get("code", ""))
    out_buf = io.StringIO()
    err_buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
            prepared = preparse(code)
            run_scope = dict(BASE_SCOPE)
            exec(prepared, run_scope, run_scope)
        _emit({"ok": True, "stdout": out_buf.getvalue(), "stderr": err_buf.getvalue()})
    except Exception:
        traceback.print_exc(file=err_buf)
        _emit(
            {
                "ok": False,
                "stdout": out_buf.getvalue(),
                "stderr": err_buf.getvalue(),
                "error": "execution failed",
            }
        )
"""


@dataclass
class SageExecutionResult:
    status: Literal["success", "error", "blocked", "timeout"]
    preflight_ok: bool
    stdout_full: str
    stdout_summary: str
    is_truncated: bool
    stderr: str
    returncode: int
    validation_errors: list[str] = field(default_factory=list)
    executed_code: str = ""


def _effective_sage_bin(sage_bin: str) -> str:
    explicit = (sage_bin or "").strip()
    env_override = os.getenv("SAGE_BIN", "").strip()
    if explicit and explicit != "sage":
        return explicit
    if env_override:
        return env_override
    return explicit or "sage"


def _decode_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _blocked_import_name(name: str) -> str | None:
    root = (name or "").split(".", 1)[0]
    if root in BLOCKED_IMPORT_ROOTS:
        return root
    return None


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    return None


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


def _two_char_generator_parameter(name: str) -> str | None:
    token = (name or "").strip()
    if len(token) != 2:
        return None
    if token[0] not in {"S", "T", "D", "F", "Q", "J", "C"}:
        return None
    if not re.fullmatch(r"[a-z_]", token[1]):
        return None
    return token[1]


def _collect_ring_bindings(code: str) -> tuple[dict[str, str], set[str]]:
    ring_by_symbol: dict[str, str] = {}
    base_vars: set[str] = set()

    # Accept ring builders with chained calls like P.fraction_field()['y']
    # and empty-generator notation like Frac(R)[].
    shorthand_ring_re = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*="
        r"\s*[^\n]*?\[\s*(?:['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*)?\]"
    )
    plain_ring_re = re.compile(
        r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^\n]*?\[\s*(?:['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*)?\]"
    )

    for match in shorthand_ring_re.finditer(code):
        symbol = match.group(1)
        var_decl = match.group(2)
        var_literal = match.group(3) or ""
        ring_by_symbol[symbol] = var_decl
        base_vars.add(var_decl)
        if var_literal:
            base_vars.add(var_literal)

    for match in plain_ring_re.finditer(code):
        symbol = match.group(1)
        var_literal = match.group(2) or ""
        if var_literal:
            ring_by_symbol[symbol] = var_literal
            base_vars.add(var_literal)

    return ring_by_symbol, base_vars


def _resolve_ring_base_var(ring_expr: str, ring_by_symbol: dict[str, str]) -> str | None:
    expr = (ring_expr or "").strip()
    if not expr:
        return None
    if expr in ring_by_symbol:
        return ring_by_symbol[expr]
    literal = re.search(r"\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]", expr)
    if literal:
        return literal.group(1)
    return None


def _validate_generator_base_variable_alignment(code: str) -> list[str]:
    if "OreAlgebra" not in code:
        return []

    ring_by_symbol, base_vars = _collect_ring_bindings(code)
    errors: list[str] = []
    seen: set[str] = set()

    def _append_alignment_error(generator: str, param: str, base_var: str | None) -> None:
        if base_var:
            msg = (
                "Generator/base-variable mismatch: "
                f"`{generator}` uses parameter `{param}`, but OreAlgebra base variable is `{base_var}`. "
                "Use matching pairs like `Sx`/`Tx`/`Dx` with `x`, or `Sn`/`Tn`/`Dn` with `n`."
            )
        else:
            msg = (
                "Generator/base-variable mismatch: "
                f"`{generator}` uses parameter `{param}`, but no matching base variable `{param}` "
                "was found in ring definitions."
            )
        if msg in seen:
            return
        seen.add(msg)
        errors.append(msg)

    ore_call_with_gen_re = re.compile(
        r"OreAlgebra\(\s*([^,\n\)]+)\s*,\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
    )
    ore_shorthand_re = re.compile(
        r"\b[A-Za-z_][A-Za-z0-9_]*\s*\.<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*="
        r"\s*OreAlgebra\(\s*([^\n\)]*)\)"
    )

    for match in ore_call_with_gen_re.finditer(code):
        ring_expr = match.group(1)
        generator = match.group(2)
        param = _two_char_generator_parameter(generator)
        if not param:
            continue
        base_var = _resolve_ring_base_var(ring_expr, ring_by_symbol)
        if base_var is not None and param != base_var:
            _append_alignment_error(generator, param, base_var)
        elif base_var is None and base_vars and param not in base_vars:
            _append_alignment_error(generator, param, None)

    for match in ore_shorthand_re.finditer(code):
        generator = match.group(1)
        param = _two_char_generator_parameter(generator)
        if not param:
            continue
        args = match.group(2)
        ring_expr = args.split(",", 1)[0].strip()
        base_var = _resolve_ring_base_var(ring_expr, ring_by_symbol)
        if base_var is not None and param != base_var:
            _append_alignment_error(generator, param, base_var)
        elif base_var is None and base_vars and param not in base_vars:
            _append_alignment_error(generator, param, None)

    return errors


def _looks_like_generator_attribute(name: str) -> bool:
    token = (name or "").strip()
    if not token:
        return False
    if not re.fullmatch(r"[A-Z][A-Za-z0-9_]*", token):
        return False
    return True


def _collect_ore_algebra_bindings(tree: ast.AST) -> set[str]:
    bindings: set[str] = set()
    for node in ast.walk(tree):
        value: ast.AST | None = None
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            value = node.value
            targets = list(node.targets)
        elif isinstance(node, ast.AnnAssign):
            value = node.value
            targets = [node.target]
        if not isinstance(value, ast.Call):
            continue
        call_name = _dotted_name(value.func)
        if not call_name or call_name.split(".")[-1] != "OreAlgebra":
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                bindings.add(target.id)
    return bindings


def _collect_assigned_names_from_target(target: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(target, ast.Name):
        names.add(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            names.update(_collect_assigned_names_from_target(elt))
    return names


def _collect_bound_names(tree: ast.AST) -> set[str]:
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bound.update(_collect_assigned_names_from_target(target))
        elif isinstance(node, ast.AnnAssign):
            bound.update(_collect_assigned_names_from_target(node.target))
        elif isinstance(node, ast.AugAssign):
            bound.update(_collect_assigned_names_from_target(node.target))
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            bound.update(_collect_assigned_names_from_target(node.target))
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    bound.update(_collect_assigned_names_from_target(item.optional_vars))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bound.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                bound.add(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return bound


def _validate_generator_attribute_access(tree: ast.AST) -> list[str]:
    algebra_bindings = _collect_ore_algebra_bindings(tree)
    if not algebra_bindings:
        return []

    errors: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        if not isinstance(node.value, ast.Name):
            continue
        if node.value.id not in algebra_bindings:
            continue
        if not _looks_like_generator_attribute(node.attr):
            continue
        line_hint = f" (line {node.lineno})" if getattr(node, "lineno", None) else ""
        message = (
            "Generator binding/access issue: "
            f"`{node.value.id}.{node.attr}` looks like generator access on an OreAlgebra object{line_hint}. "
            f"Do not access generators as attributes like `{node.value.id}.{node.attr}`; "
            f"bind the generator with `{node.attr} = {node.value.id}.gen()` or use Sage shorthand "
            f"`{node.value.id}.<{node.attr}> = OreAlgebra(...)`."
        )
        if message in seen:
            continue
        seen.add(message)
        errors.append(message)
    return errors


def _validate_unbound_generator_use(tree: ast.AST) -> list[str]:
    bound = _collect_bound_names(tree)
    errors: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        generator_param = _two_char_generator_parameter(node.id)
        if generator_param is None:
            continue
        if node.id in bound:
            continue
        line_hint = f" (line {node.lineno})" if getattr(node, "lineno", None) else ""
        message = (
            "Unbound generator issue: "
            f"`{node.id}` is used before being bound{line_hint}. "
            f"Bind it explicitly with `{node.id} = A.gen()`, use Sage shorthand "
            f"`A.<{node.id}> = OreAlgebra(...)`, or construct it via a helper like "
            f"`Dops, x, {node.id} = DifferentialOperators()`."
        )
        if message in seen:
            continue
        seen.add(message)
        errors.append(message)
    return errors


def _validate_unbound_base_ring_variable_use(tree: ast.AST, base_vars: set[str]) -> list[str]:
    candidates = {
        value
        for value in (base_vars or set())
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value or "")
    }
    if not candidates:
        return []

    bound = _collect_bound_names(tree)
    errors: list[str] = []
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Name):
            continue
        if not isinstance(node.ctx, ast.Load):
            continue
        if node.id not in candidates:
            continue
        if node.id in bound:
            continue
        line_hint = f" (line {node.lineno})" if getattr(node, "lineno", None) else ""
        message = (
            f"{UNBOUND_BASE_VAR_PREFIX} "
            f"`{node.id}` is used before being bound{line_hint}. "
            f"Bind it explicitly with `{node.id} = R.gen()` after a ring definition "
            f"like `R = ...['{node.id}']`, or use Sage shorthand "
            f"`R.<{node.id}> = ...`."
        )
        if message in seen:
            continue
        seen.add(message)
        errors.append(message)
    return errors


def _extract_unbound_base_ring_vars(validation_errors: list[str]) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    pattern = re.compile(rf"{re.escape(UNBOUND_BASE_VAR_PREFIX)}\s*`([A-Za-z_][A-Za-z0-9_]*)`")
    for item in validation_errors:
        match = pattern.search(str(item or ""))
        if not match:
            continue
        name = match.group(1)
        if name in seen:
            continue
        seen.add(name)
        names.append(name)
    return names


def _source_bound_names(code: str) -> set[str]:
    names: set[str] = set()
    assign_re = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    shorthand_re = re.compile(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*\.<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>\s*=")
    for raw in (code or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        match_assign = assign_re.match(line)
        if match_assign:
            names.add(match_assign.group(1))
        match_shorthand = shorthand_re.match(line)
        if match_shorthand:
            names.add(match_shorthand.group(1))
    return names


def _pick_fresh_name(preferred: str, bound_names: set[str]) -> str:
    base = preferred if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", preferred or "") else "R"
    if base not in bound_names:
        return base
    suffix = 2
    while f"{base}{suffix}" in bound_names:
        suffix += 1
    return f"{base}{suffix}"


def _apply_unbound_base_ring_var_autofix(
    code: str,
    validation_errors: list[str],
) -> tuple[str, list[str]]:
    missing_vars = _extract_unbound_base_ring_vars(validation_errors)
    if not missing_vars:
        return code, []

    lines = (code or "").splitlines()
    if not lines:
        return code, []

    notes: list[str] = []
    missing_set = set(missing_vars)
    bound_names = _source_bound_names(code)
    fixed_lines = list(lines)
    changed = False

    # Pattern 1: explicit ring variable assignment, e.g. R = ZZ['n'].
    ring_assign_re = re.compile(
        r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?!.*\bOreAlgebra\s*\().*"
        r"\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\]\s*$"
    )
    inserts: list[tuple[int, str]] = []
    for idx, raw in enumerate(fixed_lines):
        match = ring_assign_re.match(raw)
        if not match:
            continue
        indent, ring_name, base_var = match.groups()
        if base_var not in missing_set:
            continue
        if base_var in bound_names:
            continue
        line = f"{indent}{base_var} = {ring_name}.gen()"
        inserts.append((idx + 1, line))
        bound_names.add(base_var)
        notes.append(f"auto-fix: inserted `{base_var} = {ring_name}.gen()`")

    if inserts:
        offset = 0
        for index, line in inserts:
            fixed_lines.insert(index + offset, line)
            offset += 1
        changed = True

    unresolved = [name for name in missing_vars if name not in bound_names]
    if unresolved:
        # Pattern 2: inline OreAlgebra(ZZ['n'], 'Sn') assignment without a ring symbol.
        ore_inline_re = re.compile(
            r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*=\s*OreAlgebra\(\s*"
            r"((?:QQ|ZZ|RR|CC|QQbar)\s*\[\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]\s*\])"
            r"\s*,\s*(['\"][A-Za-z_][A-Za-z0-9_]*['\"])\s*\)\s*$"
        )
        for idx, raw in enumerate(fixed_lines):
            match = ore_inline_re.match(raw)
            if not match:
                continue
            indent, algebra_name, ring_expr, base_var, generator_literal = match.groups()
            if base_var not in unresolved:
                continue
            if base_var in bound_names:
                continue
            ring_symbol = _pick_fresh_name("R", bound_names)
            replacement = [
                f"{indent}{ring_symbol} = {ring_expr}",
                f"{indent}{base_var} = {ring_symbol}.gen()",
                f"{indent}{algebra_name} = OreAlgebra({ring_symbol}, {generator_literal})",
            ]
            fixed_lines[idx:idx + 1] = replacement
            bound_names.add(ring_symbol)
            bound_names.add(base_var)
            notes.append(
                "auto-fix: rewrote inline OreAlgebra ring to explicit ring binding "
                f"and inserted `{base_var} = {ring_symbol}.gen()`"
            )
            changed = True
            unresolved = [name for name in unresolved if name != base_var]
            if not unresolved:
                break

    if not changed:
        return code, []
    return "\n".join(fixed_lines).strip(), notes


def _validation_severity(errors: list[str]) -> tuple[int, int]:
    blocking = sum(1 for item in errors if _is_blocking_validation_error(item))
    return (blocking, len(errors))


def _summarize_stdout(stdout_text: str, max_chars: int = OUTPUT_SUMMARY_MAX_CHARS) -> tuple[str, bool]:
    if len(stdout_text) <= max_chars:
        return stdout_text, False

    suffix = OUTPUT_TRUNCATION_HINT.format(length=len(stdout_text))
    if len(suffix) >= max_chars:
        return suffix[:max_chars], True

    head = stdout_text[: max_chars - len(suffix)]
    return f"{head}{suffix}", True


def _result(
    *,
    status: Literal["success", "error", "blocked", "timeout"],
    preflight_ok: bool,
    stdout_full: str = "",
    stderr: str = "",
    returncode: int = 0,
    validation_errors: list[str] | None = None,
    executed_code: str = "",
) -> SageExecutionResult:
    stdout_summary, is_truncated = _summarize_stdout(stdout_full)
    return SageExecutionResult(
        status=status,
        preflight_ok=preflight_ok,
        stdout_full=stdout_full,
        stdout_summary=stdout_summary,
        is_truncated=is_truncated,
        stderr=stderr,
        returncode=returncode,
        validation_errors=list(validation_errors or []),
        executed_code=str(executed_code or "").strip(),
    )


def _script_path() -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return SCRIPT_LOG_DIR / f"sage_exec_{timestamp}_{uuid4().hex[:8]}.py"


def _preparse_with_sage(code: str, sage_bin: str) -> str:
    effective_bin = _effective_sage_bin(sage_bin)
    helper = (
        "import sys\n"
        "from sage.all import preparse\n"
        "sys.stdout.write(preparse(sys.stdin.read()))\n"
    )
    try:
        proc = subprocess.run(
            [effective_bin, "-python", "-c", helper],
            input=code.encode("utf-8"),
            capture_output=True,
            check=False,
            timeout=60,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Sage binary not found: {effective_bin}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"Sage preparser timed out after 60 seconds using {effective_bin}"
        ) from exc

    if proc.returncode != 0:
        stderr_text = _decode_output(proc.stderr).strip()
        detail = stderr_text or f"exit code {proc.returncode}"
        raise RuntimeError(f"Sage preparser failed: {detail}")

    return _decode_output(proc.stdout)


def _normalize_for_ore_algebra_runtime(code: str) -> str:
    stripped = code.strip()
    if not stripped:
        return ""
    if any(marker in stripped for marker in ORE_ALGEBRA_MARKERS):
        return stripped
    return f"from ore_algebra import *\n{stripped}"


class _WarmSageSessionManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc: subprocess.Popen[str] | None = None
        self._sage_bin = ""
        self._last_used_mono = 0.0
        self._run_count = 0
        self._idle_timer: threading.Timer | None = None
        self._idle_generation = 0

    def _schedule_idle_shutdown_locked(self, warm_ttl_seconds: int) -> None:
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if warm_ttl_seconds <= 0:
            return

        self._idle_generation += 1
        generation = self._idle_generation

        def _shutdown_if_still_idle() -> None:
            with self._lock:
                if generation != self._idle_generation:
                    return
                if self._proc is None:
                    return
                if (time.monotonic() - self._last_used_mono) < warm_ttl_seconds:
                    return
                self._close_locked()

        timer = threading.Timer(warm_ttl_seconds, _shutdown_if_still_idle)
        timer.daemon = True
        self._idle_timer = timer
        timer.start()

    def _close_locked(self) -> None:
        proc = self._proc
        self._proc = None
        self._sage_bin = ""
        self._last_used_mono = 0.0
        self._run_count = 0
        self._idle_generation += 1
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                try:
                    proc.stdin.write(json.dumps({"cmd": "shutdown"}, ensure_ascii=True) + "\n")
                    proc.stdin.flush()
                except Exception:
                    pass
        finally:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=1)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass

    def _readline_with_timeout_locked(self, timeout_seconds: float) -> str | None:
        if self._proc is None or self._proc.stdout is None:
            return None
        if timeout_seconds <= 0:
            timeout_seconds = 0.001
        fd = self._proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_seconds)
        if not ready:
            return None
        return self._proc.stdout.readline()

    def _request_locked(self, payload: dict, timeout_seconds: float) -> tuple[dict | None, str]:
        proc = self._proc
        if proc is None or proc.stdin is None:
            return None, "warm session is not running"
        try:
            proc.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
            proc.stdin.flush()
        except Exception as exc:
            return None, f"warm session write failed: {exc}"

        line = self._readline_with_timeout_locked(timeout_seconds=timeout_seconds)
        if line is None:
            return None, "warm session timed out waiting for response"
        if line == "":
            return None, "warm session terminated unexpectedly"
        try:
            response = json.loads(line)
        except json.JSONDecodeError as exc:
            return None, f"warm session returned invalid JSON: {exc.msg}"
        if not isinstance(response, dict):
            return None, "warm session returned a non-object payload"
        return response, ""

    def _ensure_warm_locked(
        self,
        *,
        sage_bin: str,
        warm_ttl_seconds: int,
        startup_timeout: int,
        max_runs_per_session: int,
    ) -> tuple[bool, str]:
        effective_bin = _effective_sage_bin(sage_bin)
        now = time.monotonic()

        if self._proc is not None:
            proc_dead = self._proc.poll() is not None
            expired = warm_ttl_seconds > 0 and (now - self._last_used_mono) > warm_ttl_seconds
            run_limit_hit = max_runs_per_session > 0 and self._run_count >= max_runs_per_session
            bin_changed = bool(self._sage_bin and self._sage_bin != effective_bin)
            if proc_dead or expired or run_limit_hit or bin_changed:
                self._close_locked()

        if self._proc is None:
            try:
                self._proc = subprocess.Popen(
                    [effective_bin, "-python", "-u", "-c", WARM_WORKER_CODE],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
            except FileNotFoundError:
                self._close_locked()
                return False, f"Sage binary not found: {effective_bin}"
            except Exception as exc:
                self._close_locked()
                return False, f"Failed to start warm Sage session: {exc}"

            self._sage_bin = effective_bin
            self._run_count = 0
            ping, err = self._request_locked({"cmd": "ping"}, timeout_seconds=max(startup_timeout, 1))
            if ping is None:
                self._close_locked()
                return False, err
            if not ping.get("ok", False):
                self._close_locked()
                return False, _safe_worker_error(ping, "Warm Sage ping failed.")

        self._last_used_mono = now
        self._schedule_idle_shutdown_locked(warm_ttl_seconds)
        return True, "warm_ready"

    def prewarm(
        self,
        *,
        sage_bin: str,
        warm_ttl_seconds: int,
        startup_timeout: int,
        max_runs_per_session: int,
    ) -> tuple[bool, str]:
        with self._lock:
            return self._ensure_warm_locked(
                sage_bin=sage_bin,
                warm_ttl_seconds=warm_ttl_seconds,
                startup_timeout=startup_timeout,
                max_runs_per_session=max_runs_per_session,
            )

    def execute(
        self,
        *,
        code: str,
        sage_bin: str,
        timeout: int,
        warm_ttl_seconds: int,
        startup_timeout: int,
        max_runs_per_session: int,
    ) -> tuple[SageExecutionResult | None, str]:
        with self._lock:
            ok, msg = self._ensure_warm_locked(
                sage_bin=sage_bin,
                warm_ttl_seconds=warm_ttl_seconds,
                startup_timeout=startup_timeout,
                max_runs_per_session=max_runs_per_session,
            )
            if not ok:
                return None, msg

            response, err = self._request_locked(
                {"cmd": "exec", "code": code},
                timeout_seconds=max(timeout + WARM_REQUEST_MARGIN_SECONDS, 1),
            )
            if response is None:
                self._close_locked()
                if "timed out" in err.lower():
                    timeout_message = f"Execution timed out after {timeout} seconds."
                    return (
                        _result(
                            status="timeout",
                            preflight_ok=True,
                            stderr=timeout_message,
                            returncode=-124,
                        ),
                        err,
                    )
                return None, err

            self._last_used_mono = time.monotonic()
            self._run_count += 1

            ok_flag = bool(response.get("ok"))
            stdout_text = _decode_output(response.get("stdout"))
            stderr_text = _decode_output(response.get("stderr"))
            if ok_flag:
                return (
                    _result(
                        status="success",
                        preflight_ok=True,
                        stdout_full=stdout_text,
                        stderr=stderr_text,
                        returncode=0,
                    ),
                    "warm_success",
                )
            return (
                _result(
                    status="error",
                    preflight_ok=True,
                    stdout_full=stdout_text,
                    stderr=stderr_text or _safe_worker_error(response, "Warm Sage execution failed."),
                    returncode=1,
                ),
                "warm_error",
            )

    def preparse(
        self,
        *,
        code: str,
        sage_bin: str,
        warm_ttl_seconds: int,
        startup_timeout: int,
        max_runs_per_session: int,
    ) -> tuple[str | None, str]:
        with self._lock:
            ok, msg = self._ensure_warm_locked(
                sage_bin=sage_bin,
                warm_ttl_seconds=warm_ttl_seconds,
                startup_timeout=startup_timeout,
                max_runs_per_session=max_runs_per_session,
            )
            if not ok:
                return None, msg
            response, err = self._request_locked(
                {"cmd": "preparse", "code": code},
                timeout_seconds=max(startup_timeout, 1),
            )
            if response is None:
                self._close_locked()
                return None, err
            if not response.get("ok", False):
                return None, _safe_worker_error(response, "Warm Sage preparse failed.")
            self._last_used_mono = time.monotonic()
            preparsed = _decode_output(response.get("preparsed"))
            return preparsed, "warm_preparse_success"

    def close(self) -> None:
        with self._lock:
            self._close_locked()


def _safe_worker_error(payload: dict, fallback: str) -> str:
    message = _decode_output(payload.get("error")).strip()
    if message:
        return message
    return fallback


_WARM_SAGE_MANAGER = _WarmSageSessionManager()


def prewarm_sage_session(
    *,
    sage_bin: str = "sage",
    warm_ttl_seconds: int = WARM_DEFAULT_IDLE_SECONDS,
    startup_timeout: int = WARM_DEFAULT_STARTUP_TIMEOUT,
    max_runs_per_session: int = WARM_DEFAULT_MAX_RUNS,
) -> tuple[bool, str]:
    return _WARM_SAGE_MANAGER.prewarm(
        sage_bin=sage_bin,
        warm_ttl_seconds=warm_ttl_seconds,
        startup_timeout=startup_timeout,
        max_runs_per_session=max_runs_per_session,
    )


def prewarm_sage_session_async(
    *,
    sage_bin: str = "sage",
    warm_ttl_seconds: int = WARM_DEFAULT_IDLE_SECONDS,
    startup_timeout: int = WARM_DEFAULT_STARTUP_TIMEOUT,
    max_runs_per_session: int = WARM_DEFAULT_MAX_RUNS,
) -> None:
    def _runner() -> None:
        try:
            prewarm_sage_session(
                sage_bin=sage_bin,
                warm_ttl_seconds=warm_ttl_seconds,
                startup_timeout=startup_timeout,
                max_runs_per_session=max_runs_per_session,
            )
        except Exception:
            # Keep async warmup best-effort.
            return

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()


def shutdown_warm_sage_session() -> None:
    _WARM_SAGE_MANAGER.close()


def _execute_with_warm_session(
    *,
    code: str,
    sage_bin: str,
    timeout: int,
    warm_ttl_seconds: int,
    startup_timeout: int,
    max_runs_per_session: int,
) -> tuple[SageExecutionResult | None, str]:
    return _WARM_SAGE_MANAGER.execute(
        code=code,
        sage_bin=sage_bin,
        timeout=timeout,
        warm_ttl_seconds=warm_ttl_seconds,
        startup_timeout=startup_timeout,
        max_runs_per_session=max_runs_per_session,
    )


def _preparse_with_warm_session(
    *,
    code: str,
    sage_bin: str,
    warm_ttl_seconds: int,
    startup_timeout: int,
    max_runs_per_session: int,
) -> tuple[str | None, str]:
    return _WARM_SAGE_MANAGER.preparse(
        code=code,
        sage_bin=sage_bin,
        warm_ttl_seconds=warm_ttl_seconds,
        startup_timeout=startup_timeout,
        max_runs_per_session=max_runs_per_session,
    )


def _validate_preparsed_code(preparsed_code: str, *, base_vars: set[str] | None = None) -> list[str]:
    errors: list[str] = []
    try:
        tree = ast.parse(preparsed_code)
    except SyntaxError as exc:
        detail = exc.msg
        if exc.lineno is not None:
            detail = f"{detail} (line {exc.lineno})"
        _append_error(errors, f"Preparsed code is not valid Python: {detail}")
        return errors

    errors.extend(_validate_generator_attribute_access(tree))
    errors.extend(_validate_unbound_generator_use(tree))
    errors.extend(_validate_unbound_base_ring_variable_use(tree, base_vars or set()))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                blocked = _blocked_import_name(alias.name)
                if blocked is not None:
                    _append_error(errors, f"Blocked import detected: {blocked}")
        elif isinstance(node, ast.ImportFrom):
            blocked = _blocked_import_name(node.module or "")
            if blocked is not None:
                _append_error(errors, f"Blocked import detected: {blocked}")
        elif isinstance(node, ast.Call):
            call_name = _call_name(node.func)
            if call_name in BLOCKED_CALLS:
                _append_error(errors, f"Blocked builtin call detected: {call_name}()")

    return errors


def _is_blocking_validation_error(message: str) -> bool:
    text = (message or "").strip()
    if not text:
        return False
    return any(text.startswith(prefix) for prefix in BLOCKING_VALIDATION_PREFIXES)


def _merge_validation_messages(existing: list[str], extras: list[str]) -> list[str]:
    merged = list(existing)
    for item in extras:
        if item not in merged:
            merged.append(item)
    return merged


def compare_sage_outputs_semantically(
    observed: str,
    expected: str,
    *,
    sage_bin: str = "sage",
    timeout: int = SEMANTIC_COMPARE_DEFAULT_TIMEOUT,
) -> tuple[bool, str]:
    """Compare two output strings with Sage-aware symbolic equivalence.

    The comparator is best-effort: it returns ``(False, reason)`` if Sage is
    unavailable, parsing fails, or expressions are not equivalent.
    """
    observed_text = str(observed or "").strip()
    expected_text = str(expected or "").strip()
    if not observed_text or not expected_text:
        return False, "Semantic comparison skipped: observed/expected output is empty."

    effective_bin = _effective_sage_bin(sage_bin)
    payload = json.dumps(
        {"observed": observed_text, "expected": expected_text},
        ensure_ascii=False,
    ).encode("utf-8")

    try:
        proc = subprocess.run(
            [effective_bin, "-python", "-c", SEMANTIC_COMPARE_WORKER_CODE],
            input=payload,
            capture_output=True,
            check=False,
            timeout=max(1, int(timeout)),
        )
    except FileNotFoundError:
        return False, f"Sage binary not found: {effective_bin}"
    except subprocess.TimeoutExpired:
        return False, f"Sage semantic comparison timed out after {timeout} seconds."
    except Exception as exc:
        return False, f"Sage semantic comparison failed: {exc}"

    stdout_text = _decode_output(proc.stdout).strip()
    stderr_text = _decode_output(proc.stderr).strip()
    if proc.returncode != 0:
        detail = stderr_text or stdout_text or f"exit code {proc.returncode}"
        return False, f"Sage semantic comparison process failed: {detail}"
    if not stdout_text:
        return False, "Sage semantic comparison returned no output."

    line = stdout_text.splitlines()[-1]
    try:
        parsed = json.loads(line)
    except json.JSONDecodeError:
        detail = line[:200]
        return False, f"Sage semantic comparison returned non-JSON output: {detail}"

    equal = bool(parsed.get("equal"))
    reason = str(parsed.get("reason") or "").strip()
    if not reason:
        reason = "Outputs are semantically equivalent." if equal else "Outputs are not semantically equivalent."
    return equal, reason


def validate_generated_code(
    code: str,
    sage_bin: str = "sage",
    preparsed_override: str | None = None,
) -> list[str]:
    errors: list[str] = []
    stripped = code.strip()
    if not stripped:
        return ["Code is empty."]

    if not any(marker in code for marker in ORE_ALGEBRA_MARKERS):
        _append_error(
            errors,
            "Code must contain an ore_algebra marker (`OreAlgebra` or `ore_algebra`).",
        )

    errors.extend(_validate_generator_base_variable_alignment(code))
    _, base_vars = _collect_ring_bindings(code)

    preparsed_code = preparsed_override
    if preparsed_code is None:
        try:
            preparsed_code = _preparse_with_sage(code, sage_bin=sage_bin)
        except Exception as exc:
            _append_error(errors, str(exc))
            return errors

    errors.extend(_validate_preparsed_code(preparsed_code, base_vars=base_vars))
    return errors


def run_sage_code(code: str, sage_bin: str = "sage", timeout: int = 60) -> SageExecutionResult:
    effective_bin = _effective_sage_bin(sage_bin)
    normalized_code = str(code or "").strip()
    final_script = f"{AUTO_IMPORTS}{normalized_code}\n"

    print("=== Sage code to execute ===")
    print(normalized_code.rstrip())

    SCRIPT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    script_path = _script_path()
    script_path.write_text(final_script, encoding="utf-8")

    should_delete_script = False
    try:
        preparsed_script = _preparse_with_sage(final_script, sage_bin=effective_bin)
        script_path.write_text(preparsed_script, encoding="utf-8")
        proc = subprocess.run(
            [effective_bin, "-python", str(script_path)],
            capture_output=True,
            check=False,
            timeout=timeout,
        )
        stdout_text = _decode_output(proc.stdout)
        stderr_text = _decode_output(proc.stderr)
        status: Literal["success", "error"] = "success" if proc.returncode == 0 else "error"
        if status == "success":
            should_delete_script = True
        else:
            print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status=status,
            preflight_ok=True,
            stdout_full=stdout_text,
            stderr=stderr_text,
            returncode=proc.returncode,
            executed_code=normalized_code,
        )
    except subprocess.TimeoutExpired as exc:
        stdout_text = _decode_output(exc.stdout)
        stderr_text = _decode_output(exc.stderr)
        timeout_message = f"Execution timed out after {timeout} seconds."
        if stderr_text:
            stderr_text = f"{stderr_text.rstrip()}\n{timeout_message}"
        else:
            stderr_text = timeout_message
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="timeout",
            preflight_ok=True,
            stdout_full=stdout_text,
            stderr=stderr_text,
            returncode=-124,
            executed_code=normalized_code,
        )
    except FileNotFoundError:
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="error",
            preflight_ok=True,
            stderr=f"Sage binary not found: {effective_bin}",
            returncode=127,
            executed_code=normalized_code,
        )
    except Exception as exc:
        print(f"Sage script kept for debugging: {script_path}")
        return _result(
            status="error",
            preflight_ok=True,
            stderr=str(exc),
            returncode=1,
            executed_code=normalized_code,
        )
    finally:
        if should_delete_script:
            try:
                script_path.unlink()
            except OSError:
                pass


def validate_and_run_sage(
    code: str,
    sage_bin: str = "sage",
    timeout: int = 60,
    use_warm_session: bool = False,
    warm_ttl_seconds: int = WARM_DEFAULT_IDLE_SECONDS,
    warm_startup_timeout: int = WARM_DEFAULT_STARTUP_TIMEOUT,
    warm_max_runs_per_session: int = WARM_DEFAULT_MAX_RUNS,
) -> SageExecutionResult:
    normalized_code = _normalize_for_ore_algebra_runtime(code)
    preparsed_override: str | None = None
    if use_warm_session:
        preparsed_override, warm_preparse_note = _preparse_with_warm_session(
            code=normalized_code,
            sage_bin=sage_bin,
            warm_ttl_seconds=warm_ttl_seconds,
            startup_timeout=warm_startup_timeout,
            max_runs_per_session=warm_max_runs_per_session,
        )
        if preparsed_override is None:
            print(f"Warm Sage preparse unavailable, using cold validation: {warm_preparse_note}")

    validation_errors = validate_generated_code(
        normalized_code,
        sage_bin=sage_bin,
        preparsed_override=preparsed_override,
    )

    fixed_code, fix_notes = _apply_unbound_base_ring_var_autofix(
        normalized_code,
        validation_errors,
    )
    if fix_notes and fixed_code != normalized_code:
        fixed_preparsed_override: str | None = None
        if use_warm_session:
            fixed_preparsed_override, fixed_warm_preparse_note = _preparse_with_warm_session(
                code=fixed_code,
                sage_bin=sage_bin,
                warm_ttl_seconds=warm_ttl_seconds,
                startup_timeout=warm_startup_timeout,
                max_runs_per_session=warm_max_runs_per_session,
            )
            if fixed_preparsed_override is None:
                print(
                    "Warm Sage preparse unavailable for declaration auto-fix candidate, "
                    f"using cold validation: {fixed_warm_preparse_note}"
                )
        fixed_validation_errors = validate_generated_code(
            fixed_code,
            sage_bin=sage_bin,
            preparsed_override=fixed_preparsed_override,
        )
        original_severity = _validation_severity(validation_errors)
        fixed_severity = _validation_severity(fixed_validation_errors)
        original_unbound = len(_extract_unbound_base_ring_vars(validation_errors))
        fixed_unbound = len(_extract_unbound_base_ring_vars(fixed_validation_errors))
        should_keep_fix = fixed_severity < original_severity or (
            fixed_unbound < original_unbound and fixed_severity[0] <= original_severity[0]
        )
        if should_keep_fix:
            normalized_code = fixed_code
            preparsed_override = fixed_preparsed_override
            auto_fix_messages = [f"Auto-fix applied: {note}" for note in fix_notes]
            validation_errors = _merge_validation_messages(
                fixed_validation_errors,
                auto_fix_messages,
            )

    blocking_errors = [item for item in validation_errors if _is_blocking_validation_error(item)]
    advisory_errors = [item for item in validation_errors if not _is_blocking_validation_error(item)]

    if blocking_errors:
        return _result(
            status="blocked",
            preflight_ok=False,
            stderr="Code failed validation.",
            returncode=-1,
            validation_errors=validation_errors,
            executed_code=normalized_code,
        )
    if use_warm_session:
        warm_result, warm_note = _execute_with_warm_session(
            code=normalized_code,
            sage_bin=sage_bin,
            timeout=timeout,
            warm_ttl_seconds=warm_ttl_seconds,
            startup_timeout=warm_startup_timeout,
            max_runs_per_session=warm_max_runs_per_session,
        )
        if warm_result is not None:
            warm_result.executed_code = normalized_code
            if advisory_errors:
                warm_result.validation_errors = _merge_validation_messages(
                    warm_result.validation_errors,
                    advisory_errors,
                )
            return warm_result
        print(f"Warm Sage unavailable, falling back to cold execution: {warm_note}")
    cold_result = run_sage_code(normalized_code, sage_bin=sage_bin, timeout=timeout)
    if advisory_errors:
        cold_result.validation_errors = _merge_validation_messages(
            cold_result.validation_errors,
            advisory_errors,
        )
    if not cold_result.executed_code:
        cold_result.executed_code = normalized_code
    return cold_result
