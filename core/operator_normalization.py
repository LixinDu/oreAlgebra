#!/usr/bin/env python3
"""Shared helpers for normalizing extracted ore_algebra operator text.

The main goal is to repair common human-written implicit multiplication
patterns before code generation, for example:

- ``5x`` -> ``5*x``
- ``(-2x^2 + x + 1)Dx`` -> ``(-2*x^2 + x + 1)*Dx``
- ``xDx`` -> ``x*Dx`` when the generator is known
- ``(-5*x^2 + x - 1)Sx`` -> ``(-5*x^2 + x - 1)*Sx``

This helper intentionally stays conservative.  It focuses on
high-confidence operator-expression cleanup and avoids trying to be a
general-purpose mathematical parser.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable


SAFE_IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GENERATOR_RE = re.compile(r"^[DSFTQJ][A-Za-z_][A-Za-z0-9_]*$")
NUMBER_RE = re.compile(r"^(?:\d+(?:\.\d*)?|\.\d+)$")
KNOWN_CALLABLE_NAMES = {
    "abs",
    "acos",
    "acosh",
    "asin",
    "asinh",
    "atan",
    "atanh",
    "ceil",
    "cos",
    "cosh",
    "exp",
    "floor",
    "log",
    "root",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
}


def clean_operator_expression(text: str) -> str:
    """Strip common natural-language wrappers from an extracted operator."""

    value = str(text or "").strip()
    value = re.sub(
        r"^\s*(?:the\s+)?(?:q-shift\s+|shift\s+|recurrence\s+|differential\s+)?operator\b",
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()
    value = re.sub(r"^\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*", "", value)
    return value.strip(" .,:;")


def extract_question_symbols(question: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Infer coefficient variables and Ore generators from question ring text."""

    match = re.search(r"\bin\s+(?P<tail>[^.?!]+)", str(question or ""), flags=re.IGNORECASE)
    if not match:
        return (), ()

    tail = match.group("tail")
    raw_groups = re.findall(
        r"[\[\(]\s*([A-Za-z_][A-Za-z0-9_]*(?:\s*,\s*[A-Za-z_][A-Za-z0-9_]*)*)\s*[\]\)]",
        tail,
    )
    if not raw_groups:
        return (), ()

    parsed_groups: list[list[str]] = []
    for group in raw_groups:
        names = [item.strip() for item in group.split(",") if item.strip()]
        valid = [name for name in names if SAFE_IDENTIFIER_RE.fullmatch(name)]
        if valid:
            parsed_groups.append(valid)

    if not parsed_groups:
        return (), ()

    generators: list[str] = []
    variables: list[str] = []
    if all(GENERATOR_RE.fullmatch(name) for name in parsed_groups[-1]):
        generators = parsed_groups[-1]
        for group in parsed_groups[:-1]:
            variables.extend(group)
    else:
        for group in parsed_groups:
            variables.extend(group)

    return _unique(variables), _unique(generators)


def enrich_known_symbols(
    text: str,
    *,
    known_variables: Iterable[str] = (),
    known_generators: Iterable[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Augment known symbols with generator hints seen in the expression."""

    variables = [item for item in known_variables if SAFE_IDENTIFIER_RE.fullmatch(str(item or "").strip())]
    generators = [item for item in known_generators if GENERATOR_RE.fullmatch(str(item or "").strip())]

    for match in re.finditer(r"\b([DSFTQJ][A-Za-z_][A-Za-z0-9_]*)\b", str(text or "")):
        generators.append(match.group(1))

    for generator in generators:
        suffix = generator[1:]
        if SAFE_IDENTIFIER_RE.fullmatch(suffix):
            variables.append(suffix)

    return _unique(variables), _unique(generators)


def normalize_operator_expression(
    text: str,
    *,
    known_variables: Iterable[str] = (),
    known_generators: Iterable[str] = (),
) -> str:
    """Normalize common implicit-multiplication patterns in operator text."""

    value = clean_operator_expression(text)
    if not value:
        return value

    variables, generators = enrich_known_symbols(
        value,
        known_variables=known_variables,
        known_generators=known_generators,
    )
    known_symbols = tuple(
        sorted(
            set(variables) | set(generators),
            key=lambda item: (-len(item), item),
        )
    )

    tokens = _tokenize(value, known_symbols)
    if not tokens:
        return value

    output: list[str] = []
    pending_ws = ""
    previous_non_ws = ""
    known_symbol_set = set(known_symbols)

    for token in tokens:
        if token.isspace():
            pending_ws += token
            continue

        if previous_non_ws:
            if _should_insert_mul(previous_non_ws, token, known_symbol_set):
                output.append(" * " if pending_ws else "*")
            else:
                output.append(pending_ws)
        output.append(token)
        pending_ws = ""
        previous_non_ws = token

    output.append(pending_ws)
    return "".join(output).strip()


def _unique(items: Iterable[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if value and value not in seen:
            seen.append(value)
    return tuple(seen)


def _tokenize(text: str, known_symbols: tuple[str, ...]) -> list[str]:
    tokens: list[str] = []
    index = 0
    limit = len(text)

    while index < limit:
        ch = text[index]
        if ch.isspace():
            end = index + 1
            while end < limit and text[end].isspace():
                end += 1
            tokens.append(text[index:end])
            index = end
            continue

        if ch in "+-*/^(),[]=":
            tokens.append(ch)
            index += 1
            continue

        if ch.isdigit() or (ch == "." and index + 1 < limit and text[index + 1].isdigit()):
            end = index + 1
            while end < limit and (text[end].isdigit() or text[end] == "."):
                end += 1
            tokens.append(text[index:end])
            index = end
            continue

        if ch.isalpha() or ch == "_":
            end = index + 1
            while end < limit and (text[end].isalnum() or text[end] == "_"):
                end += 1
            identifier = text[index:end]
            tokens.extend(_segment_identifier(identifier, known_symbols))
            index = end
            continue

        tokens.append(ch)
        index += 1

    return tokens


@lru_cache(maxsize=512)
def _segment_identifier(identifier: str, known_symbols: tuple[str, ...]) -> tuple[str, ...]:
    if identifier in known_symbols or not known_symbols:
        return (identifier,)

    symbols = tuple(sorted(known_symbols, key=lambda item: (-len(item), item)))
    limit = len(identifier)
    memo: dict[int, tuple[str, ...] | None] = {}

    def solve(position: int) -> tuple[str, ...] | None:
        if position == limit:
            return ()
        if position in memo:
            return memo[position]

        for symbol in symbols:
            if identifier.startswith(symbol, position):
                rest = solve(position + len(symbol))
                if rest is not None:
                    memo[position] = (symbol,) + rest
                    return memo[position]

        memo[position] = None
        return None

    parts = solve(0)
    if parts is None or len(parts) < 2:
        return (identifier,)
    return parts


def _should_insert_mul(left: str, right: str, known_symbols: set[str]) -> bool:
    if not left or not right:
        return False
    if _is_operator(left) or _is_operator(right):
        return False
    if left in {"(", "["} or right in {")", "]"}:
        return False

    if left in {")", "]"}:
        return _is_identifier(right) or _is_number(right) or right in {"(", "["}

    if _is_number(left):
        return _is_identifier(right) or right in {"(", "["}

    if _is_identifier(left):
        if _is_identifier(right):
            return True
        if right in {"(", "["}:
            return left in known_symbols or len(left) == 1 or left not in KNOWN_CALLABLE_NAMES

    return False


def _is_operator(token: str) -> bool:
    return token in {"+", "-", "*", "/", "^", ",", "="}


def _is_number(token: str) -> bool:
    return bool(NUMBER_RE.fullmatch(token))


def _is_identifier(token: str) -> bool:
    return bool(SAFE_IDENTIFIER_RE.fullmatch(token))
