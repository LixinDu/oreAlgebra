#!/usr/bin/env python3
"""LLM service layer for question answering over retrieved ore_algebra context."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, List, Sequence, Tuple

try:
    from dotenv import load_dotenv
except Exception:
    def load_dotenv(*_args: object, **_kwargs: object) -> bool:
        return False
from core.sage_runtime import SageExecutionResult
from workflows.task_workflows import build_capability_family_prompt_block, normalize_family_id

# Auto-load .env so provider keys can be omitted in UI.
load_dotenv(override=False)


@dataclass
class ContextItem:
    context_id: str
    source_type: str
    title: str
    location: str
    text: str
    score: float = 0.0


@dataclass
class CodeGenerationRequest:
    question: str
    contexts: Sequence[ContextItem]
    task_workflow_hint: str = ""
    resolved_task: Any | None = None
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    base_url: str | None = None
    max_output_tokens: int | None = None


@dataclass
class CodeGenerationResponse:
    code: str
    citations_used: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class ExecutionAwareAnswerRequest:
    question: str
    contexts: Sequence[ContextItem]
    original_code: str
    execution_result: SageExecutionResult | None = None
    execution_skipped_reason: str = ""
    code_generation_citations: List[str] = field(default_factory=list)
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    base_url: str | None = None
    max_output_tokens: int | None = None


@dataclass
class FinalAnswerResponse:
    answer: str
    citations_used: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class Subtask:
    step_id: int
    title: str
    instruction: str
    retrieval_query: str
    family_id: str = ""


@dataclass
class PlanResponse:
    subtasks: List[Subtask] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class StepDecision:
    action: str
    reason: str
    next_query: str = ""
    confidence: float = 0.0
    raw_response: str = ""


ORE_ALGEBRA_RULES = """
OreAlgebra constructor and generator naming rules:
- If no custom behavior is specified, generator names define operator type.
- For non-commutative generators, use one-letter prefix + base-ring variable name.
- Typical defaults:
  - Dt means standard derivation d/dt on base variable t.
  - Sn means standard shift on base variable n (n -> n + 1).

Allowed prefixes and meanings:
- Dx: standard derivation d/dx
- Sx: standard shift x -> x + 1
- Tx: Eulerian derivation x*d/dx
- Fx: forward difference Δx
- Qx: q-shift x -> q*x
- Jx: q-derivation (Jackson derivation)
- Cx: commutative generator

Commutation rules with base variable x:
- Dx*x = x*Dx + 1
- Sx*x = (x + 1)*Sx
- Tx*x = x*Tx + x
- Fx*x = (x + 1)*Fx + 1
- Qx*x = q*x*Qx
- Jx*x = q*x*Jx + 1
- Cx*x = x*Cx

Guessing rules:
- To find an operator that annihilates a given sequence, generate enough terms as a list and call guess(data, A) where A is the target algebra. Do NOT use guess_rec(data, n, Sn) — it has coercion issues on some data types.
- Example: guess([0, 1, 4, 9, 16, 25, 36, 49, 64, 81], OreAlgebra(QQ['n'], 'Sn'))
"""

DEFAULT_MISTAKE_RULES_PATH = Path(__file__).resolve().parents[1] / "config" / "mistake_rules.json"


def _load_mistake_rules_payload() -> dict:
    path_raw = (os.getenv("MISTAKE_RULES_FILE", "") or "").strip()
    path = Path(path_raw).expanduser() if path_raw else DEFAULT_MISTAKE_RULES_PATH
    if not path.exists():
        return {"global": [], "rules": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"global": [], "rules": []}
    if not isinstance(payload, dict):
        return {"global": [], "rules": []}
    return payload


def _build_mistake_avoidance_block(request: CodeGenerationRequest) -> str:
    payload = _load_mistake_rules_payload()
    global_rules = _coerce_string_list(payload.get("global"))
    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raw_rules = []

    question_haystack = request.question.lower()
    context_haystack = "\n".join(
        f"{item.title}\n{item.location}\n{item.text}" for item in request.contexts
    ).lower()

    lines: List[str] = []
    seen_messages = set()

    for message in global_rules:
        msg = message.strip()
        if not msg or msg in seen_messages:
            continue
        seen_messages.add(msg)
        lines.append(f"- {msg}")

    for idx, item in enumerate(raw_rules, start=1):
        if not isinstance(item, dict):
            continue
        message = str(item.get("message", "")).strip()
        when_any = [token.lower() for token in _coerce_string_list(item.get("when_any")) if token.strip()]
        scope = str(item.get("scope", "all")).strip().lower()
        if not message or not when_any:
            continue
        if scope == "question":
            haystack = question_haystack
        elif scope == "context":
            haystack = context_haystack
        else:
            haystack = f"{question_haystack}\n{context_haystack}"

        if not any(token in haystack for token in when_any):
            continue

        if message in seen_messages:
            continue
        seen_messages.add(message)
        rule_id = str(item.get("id", "")).strip() or f"rule_{idx}"
        lines.append(f"- [{rule_id}] {message}")
        if len(lines) >= 8:
            break

    return "\n".join(lines)


def _context_block(contexts: Sequence[ContextItem]) -> str:
    blocks = []
    for item in contexts:
        blocks.append(
            "\n".join(
                [
                    f"[{item.context_id}]",
                    f"source_type: {item.source_type}",
                    f"title: {item.title}",
                    f"location: {item.location}",
                    f"score: {item.score:.4f}",
                    "text:",
                    item.text,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_code_generation_prompt(request: CodeGenerationRequest) -> str:
    schema = {
        "code": "string (empty if context is insufficient or no code is needed)",
        "citations_used": ["ctx_1", "ctx_2"],
        "missing_info": ["string"],
    }
    workflow_hint = ""
    if request.task_workflow_hint.strip():
        workflow_hint = f"\n{request.task_workflow_hint.strip()}\n"
    return f"""You generate SageMath code for ore_algebra using only provided retrieval context.

Rules:
- Use only facts, APIs, and syntax supported by the context.
- Return valid JSON only (no markdown).
- If context is insufficient, return empty code and explain what is missing.
- Do not simulate execution; the app will validate and run the code separately.
- Do not include markdown fences in the `code` string.
- Do not add `from sage.all import *`; the runtime adds it automatically.
- Add explicit ore_algebra imports for APIs you use, e.g. `from ore_algebra import OreAlgebra` or `from ore_algebra import DifferentialOperators`.
- Prefer explicit ore_algebra imports over wildcard imports, unless wildcard style is clearly required by the retrieved example.
- Cite only the provided context IDs.
- Sage shorthand is allowed when it improves clarity because the runtime will preparse before execution.
- When computing a final result, include explicit `print(...)` statements for key outputs so script execution shows the result in stdout.
- If the question asks for a multi-step result such as convert-then-compute or guess-then-continue, complete the full chain instead of stopping after the intermediate operator.
- If the question asks to find, infer, convert, or identify an operator before computing downstream values, print that operator in stdout before printing the downstream values.
- When converting between operator types, use the correct source algebra and a distinct target algebra when needed.
- Apply the OreAlgebra constructor/generator rules below when writing code.

{ORE_ALGEBRA_RULES}
{workflow_hint}

JSON schema:
{json.dumps(schema, ensure_ascii=True)}

Question:
{request.question}

Context:
{_context_block(request.contexts)}
"""


def build_code_correction_prompt(
    request: CodeGenerationRequest,
    original_code: str,
    validation_errors: Sequence[str],
) -> str:
    mistake_block = _build_mistake_avoidance_block(request)
    mistake_section = ""
    if mistake_block:
        mistake_section = (
            "Mistake-avoidance reminders for this correction:\n"
            f"{mistake_block}\n\n"
        )
    schema = {
        "code": "string",
        "citations_used": ["ctx_1", "ctx_2"],
        "missing_info": ["string"],
    }
    workflow_hint = ""
    if request.task_workflow_hint.strip():
        workflow_hint = f"\n{request.task_workflow_hint.strip()}\n"
    error_lines = "\n".join(f"- {item}" for item in validation_errors if str(item).strip()) or "- (none provided)"
    return f"""Fix the generated SageMath ore_algebra code using the validation errors below.

Rules:
- Keep the original intent and output shape.
- Make minimal edits; do not rewrite from scratch unless required.
- Fix validation errors first.
- Use only APIs/syntax supported by the provided context.
- Return valid JSON only (no markdown), following the schema exactly.
- Do not include markdown fences in the `code` string.

Validation errors to fix:
{error_lines}

Original code:
```python
{original_code.strip() or "(empty)"}
```

{workflow_hint}
{mistake_section}JSON schema:
{json.dumps(schema, ensure_ascii=True)}

Question:
{request.question}

Context:
{_context_block(request.contexts)}
"""


def _execution_result_block(
    execution_result: SageExecutionResult | None,
    execution_skipped_reason: str,
) -> str:
    if execution_result is None:
        reason = execution_skipped_reason.strip() or "Execution was skipped."
        return "\n".join(
            [
                "status: skipped",
                "preflight_ok: false",
                "is_truncated: false",
                "returncode: n/a",
                f"skip_reason: {reason}",
                "stdout_summary:",
                "(none)",
                "stderr:",
                "(none)",
                "validation_errors:",
                "(none)",
            ]
        )

    validation_errors = execution_result.validation_errors or ["(none)"]
    stdout_summary = execution_result.stdout_summary or "(empty)"
    stderr_text = execution_result.stderr or "(empty)"
    lines = [
        f"status: {execution_result.status}",
        f"preflight_ok: {str(execution_result.preflight_ok).lower()}",
        f"is_truncated: {str(execution_result.is_truncated).lower()}",
        f"returncode: {execution_result.returncode}",
        "validation_errors:",
        *validation_errors,
        "stdout_summary:",
        stdout_summary,
        "stderr:",
        stderr_text,
    ]
    if execution_result.is_truncated:
        lines.append("note: Full output is available in the 'Full Output' section.")
    return "\n".join(lines)


def build_execution_answer_prompt(request: ExecutionAwareAnswerRequest) -> str:
    schema = {
        "answer": "string",
        "citations_used": ["ctx_1", "ctx_2"],
        "missing_info": ["string"],
    }
    code_generation_citations = ", ".join(request.code_generation_citations) or "(none)"
    original_code = request.original_code.strip() or "(no code generated)"
    return f"""You answer a user question about ore_algebra using retrieved context and an app-supplied Sage execution result.

Rules:
- Use retrieved context as the factual source of truth.
- Use the execution result only as observed runtime evidence.
- Return valid JSON only (no markdown).
- Do not generate replacement code or code fences in the final answer.
- Format mathematical expressions in the prose answer using LaTeX.
- Do not convert raw runtime output into LaTeX.
- Cite only the provided context IDs.
- Never hallucinate a computed math result when execution did not succeed.
- If execution status is `blocked`, `error`, or `timeout`, explain why using `validation_errors` and/or `stderr`.
- If `is_truncated` is true, say that the full result is visible in the `Full Output` section and summarize only the visible part.
- If execution status is `success` and `stdout_summary` is non-empty, include the concrete computed runtime result from `stdout_summary` in the answer.
- If execution status is `success` but `stdout_summary` is empty, explicitly say execution succeeded but produced no visible stdout result.
- Your output must be valid JSON. Escape every LaTeX backslash in JSON strings as `\\`.

JSON schema:
{json.dumps(schema, ensure_ascii=True)}

Question:
{request.question}

Retrieved context:
{_context_block(request.contexts)}

Original generated code:
```python
{original_code}
```

Code generation citations:
{code_generation_citations}

Sage execution result:
{_execution_result_block(request.execution_result, request.execution_skipped_reason)}
"""


def _iter_balanced_json_candidates(text: str) -> Iterator[str]:
    """Yield all balanced {...} spans, scanning left-to-right, string-aware."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            else:
                if ch == '"':
                    in_string = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[start : i + 1].strip()
                        break
        start = text.find("{", start + 1)


def _extract_json_object(text: str) -> str:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    first = ""
    for cand in _iter_balanced_json_candidates(text):
        if not first:
            first = cand
        try:
            json.loads(cand)
            return cand
        except json.JSONDecodeError:
            continue
    if first:
        return first
    # Fallback: first { to last } (legacy behavior).
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text.strip()


def _coerce_string_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        if item is None:
            continue
        out.append(str(item))
    return out


def _coerce_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def parse_code_generation_response(
    raw_text: str,
    allowed_context_ids: Sequence[str],
) -> CodeGenerationResponse:
    allowed = set(allowed_context_ids)
    payload = _loads_json_object(raw_text)

    code = str(payload.get("code", "")).strip()
    missing_info = _coerce_string_list(payload.get("missing_info"))
    citations = [c for c in _coerce_string_list(payload.get("citations_used")) if c in allowed]

    return CodeGenerationResponse(
        code=code,
        citations_used=citations,
        missing_info=missing_info,
        raw_response=raw_text,
    )


def parse_execution_answer_response(
    raw_text: str,
    allowed_context_ids: Sequence[str],
) -> FinalAnswerResponse:
    allowed = set(allowed_context_ids)
    payload = _loads_json_object(raw_text)

    answer = str(payload.get("answer", "")).strip()
    missing_info = _coerce_string_list(payload.get("missing_info"))
    citations = [c for c in _coerce_string_list(payload.get("citations_used")) if c in allowed]

    return FinalAnswerResponse(
        answer=answer,
        citations_used=citations,
        missing_info=missing_info,
        raw_response=raw_text,
    )


def _json_repair_prompt(
    *,
    raw_output: str,
    allowed_context_ids: Sequence[str],
    required_keys_text: str,
) -> str:
    return f"""Fix the following output into valid JSON only.
Allowed context IDs: {list(allowed_context_ids)}
Required keys: {required_keys_text}
Important: if the answer contains LaTeX, escape backslashes in JSON strings (use `\\\\` in JSON text).

Output to fix:
{raw_output}
"""


def _loads_json_object(raw_text: str) -> dict:
    candidate = _extract_json_object(raw_text)

    def _repair_invalid_backslashes(text: str) -> str:
        # 1) Escape invalid JSON backslashes, treating \u as valid only for uXXXX.
        repaired = re.sub(r'\\(?!["\\/bfnrt]|u[0-9a-fA-F]{4})', r"\\\\", text)
        # 2) Preserve common LaTeX commands that start with JSON-valid escapes (\t, \n, \r, \b, \f).
        #    Example: \text should be literal backslash + text, not tab + "ext".
        repaired = re.sub(r'\\([tnrbf])(?=[A-Za-z])', r"\\\\\1", repaired)
        return repaired

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError as exc:
        message = str(exc)
        if "Invalid \\escape" in message or "Invalid \\uXXXX escape" in message:
            repaired = _repair_invalid_backslashes(candidate)
            payload = json.loads(repaired)
        elif "Unterminated string" in message:
            snippet = (raw_text or "")[:200].replace("\n", "\\n")
            raise json.JSONDecodeError(
                f"{message}. LLM output likely truncated (raise max_output_tokens). "
                f"Raw prefix: {snippet!r}",
                exc.doc,
                exc.pos,
            ) from exc
        else:
            raise
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object.")
    return payload


def _normalize_base_url(base_url: str | None, default: str) -> str:
    value = (base_url or "").strip() or default
    return value.rstrip("/")


def _normalized_max_output_tokens(value: int | None) -> int | None:
    if value is None:
        return None
    try:
        tokens = int(value)
    except Exception:
        return None
    if tokens <= 0:
        return None
    return tokens


def _openai_chat_completion_kwargs(max_output_tokens: int | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    max_tokens = _normalized_max_output_tokens(max_output_tokens)
    if max_tokens is not None:
        kwargs["max_completion_tokens"] = max_tokens
    return kwargs


def _openai_should_retry_without_token_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    if "unsupported parameter" in text and "max_tokens" in text:
        return True
    if "max_completion_tokens" in text and (
        "unexpected keyword argument" in text
        or "unrecognized request argument" in text
        or "unknown parameter" in text
        or "unsupported parameter" in text
    ):
        return True
    return False


def list_ollama_models(base_url: str | None = None, timeout_seconds: float = 3.0) -> Tuple[List[str], str]:
    """Return available Ollama model names and an optional error message."""
    root = _normalize_base_url(base_url, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    url = f"{root}/api/tags"
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        payload = json.loads(body)
        raw_models = payload.get("models", [])
        names: List[str] = []
        if isinstance(raw_models, list):
            for item in raw_models:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name", "")).strip()
                if name:
                    names.append(name)
        # Stable + unique list for UI.
        unique_sorted = sorted(set(names))
        return unique_sorted, ""
    except Exception as exc:
        return [], str(exc)


def _call_openai(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    max_output_tokens: int | None = None,
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency `openai`. Install with: pip install openai") from exc

    client = OpenAI(api_key=key)
    kwargs = _openai_chat_completion_kwargs(max_output_tokens)
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise ore_algebra assistant. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
    except Exception as exc:
        if not kwargs or not _openai_should_retry_without_token_limit(exc):
            raise
        # Compatibility fallback: some SDK/API combinations reject token-limit
        # fields for specific model families even when they are expected.
        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise ore_algebra assistant. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
        )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    return str(content)


def _call_openai_streaming(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    on_chunk: Callable[[str, str], None] | None,
    max_output_tokens: int | None = None,
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency `openai`. Install with: pip install openai") from exc

    client = OpenAI(api_key=key)
    kwargs = _openai_chat_completion_kwargs(max_output_tokens)
    try:
        stream = client.chat.completions.create(
            model=model,
            temperature=temperature,
            stream=True,
            messages=[
                {
                    "role": "system",
                    "content": "You are a precise ore_algebra assistant. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ],
            **kwargs,
        )
    except Exception as exc:
        if not kwargs or not _openai_should_retry_without_token_limit(exc):
            raise
        # Fallback to non-streaming when token-limit compatibility fails.
        full_text = _call_openai(
            model=model,
            prompt=prompt,
            temperature=temperature,
            api_key=api_key,
            max_output_tokens=None,
        )
        if full_text and on_chunk is not None:
            on_chunk(full_text, full_text)
        return full_text
    acc: List[str] = []
    for chunk in stream:
        piece = ""
        try:
            delta = chunk.choices[0].delta.content
            if isinstance(delta, str):
                piece = delta
            elif isinstance(delta, list):
                parts = []
                for d in delta:
                    txt = getattr(d, "text", None)
                    if isinstance(txt, str):
                        parts.append(txt)
                piece = "".join(parts)
        except Exception:
            piece = ""
        if not piece:
            continue
        acc.append(piece)
        if on_chunk is not None:
            on_chunk(piece, "".join(acc))
    return "".join(acc)


def _call_gemini(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    max_output_tokens: int | None = None,
) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `google-generativeai`. Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=key)
    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
    }
    max_tokens = _normalized_max_output_tokens(max_output_tokens)
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens
    model_obj = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    resp = model_obj.generate_content(prompt)
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text
    # Fallback for SDK variants.
    return str(resp)


def _call_gemini_streaming(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    on_chunk: Callable[[str, str], None] | None,
    max_output_tokens: int | None = None,
) -> str:
    key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set.")

    try:
        import google.generativeai as genai  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `google-generativeai`. Install with: pip install google-generativeai"
        ) from exc

    genai.configure(api_key=key)
    generation_config = {
        "temperature": temperature,
        "response_mime_type": "application/json",
    }
    max_tokens = _normalized_max_output_tokens(max_output_tokens)
    if max_tokens is not None:
        generation_config["max_output_tokens"] = max_tokens
    model_obj = genai.GenerativeModel(model_name=model, generation_config=generation_config)
    stream = model_obj.generate_content(prompt, stream=True)
    acc: List[str] = []
    for chunk in stream:
        piece = getattr(chunk, "text", None)
        if not isinstance(piece, str) or not piece:
            continue
        acc.append(piece)
        if on_chunk is not None:
            on_chunk(piece, "".join(acc))
    return "".join(acc)


def _anthropic_temperature_supported(model: str) -> bool:
    model_lower = model.lower()
    if "opus-4-7" in model_lower or model_lower.startswith("claude-opus-4-7"):
        return False
    return True


def _anthropic_api_key(api_key: str | None) -> str:
    key = api_key or os.getenv("ANTHROPIC_API_KEY") or os.getenv("My_Claude_Key")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY (or My_Claude_Key) is not set."
        )
    return key


def _call_anthropic(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    max_output_tokens: int | None = None,
) -> str:
    key = _anthropic_api_key(api_key)

    try:
        import anthropic  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `anthropic`. Install with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=key)
    max_tokens = _normalized_max_output_tokens(max_output_tokens) or 16384
    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": (
            "You are a precise ore_algebra assistant. Respond with a single JSON "
            "object and nothing else. Do not add prose before or after the JSON. "
            "Do not wrap the JSON in Markdown fences."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    if _anthropic_temperature_supported(model):
        kwargs["temperature"] = temperature

    resp = client.messages.create(**kwargs)
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


def _call_anthropic_streaming(
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    on_chunk: Callable[[str, str], None] | None,
    max_output_tokens: int | None = None,
) -> str:
    key = _anthropic_api_key(api_key)

    try:
        import anthropic  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `anthropic`. Install with: pip install anthropic"
        ) from exc

    client = anthropic.Anthropic(api_key=key)
    max_tokens = _normalized_max_output_tokens(max_output_tokens) or 16384
    stream_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": (
            "You are a precise ore_algebra assistant. Respond with a single JSON "
            "object and nothing else. Do not add prose before or after the JSON. "
            "Do not wrap the JSON in Markdown fences."
        ),
        "messages": [{"role": "user", "content": prompt}],
    }
    if _anthropic_temperature_supported(model):
        stream_kwargs["temperature"] = temperature

    acc: List[str] = []
    with client.messages.stream(**stream_kwargs) as stream:
        for piece in stream.text_stream:
            if not piece:
                continue
            acc.append(piece)
            if on_chunk is not None:
                on_chunk(piece, "".join(acc))
    return "".join(acc)


def _call_ollama(
    model: str,
    prompt: str,
    temperature: float,
    base_url: str | None,
    max_output_tokens: int | None = None,
) -> str:
    root = _normalize_base_url(base_url, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    url = f"{root}/api/generate"
    options = {"temperature": float(temperature)}
    max_tokens = _normalized_max_output_tokens(max_output_tokens)
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "prompt": (
            "You are a precise ore_algebra assistant. Return valid JSON only.\n\n"
            f"{prompt}"
        ),
        "stream": False,
        "format": "json",
        "options": options,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120.0) as resp:
            body = resp.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama request failed: {exc}") from exc

    try:
        data = json.loads(body)
    except Exception as exc:
        raise RuntimeError(f"Ollama returned non-JSON response: {body[:2000]}") from exc

    text = data.get("response")
    if isinstance(text, str):
        return text
    return str(data)


def _call_ollama_streaming(
    model: str,
    prompt: str,
    temperature: float,
    base_url: str | None,
    on_chunk: Callable[[str, str], None] | None,
    max_output_tokens: int | None = None,
) -> str:
    root = _normalize_base_url(base_url, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    url = f"{root}/api/generate"
    options = {"temperature": float(temperature)}
    max_tokens = _normalized_max_output_tokens(max_output_tokens)
    if max_tokens is not None:
        options["num_predict"] = max_tokens
    payload = {
        "model": model,
        "prompt": (
            "You are a precise ore_algebra assistant. Return valid JSON only.\n\n"
            f"{prompt}"
        ),
        "stream": True,
        "format": "json",
        "options": options,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    acc: List[str] = []
    try:
        with urllib.request.urlopen(req, timeout=300.0) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = item.get("response")
                if not isinstance(piece, str) or not piece:
                    continue
                acc.append(piece)
                if on_chunk is not None:
                    on_chunk(piece, "".join(acc))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Ollama streaming request failed: {exc}") from exc
    return "".join(acc)


def _call_llm(
    provider: str,
    model: str,
    prompt: str,
    temperature: float,
    api_key: str | None,
    base_url: str | None = None,
    stream: bool = False,
    on_chunk: Callable[[str, str], None] | None = None,
    max_output_tokens: int | None = None,
) -> str:
    if provider == "openai":
        if stream:
            return _call_openai_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
                on_chunk=on_chunk,
                max_output_tokens=max_output_tokens,
            )
        return _call_openai(
            model=model,
            prompt=prompt,
            temperature=temperature,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider == "gemini":
        if stream:
            return _call_gemini_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
                on_chunk=on_chunk,
                max_output_tokens=max_output_tokens,
            )
        return _call_gemini(
            model=model,
            prompt=prompt,
            temperature=temperature,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    if provider == "ollama":
        if stream:
            return _call_ollama_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                base_url=base_url,
                on_chunk=on_chunk,
                max_output_tokens=max_output_tokens,
            )
        return _call_ollama(
            model=model,
            prompt=prompt,
            temperature=temperature,
            base_url=base_url,
            max_output_tokens=max_output_tokens,
        )
    if provider == "anthropic":
        if stream:
            return _call_anthropic_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
                on_chunk=on_chunk,
                max_output_tokens=max_output_tokens,
            )
        return _call_anthropic(
            model=model,
            prompt=prompt,
            temperature=temperature,
            api_key=api_key,
            max_output_tokens=max_output_tokens,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def build_plan_prompt(question: str, max_steps: int) -> str:
    schema = {
        "subtasks": [
            {
                "step_id": 1,
                "family_id": "A",
                "title": "string",
                "instruction": "string",
                "retrieval_query": "string",
            }
        ]
    }
    families_block = build_capability_family_prompt_block()
    return f"""You are planning a retrieval workflow for ore_algebra.

Goal:
- Break the question into focused subtasks.
- Each subtask must be answerable via retrieval from API symbols + PDF support docs.
- Keep steps concise and executable in order.
- Tag each subtask with exactly one capability family_id from A-G.
- Return at most {max_steps} subtasks.
- Return valid JSON only.

{families_block}

Question:
{question}

JSON schema:
{json.dumps(schema, ensure_ascii=True)}
"""


def parse_plan_response(raw_text: str, max_steps: int, fallback_query: str) -> PlanResponse:
    payload = _loads_json_object(raw_text)
    raw_subtasks = payload.get("subtasks", [])
    subtasks: List[Subtask] = []
    if isinstance(raw_subtasks, list):
        for idx, item in enumerate(raw_subtasks[:max_steps], start=1):
            if not isinstance(item, dict):
                continue
            step_id = int(item.get("step_id") or idx)
            title = str(item.get("title", "")).strip() or f"Step {idx}"
            instruction = str(item.get("instruction", "")).strip()
            query = str(item.get("retrieval_query", "")).strip() or instruction or title
            family_id = normalize_family_id(str(item.get("family_id", "")).strip())
            subtasks.append(
                Subtask(
                    step_id=step_id,
                    title=title,
                    instruction=instruction,
                    retrieval_query=query,
                    family_id=family_id,
                )
            )
    if not subtasks:
        subtasks = [
            Subtask(
                step_id=1,
                title="Direct retrieval",
                instruction="Retrieve direct evidence for the user question",
                retrieval_query=fallback_query or "ore_algebra usage details",
            )
        ]
    return PlanResponse(subtasks=subtasks, raw_response=raw_text)


def plan_subtasks(
    question: str,
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    max_steps: int = 5,
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
) -> PlanResponse:
    prompt = build_plan_prompt(question=question, max_steps=max_steps)
    raw = _call_llm(
        provider=provider,
        model=model,
        prompt=prompt,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
    )
    try:
        return parse_plan_response(raw_text=raw, max_steps=max_steps, fallback_query=question)
    except Exception:
        # deterministic fallback
        return PlanResponse(
            subtasks=[
                Subtask(
                    step_id=1,
                    title="Direct retrieval",
                    instruction="Retrieve direct evidence for the user question",
                    retrieval_query=question,
                )
            ],
            raw_response=raw,
        )


def build_decision_prompt(
    question: str,
    current_step: Subtask,
    context_items: Sequence[ContextItem],
) -> str:
    schema = {
        "action": "continue | refine_query | stop",
        "reason": "string",
        "next_query": "string",
        "confidence": 0.0,
    }
    return f"""Decide the next action in a step-by-step retrieval workflow.

Actions:
- continue: enough evidence for this step, move to next planned step.
- refine_query: evidence is weak; run one more retrieval for this step with next_query.
- stop: enough evidence overall or no productive next action.

Question:
{question}

Current step:
- step_id: {current_step.step_id}
- title: {current_step.title}
- instruction: {current_step.instruction}
- retrieval_query: {current_step.retrieval_query}

Retrieved context:
{_context_block(context_items)}

Return valid JSON only.
JSON schema:
{json.dumps(schema, ensure_ascii=True)}
"""


def parse_decision_response(raw_text: str) -> StepDecision:
    payload = _loads_json_object(raw_text)
    action = str(payload.get("action", "continue")).strip().lower()
    if action not in {"continue", "refine_query", "stop"}:
        action = "continue"
    reason = str(payload.get("reason", "")).strip()
    next_query = str(payload.get("next_query", "")).strip()
    confidence = _coerce_float(payload.get("confidence"), default=0.0)
    return StepDecision(
        action=action,
        reason=reason,
        next_query=next_query,
        confidence=confidence,
        raw_response=raw_text,
    )


def decide_next_action(
    question: str,
    current_step: Subtask,
    context_items: Sequence[ContextItem],
    provider: str,
    model: str,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float = 0.1,
    max_output_tokens: int | None = None,
) -> StepDecision:
    prompt = build_decision_prompt(
        question=question,
        current_step=current_step,
        context_items=context_items,
    )
    raw = _call_llm(
        provider=provider,
        model=model,
        prompt=prompt,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
        max_output_tokens=max_output_tokens,
    )
    try:
        return parse_decision_response(raw)
    except Exception:
        return StepDecision(
            action="continue",
            reason="Fallback decision: continue to next step.",
            next_query="",
            confidence=0.0,
            raw_response=raw,
        )


def generate_code_with_llm(
    request: CodeGenerationRequest,
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
) -> CodeGenerationResponse:
    prompt = build_code_generation_prompt(request)
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

    try:
        return parse_code_generation_response(raw, allowed_context_ids=allowed_ids)
    except Exception:
        if parse_repair_attempts <= 0:
            raise

    repaired = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=_json_repair_prompt(
            raw_output=raw,
            allowed_context_ids=allowed_ids,
            required_keys_text=(
                "code (string), citations_used (array of allowed IDs), "
                "missing_info (array of strings)"
            ),
        ),
        temperature=0.1,
        api_key=api_key,
        base_url=request.base_url,
        max_output_tokens=request.max_output_tokens,
    )
    return parse_code_generation_response(repaired, allowed_context_ids=allowed_ids)


def repair_code_with_llm(
    *,
    request: CodeGenerationRequest,
    original_code: str,
    validation_errors: Sequence[str],
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
) -> CodeGenerationResponse:
    prompt = build_code_correction_prompt(
        request=request,
        original_code=original_code,
        validation_errors=validation_errors,
    )
    raw = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=prompt,
        temperature=0.1,
        api_key=api_key,
        base_url=request.base_url,
        max_output_tokens=request.max_output_tokens,
    )
    allowed_ids = [c.context_id for c in request.contexts]

    try:
        return parse_code_generation_response(raw, allowed_context_ids=allowed_ids)
    except Exception:
        if parse_repair_attempts <= 0:
            raise

    repaired = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=_json_repair_prompt(
            raw_output=raw,
            allowed_context_ids=allowed_ids,
            required_keys_text=(
                "code (string), citations_used (array of allowed IDs), "
                "missing_info (array of strings)"
            ),
        ),
        temperature=0.1,
        api_key=api_key,
        base_url=request.base_url,
        max_output_tokens=request.max_output_tokens,
    )
    return parse_code_generation_response(repaired, allowed_context_ids=allowed_ids)


def answer_with_execution_llm(
    request: ExecutionAwareAnswerRequest,
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
    stream: bool = False,
    on_chunk: Callable[[str, str], None] | None = None,
) -> FinalAnswerResponse:
    prompt = build_execution_answer_prompt(request)
    raw = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        api_key=api_key,
        base_url=request.base_url,
        stream=stream,
        on_chunk=on_chunk,
        max_output_tokens=request.max_output_tokens,
    )
    allowed_ids = [c.context_id for c in request.contexts]

    try:
        return parse_execution_answer_response(raw, allowed_context_ids=allowed_ids)
    except Exception:
        if parse_repair_attempts <= 0:
            raise

    repaired = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=_json_repair_prompt(
            raw_output=raw,
            allowed_context_ids=allowed_ids,
            required_keys_text=(
                "answer (string), citations_used (array of allowed IDs), "
                "missing_info (array of strings)"
            ),
        ),
        temperature=0.1,
        api_key=api_key,
        base_url=request.base_url,
        max_output_tokens=request.max_output_tokens,
    )
    return parse_execution_answer_response(repaired, allowed_context_ids=allowed_ids)
