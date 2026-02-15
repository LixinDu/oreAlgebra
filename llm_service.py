#!/usr/bin/env python3
"""LLM service layer for question answering over retrieved ore_algebra context."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Sequence, Tuple

from dotenv import load_dotenv

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
class LLMRequest:
    question: str
    contexts: Sequence[ContextItem]
    provider: str = "openai"
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    base_url: str | None = None


@dataclass
class LLMResponse:
    answer: str
    code: str
    citations_used: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
    raw_response: str = ""


@dataclass
class Subtask:
    step_id: int
    title: str
    instruction: str
    retrieval_query: str


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


def build_prompt(request: LLMRequest) -> str:
    schema = {
        "answer": "string",
        "code": "string (empty if no code is needed)",
        "citations_used": ["ctx_1", "ctx_2"],
        "missing_info": ["string"],
    }
    ore_algebra_rules = """
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
"""
    return f"""You answer questions about ore_algebra using only provided context.

Rules:
- Use only facts from context.
- If context is insufficient, explain what is missing.
- Cite context IDs you used.
- Return valid JSON only (no markdown).
- Apply the OreAlgebra constructor/generator rules below when producing or checking code.

{ore_algebra_rules}

JSON schema:
{json.dumps(schema, ensure_ascii=True)}

Question:
{request.question}

Context:
{_context_block(request.contexts)}
"""


def _extract_json_object(text: str) -> str:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
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


def parse_response(raw_text: str, allowed_context_ids: Sequence[str]) -> LLMResponse:
    allowed = set(allowed_context_ids)
    payload = json.loads(_extract_json_object(raw_text))

    answer = str(payload.get("answer", "")).strip()
    code = str(payload.get("code", "")).strip()
    missing_info = _coerce_string_list(payload.get("missing_info"))
    citations = [c for c in _coerce_string_list(payload.get("citations_used")) if c in allowed]

    return LLMResponse(
        answer=answer,
        code=code,
        citations_used=citations,
        missing_info=missing_info,
        raw_response=raw_text,
    )


def _normalize_base_url(base_url: str | None, default: str) -> str:
    value = (base_url or "").strip() or default
    return value.rstrip("/")


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


def _call_openai(model: str, prompt: str, temperature: float, api_key: str | None) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency `openai`. Install with: pip install openai") from exc

    client = OpenAI(api_key=key)
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
) -> str:
    key = api_key or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency `openai`. Install with: pip install openai") from exc

    client = OpenAI(api_key=key)
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
    )
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


def _call_gemini(model: str, prompt: str, temperature: float, api_key: str | None) -> str:
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


def _call_ollama(
    model: str,
    prompt: str,
    temperature: float,
    base_url: str | None,
) -> str:
    root = _normalize_base_url(base_url, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    url = f"{root}/api/generate"
    payload = {
        "model": model,
        "prompt": (
            "You are a precise ore_algebra assistant. Return valid JSON only.\n\n"
            f"{prompt}"
        ),
        "stream": False,
        "format": "json",
        "options": {"temperature": float(temperature)},
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
) -> str:
    root = _normalize_base_url(base_url, os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    url = f"{root}/api/generate"
    payload = {
        "model": model,
        "prompt": (
            "You are a precise ore_algebra assistant. Return valid JSON only.\n\n"
            f"{prompt}"
        ),
        "stream": True,
        "format": "json",
        "options": {"temperature": float(temperature)},
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
) -> str:
    if provider == "openai":
        if stream:
            return _call_openai_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
                on_chunk=on_chunk,
            )
        return _call_openai(model=model, prompt=prompt, temperature=temperature, api_key=api_key)
    if provider == "gemini":
        if stream:
            return _call_gemini_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                api_key=api_key,
                on_chunk=on_chunk,
            )
        return _call_gemini(model=model, prompt=prompt, temperature=temperature, api_key=api_key)
    if provider == "ollama":
        if stream:
            return _call_ollama_streaming(
                model=model,
                prompt=prompt,
                temperature=temperature,
                base_url=base_url,
                on_chunk=on_chunk,
            )
        return _call_ollama(
            model=model,
            prompt=prompt,
            temperature=temperature,
            base_url=base_url,
        )
    raise RuntimeError(f"Unsupported provider: {provider}")


def build_plan_prompt(question: str, max_steps: int) -> str:
    schema = {
        "subtasks": [
            {
                "step_id": 1,
                "title": "string",
                "instruction": "string",
                "retrieval_query": "string",
            }
        ]
    }
    return f"""You are planning a retrieval workflow for ore_algebra.

Goal:
- Break the question into focused subtasks.
- Each subtask must be answerable via retrieval from API symbols + PDF support docs.
- Keep steps concise and executable in order.
- Return at most {max_steps} subtasks.
- Return valid JSON only.

Question:
{question}

JSON schema:
{json.dumps(schema, ensure_ascii=True)}
"""


def parse_plan_response(raw_text: str, max_steps: int, fallback_query: str) -> PlanResponse:
    payload = json.loads(_extract_json_object(raw_text))
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
            subtasks.append(
                Subtask(
                    step_id=step_id,
                    title=title,
                    instruction=instruction,
                    retrieval_query=query,
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
) -> PlanResponse:
    prompt = build_plan_prompt(question=question, max_steps=max_steps)
    raw = _call_llm(
        provider=provider,
        model=model,
        prompt=prompt,
        temperature=temperature,
        api_key=api_key,
        base_url=base_url,
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
    payload = json.loads(_extract_json_object(raw_text))
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


def answer_with_llm(
    request: LLMRequest,
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
    stream: bool = False,
    on_chunk: Callable[[str, str], None] | None = None,
) -> LLMResponse:
    prompt = build_prompt(request)
    raw = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        api_key=api_key,
        base_url=request.base_url,
        stream=stream,
        on_chunk=on_chunk,
    )
    allowed_ids = [c.context_id for c in request.contexts]

    try:
        return parse_response(raw, allowed_context_ids=allowed_ids)
    except Exception:
        if parse_repair_attempts <= 0:
            raise

    repair_prompt = f"""Fix the following output into valid JSON only.
Allowed context IDs: {allowed_ids}
Required keys: answer (string), code (string), citations_used (array of allowed IDs), missing_info (array of strings).

Output to fix:
{raw}
"""
    repaired = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=repair_prompt,
        temperature=0.1,
        api_key=api_key,
        base_url=request.base_url,
    )
    return parse_response(repaired, allowed_context_ids=allowed_ids)
