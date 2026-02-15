#!/usr/bin/env python3
"""LLM service layer for question answering over retrieved ore_algebra context."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import List, Sequence

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
    model: str = "gpt-4.1-mini"
    temperature: float = 0.0


@dataclass
class LLMResponse:
    answer: str
    code: str
    citations_used: List[str] = field(default_factory=list)
    missing_info: List[str] = field(default_factory=list)
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


def _call_llm(provider: str, model: str, prompt: str, temperature: float, api_key: str | None) -> str:
    if provider == "openai":
        return _call_openai(model=model, prompt=prompt, temperature=temperature, api_key=api_key)
    if provider == "gemini":
        return _call_gemini(model=model, prompt=prompt, temperature=temperature, api_key=api_key)
    raise RuntimeError(f"Unsupported provider: {provider}")


def answer_with_llm(
    request: LLMRequest,
    api_key: str | None = None,
    parse_repair_attempts: int = 1,
) -> LLMResponse:
    prompt = build_prompt(request)
    raw = _call_llm(
        provider=request.provider,
        model=request.model,
        prompt=prompt,
        temperature=request.temperature,
        api_key=api_key,
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
        temperature=0.0,
        api_key=api_key,
    )
    return parse_response(repaired, allowed_context_ids=allowed_ids)
