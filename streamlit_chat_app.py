#!/usr/bin/env python3
"""Streamlit chat app: retrieve context, call LLM, and show cited answer."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List

import streamlit as st

from llm_service import ContextItem, LLMRequest, LLMResponse, answer_with_llm
from ore_rag_assistant import (
    RetrievalResult,
    load_index,
    location_label,
    parse_chunks,
    select_retrieval,
)


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


def _citation_lines(response: LLMResponse, context_items: List[ContextItem]) -> str:
    ctx_by_id: Dict[str, ContextItem] = {c.context_id: c for c in context_items}
    lines: List[str] = []
    for cid in response.citations_used:
        item = ctx_by_id.get(cid)
        if not item:
            continue
        lines.append(f"- {cid}: {item.title} ({item.location})")
    if not lines:
        return "No citations returned."
    return "\n".join(lines)


def _render_retrieval_results(results: List[RetrievalResult], pdf_char_limit: int) -> None:
    st.subheader("Retrieved Context")
    for r in results:
        with st.expander(_result_title(r), expanded=False):
            st.write(f"source_type: `{r.source_type}`")
            st.write(f"source: `{r.source}`")
            if r.source_type == "generated":
                if r.signature:
                    st.write(f"signature: `{r.signature}`")
                if r.module:
                    st.write(f"module: `{r.module}`")
                st.write(f"location: `{location_label(r)}`")
                st.text_area("content", value=r.text, height=260, key=f"text-{r.chunk_id}")
            else:
                section = r.section_title or "unknown"
                st.write(f"pages: `{r.page_start}-{r.page_end}`")
                st.write(f"section: `{section}`")
                preview = r.text[:pdf_char_limit] if len(r.text) > pdf_char_limit else r.text
                st.text_area("content", value=preview, height=260, key=f"text-{r.chunk_id}")


def main() -> None:
    st.set_page_config(page_title="ore_algebra Chat", layout="wide")
    st.title("ore_algebra Chat")
    st.caption("Retrieves top-k context, then sends question + context to LLM.")

    with st.sidebar:
        st.header("Index + Retrieval")
        index_path = st.text_input("Index path", ".rag/ore_algebra_both_index.json")
        k = st.slider("Top-k", min_value=1, max_value=20, value=6)
        mode = st.selectbox("Retrieval mode", ["auto", "hybrid", "dense", "lexical"], index=0)
        hybrid_alpha = st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, value=0.7)
        source_priority = st.selectbox("Source priority", ["auto", "symbols-first", "flat"], index=0)
        symbols_ratio = st.slider("Symbols ratio", min_value=0.0, max_value=1.0, value=0.75)
        max_pdf_extras = st.slider("Max PDF extras", min_value=0, max_value=10, value=2)
        pdf_char_limit = st.slider("PDF chars per context", min_value=400, max_value=3000, value=1600)

        st.header("LLM")
        provider = st.selectbox("Provider", ["openai", "gemini"], index=0)
        default_model = "gpt-4.1-mini" if provider == "openai" else "gemini-1.5-pro"
        model = st.text_input("Model", default_model)
        api_key_label = (
            "OPENAI_API_KEY (optional; else env OPENAI_API_KEY)"
            if provider == "openai"
            else "GEMINI_API_KEY (optional; else env GEMINI_API_KEY/GOOGLE_API_KEY)"
        )
        api_key = st.text_input(api_key_label, type="password")
        has_default_key = (
            bool(os.getenv("OPENAI_API_KEY"))
            if provider == "openai"
            else bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        )
        if not api_key and has_default_key:
            st.caption("No key entered. Using key from `.env`/environment.")
        elif not api_key:
            st.caption("No key entered and no default key detected in `.env`/environment.")

    question = st.chat_input("Ask a question about ore_algebra...")
    if not question:
        st.info("Enter a question to run retrieval + LLM answering.")
        return

    idx = Path(index_path).expanduser().resolve()
    if not idx.exists():
        st.error(f"Index file not found: {idx}")
        return

    try:
        with st.spinner("Loading index and retrieving context..."):
            payload = load_index(idx)
            chunks = parse_chunks(payload)
            mode_used, results = select_retrieval(
                index_payload=payload,
                chunks=chunks,
                query=question,
                k=k,
                mode=mode,
                index_path=idx,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
            )
            context_items = _to_context_items(results=results, pdf_char_limit=pdf_char_limit)
    except Exception as exc:
        st.error(f"Retrieval failed: {exc}")
        return

    st.success(f"Retrieved {len(results)} contexts using `{mode_used}` mode.")
    _render_retrieval_results(results=results, pdf_char_limit=pdf_char_limit)

    request = LLMRequest(
        question=question,
        contexts=context_items,
        provider=provider,
        model=model,
        temperature=0.0,
    )

    try:
        with st.spinner("Calling LLM..."):
            response = answer_with_llm(request=request, api_key=api_key or None)
    except Exception as exc:
        st.error(f"LLM request failed: {exc}")
        return

    st.subheader("Answer")
    st.markdown(response.answer or "(No answer returned)")

    if response.code.strip():
        st.subheader("Generated Code")
        st.code(response.code, language="python")

    st.subheader("Citations Used")
    st.code(_citation_lines(response, context_items), language="text")

    if response.missing_info:
        st.subheader("Missing Info")
        st.write("\n".join(f"- {item}" for item in response.missing_info))


if __name__ == "__main__":
    main()
