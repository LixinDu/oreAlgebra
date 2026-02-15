#!/usr/bin/env python3
"""Streamlit UI for retrieval over prebuilt ore_algebra vector stores."""

from __future__ import annotations

from pathlib import Path
from typing import List

import streamlit as st

from ore_rag_assistant import (
    format_citations,
    load_index,
    location_label,
    parse_chunks,
    select_retrieval,
)
from ore_rag_assistant import RetrievalResult


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


def _render_retrieval_results(results: List[RetrievalResult]) -> None:
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
                preview = r.text[:2200] if len(r.text) > 2200 else r.text
                st.text_area("content", value=preview, height=260, key=f"text-{r.chunk_id}")


def main() -> None:
    st.set_page_config(page_title="ore_algebra Retrieval", layout="wide")
    st.title("ore_algebra Retrieval")
    st.caption("Build indexes with ore_rag_assistant.py, then use this app for retrieval only.")

    with st.sidebar:
        st.header("Index + Retrieval")
        index_path = st.text_input("Index path", ".rag/ore_algebra_both_index.json")
        question = st.text_area("Question", height=120)
        k = st.slider("Top-k", min_value=1, max_value=20, value=6)
        mode = st.selectbox("Retrieval mode", ["auto", "hybrid", "dense", "lexical"], index=0)
        hybrid_alpha = st.slider("Hybrid alpha", min_value=0.0, max_value=1.0, value=0.7)

        source_priority = st.selectbox(
            "Source priority",
            ["auto", "symbols-first", "flat"],
            index=0,
        )
        symbols_ratio = st.slider("Symbols ratio", min_value=0.0, max_value=1.0, value=0.75)
        max_pdf_extras = st.slider("Max PDF extras", min_value=0, max_value=10, value=2)

        run_btn = st.button("Run Retrieval", type="primary")

    if not run_btn:
        st.info("Configure options and click 'Run Retrieval'.")
        return

    if not question.strip():
        st.error("Question is required.")
        return

    idx = Path(index_path).expanduser().resolve()
    if not idx.exists():
        st.error(f"Index file not found: {idx}")
        return

    with st.spinner("Loading index and retrieving context..."):
        payload = load_index(idx)
        chunks = parse_chunks(payload)
        retrieval_args = {
            "index_payload": payload,
            "chunks": chunks,
            "k": k,
            "mode": mode,
            "index_path": idx,
            "hybrid_alpha": hybrid_alpha,
            "source_priority": source_priority,
            "symbols_ratio": symbols_ratio,
            "max_pdf_extras": max_pdf_extras,
        }
        mode_used, results = select_retrieval(query=question, **retrieval_args)

    st.success(f"Retrieved {len(results)} chunks using `{mode_used}` mode.")
    _render_retrieval_results(results)

    st.subheader("Citations")
    st.code(format_citations(results), language="text")


if __name__ == "__main__":
    main()
