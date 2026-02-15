#!/usr/bin/env python3
"""Streamlit agent app: plan -> retrieve per step -> decide -> final synthesis."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

import streamlit as st

from llm_service import ContextItem, LLMRequest, LLMResponse, Subtask, answer_with_llm, decide_next_action, plan_subtasks
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


def _dedupe_results_by_chunk(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    out: List[RetrievalResult] = []
    for r in results:
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
    return out


def _citation_lines(response: LLMResponse, context_items: List[ContextItem]) -> str:
    ctx_by_id = {c.context_id: c for c in context_items}
    lines: List[str] = []
    for cid in response.citations_used:
        item = ctx_by_id.get(cid)
        if not item:
            continue
        lines.append(f"- {cid}: {item.title} ({item.location})")
    if not lines:
        return "No citations returned."
    return "\n".join(lines)


def _render_retrieval_results(results: List[RetrievalResult], pdf_char_limit: int, key_prefix: str) -> None:
    st.subheader("Retrieved Context")
    for idx, r in enumerate(results):
        with st.expander(_result_title(r), expanded=False):
            st.write(f"source_type: `{r.source_type}`")
            st.write(f"source: `{r.source}`")
            if r.source_type == "generated":
                if r.signature:
                    st.write(f"signature: `{r.signature}`")
                if r.module:
                    st.write(f"module: `{r.module}`")
                st.write(f"location: `{location_label(r)}`")
                st.text_area(
                    "content",
                    value=r.text,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-gen",
                )
            else:
                section = r.section_title or "unknown"
                st.write(f"pages: `{r.page_start}-{r.page_end}`")
                st.write(f"section: `{section}`")
                preview = r.text[:pdf_char_limit] if len(r.text) > pdf_char_limit else r.text
                st.text_area(
                    "content",
                    value=preview,
                    height=260,
                    key=f"text-{key_prefix}-{idx}-{r.chunk_id}-pdf",
                )


def _run_retrieval_for_query(
    *,
    query: str,
    payload: dict,
    chunks: list,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
) -> tuple[str, List[RetrievalResult]]:
    return select_retrieval(
        index_payload=payload,
        chunks=chunks,
        query=query,
        k=k,
        mode=mode,
        index_path=index_path,
        hybrid_alpha=hybrid_alpha,
        source_priority=source_priority,
        symbols_ratio=symbols_ratio,
        max_pdf_extras=max_pdf_extras,
    )


def _render_step_header(step: Subtask, step_query: str) -> None:
    st.markdown(f"### Step {step.step_id}: {step.title}")
    st.write(f"instruction: {step.instruction}")
    st.write(f"query: `{step_query}`")


def main() -> None:
    st.set_page_config(page_title="ore_algebra Planner", layout="wide")
    st.title("ore_algebra Planner")
    st.caption("Implements: plan subtasks -> retrieve per step -> decide next action -> final synthesis.")

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
        max_plan_steps = st.slider("Max plan steps", min_value=1, max_value=10, value=4)
        final_context_limit = st.slider("Final synthesis context count", min_value=2, max_value=30, value=10)

        st.header("LLM")
        provider = st.selectbox("Provider", ["openai", "gemini"], index=0)
        default_model = "gpt-4o-mini" if provider == "openai" else "gemini-2.5-flash"
        model = st.text_input("Model", default_model)
        temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05)
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
    if question:
        st.session_state["last_question"] = question

    if not question:
        last_question = st.session_state.get("last_question", "")
        if last_question:
            st.subheader("Last Question")
            st.markdown(last_question)
        st.info("Enter a question to run retrieval + LLM answering.")
        return

    st.subheader("User Question")
    st.markdown(question)

    idx = Path(index_path).expanduser().resolve()
    if not idx.exists():
        st.error(f"Index file not found: {idx}")
        return

    try:
        with st.spinner("Loading index..."):
            payload = load_index(idx)
            chunks = parse_chunks(payload)
    except Exception as exc:
        st.error(f"Index load failed: {exc}")
        return

    try:
        with st.spinner("Planning subtasks..."):
            plan = plan_subtasks(
                question=question,
                provider=provider,
                model=model,
                api_key=api_key or None,
                max_steps=max_plan_steps,
                temperature=temperature,
            )
    except Exception as exc:
        st.error(f"Planning failed: {exc}")
        return

    st.subheader("Plan")
    if not plan.subtasks:
        st.warning("Planner returned no subtasks.")
        return

    aggregated_results: List[RetrievalResult] = []

    for subtask in plan.subtasks:
        step_query = subtask.retrieval_query or subtask.instruction or subtask.title
        _render_step_header(subtask, step_query=step_query)

        mode_used, results = _run_retrieval_for_query(
            query=step_query,
            payload=payload,
            chunks=chunks,
            k=k,
            mode=mode,
            index_path=idx,
            hybrid_alpha=hybrid_alpha,
            source_priority=source_priority,
            symbols_ratio=symbols_ratio,
            max_pdf_extras=max_pdf_extras,
        )
        st.write(f"retrieval mode: `{mode_used}`, results: `{len(results)}`")
        _render_retrieval_results(
            results=results,
            pdf_char_limit=pdf_char_limit,
            key_prefix=f"step-{subtask.step_id}-base",
        )
        context_items = _to_context_items(results=results, pdf_char_limit=pdf_char_limit)

        decision = decide_next_action(
            question=question,
            current_step=subtask,
            context_items=context_items,
            provider=provider,
            model=model,
            api_key=api_key or None,
            temperature=temperature,
        )
        st.write(f"next action: `{decision.action}`")
        if decision.reason:
            st.write(f"reason: {decision.reason}")
        st.write(f"confidence: {decision.confidence:.2f}")

        step_effective_results = results
        if decision.action == "refine_query" and decision.next_query.strip():
            st.write(f"refine query: `{decision.next_query}`")
            mode_used2, refined_results = _run_retrieval_for_query(
                query=decision.next_query.strip(),
                payload=payload,
                chunks=chunks,
                k=k,
                mode=mode,
                index_path=idx,
                hybrid_alpha=hybrid_alpha,
                source_priority=source_priority,
                symbols_ratio=symbols_ratio,
                max_pdf_extras=max_pdf_extras,
            )
            st.write(f"refined retrieval mode: `{mode_used2}`, results: `{len(refined_results)}`")
            _render_retrieval_results(
                results=refined_results,
                pdf_char_limit=pdf_char_limit,
                key_prefix=f"step-{subtask.step_id}-refined",
            )
            step_effective_results = refined_results

        aggregated_results.extend(step_effective_results)

        if decision.action == "stop":
            st.info("Workflow stopped by agent decision.")
            break

    aggregated_results = _dedupe_results_by_chunk(aggregated_results)[:final_context_limit]
    if not aggregated_results:
        st.warning("No aggregated context available for final synthesis.")
        return

    final_context_items = _to_context_items(
        results=aggregated_results,
        pdf_char_limit=pdf_char_limit,
    )
    request = LLMRequest(
        question=question,
        contexts=final_context_items,
        provider=provider,
        model=model,
        temperature=temperature,
    )

    try:
        with st.status("Final synthesis with LLM...", expanded=True) as status:
            live_placeholder = st.empty()

            def _on_chunk(_piece: str, acc_text: str) -> None:
                # Show the live JSON stream coming from the provider.
                live_placeholder.code(acc_text[-6000:], language="json")

            final_response = answer_with_llm(
                request=request,
                api_key=api_key or None,
                stream=True,
                on_chunk=_on_chunk,
            )
            status.update(label="Final synthesis complete", state="complete")
    except Exception as exc:
        st.error(f"Final synthesis failed: {exc}")
        return

    st.subheader("Final Answer")
    st.markdown(final_response.answer or "(No answer returned)")

    if final_response.code.strip():
        st.subheader("Generated Code")
        st.code(final_response.code, language="python")

    st.subheader("Final Citations")
    st.code(_citation_lines(final_response, final_context_items), language="text")

    if final_response.missing_info:
        st.subheader("Missing Info")
        st.write("\n".join(f"- {item}" for item in final_response.missing_info))


if __name__ == "__main__":
    main()
