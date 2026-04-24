#!/usr/bin/env python3
"""Vector-store organizer for ore_algebra docs.

Features:
- Extracts and chunks PDF text with page metadata.
- Extracts symbol-level docs from generated JSONL metadata.
- Builds a persisted hybrid index: FAISS vector store + lexical TF-IDF.
- Retrieval/generation helpers are available as importable functions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from retrieval.knowledge_base import (
    default_generated_api_md_path,
    default_generated_symbols_path,
    default_index_path_for_mode,
    default_pdf_inputs,
    load_knowledge_base_profile,
)


TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
SECTION_RES = (
    re.compile(r"^\s*(\d+(?:\.\d+)*)\s+(.{3,120})\s*$"),
    re.compile(r"^\s*Chapter\s+\d+\s*[:.-]?\s*(.{3,120})\s*$", re.IGNORECASE),
)


@dataclass
class Page:
    page: int
    text: str
    section_title: str


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    source_type: str = "pdf"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: str = ""
    symbol_id: str = ""
    module: str = ""
    qualname: str = ""
    signature: str = ""
    kind: str = ""
    file_path: str = ""
    line: Optional[int] = None
    summary: str = ""
    example_count: int = 0


@dataclass
class RetrievalResult:
    chunk_id: int
    score: float
    source: str
    text: str
    source_type: str = "pdf"
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    section_title: str = ""
    symbol_id: str = ""
    module: str = ""
    qualname: str = ""
    signature: str = ""
    kind: str = ""
    file_path: str = ""
    line: Optional[int] = None
    summary: str = ""
    example_count: int = 0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def detect_section_title(page_text: str) -> str:
    lines = [line.strip() for line in page_text.splitlines()[:35] if line.strip()]
    for line in lines:
        if len(line) > 140:
            continue
        if line.isupper() and len(line.split()) > 10:
            continue
        for pattern in SECTION_RES:
            m = pattern.match(line)
            if m:
                if pattern is SECTION_RES[0]:
                    return f"{m.group(1)} {m.group(2).strip()}"
                return line
    return ""


def extract_pages(pdf_path: Path) -> List[Page]:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Missing dependency `pypdf`. Install with: pip install pypdf"
        ) from exc

    reader = PdfReader(str(pdf_path))
    pages: List[Page] = []
    current_section = ""

    for idx, p in enumerate(reader.pages, start=1):
        text = p.extract_text() or ""
        sec = detect_section_title(text)
        if sec:
            current_section = sec
        pages.append(Page(page=idx, text=text, section_title=current_section))

    return pages


def chunk_pages(
    pages: List[Page],
    source: str,
    chunk_chars: int = 3500,
    overlap_chars: int = 400,
) -> List[Chunk]:
    chunks: List[Chunk] = []
    i = 0
    chunk_id = 0

    while i < len(pages):
        start = i
        end = i
        total = 0

        while end < len(pages):
            p = pages[end]
            block_len = len(p.text) + 20
            if end > start and total + block_len > chunk_chars:
                break
            total += block_len
            end += 1
            if total >= chunk_chars:
                break

        text = "\n\n".join(
            f"[PAGE {p.page}]\n{p.text}" for p in pages[start:end]
        )
        section = ""
        for p in pages[start:end]:
            if p.section_title:
                section = p.section_title
                break

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=text,
                source=source,
                source_type="pdf",
                page_start=pages[start].page,
                page_end=pages[end - 1].page,
                section_title=section,
            )
        )
        chunk_id += 1

        if end >= len(pages):
            break

        rewind_pages = 0
        rewind_chars = 0
        j = end - 1
        while j >= start and rewind_chars < overlap_chars:
            rewind_chars += len(pages[j].text)
            rewind_pages += 1
            j -= 1

        i = end - rewind_pages
        if i <= start:
            i = start + 1

    return chunks


def _safe_int(value: object) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_str(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_examples_list(value: object) -> List[str]:
    if not isinstance(value, list):
        return []
    out: List[str] = []
    for item in value:
        s = _safe_str(item)
        if s:
            out.append(s)
    return out


def extract_generated_symbol_chunks(symbols_path: Path) -> List[Chunk]:
    if not symbols_path.exists():
        raise FileNotFoundError(f"Generated symbols file not found: {symbols_path}")

    chunks: List[Chunk] = []
    for line_no, raw_line in enumerate(symbols_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Invalid JSON at {symbols_path}:{line_no}: {exc.msg}"
            ) from exc
        if not isinstance(rec, dict):
            continue

        symbol_id = _safe_str(rec.get("id"))
        module = _safe_str(rec.get("module"))
        qualname = _safe_str(rec.get("qualname"))
        signature = _safe_str(rec.get("signature"))
        kind = _safe_str(rec.get("kind"))
        file_path = _safe_str(rec.get("file_path"))
        source_line = _safe_int(rec.get("line"))
        summary = _safe_str(rec.get("summary"))
        docstring = _safe_str(rec.get("docstring"))
        examples = _safe_examples_list(rec.get("examples"))
        example_count = _safe_int(rec.get("example_count")) or len(examples)

        display_name = qualname or symbol_id or "unknown_symbol"
        text_parts = [
            f"Symbol: {display_name}",
            f"ID: {symbol_id or display_name}",
            f"Module: {module or 'unknown'}",
            f"Kind: {kind or 'unknown'}",
            f"Signature: {signature or 'unknown'}",
        ]
        if summary:
            text_parts.append(f"Summary: {summary}")
        if docstring:
            text_parts.append(f"Docstring:\n{docstring}")
        if examples:
            rendered_examples = "\n".join(f"- {ex}" for ex in examples[:10])
            text_parts.append(f"Examples:\n{rendered_examples}")

        # Repeat key tokens once to improve lexical matching for API names/signatures.
        search_keys = " ".join(
            part for part in [symbol_id, qualname, signature, module] if part
        )
        if search_keys:
            text_parts.append(f"Search keys: {search_keys}")

        chunks.append(
            Chunk(
                chunk_id=len(chunks),
                text="\n".join(text_parts).strip(),
                source=symbols_path.name,
                source_type="generated",
                section_title=module,
                symbol_id=symbol_id,
                module=module,
                qualname=qualname,
                signature=signature,
                kind=kind,
                file_path=file_path,
                line=source_line,
                summary=summary,
                example_count=example_count,
            )
        )

    if not chunks:
        raise RuntimeError(f"No symbol records parsed from: {symbols_path}")
    return chunks


def extract_api_reference_chunks(api_reference_path: Path, chunk_chars: int = 2400) -> List[Chunk]:
    if not api_reference_path.exists():
        return []

    text = api_reference_path.read_text(encoding="utf-8")
    if not text.strip():
        return []

    sections: List[Tuple[str, str]] = []
    current_title = "API Reference"
    current_lines: List[str] = []
    for line in text.splitlines():
        if line.startswith("## "):
            if current_lines:
                sections.append((current_title, "\n".join(current_lines).strip()))
            current_title = line.lstrip("#").strip().strip("`")
            current_lines = [line]
            continue
        current_lines.append(line)
    if current_lines:
        sections.append((current_title, "\n".join(current_lines).strip()))

    chunks: List[Chunk] = []
    for title, body in sections:
        if not body:
            continue
        start = 0
        while start < len(body):
            end = min(len(body), start + chunk_chars)
            chunk_text = body[start:end].strip()
            if chunk_text:
                chunks.append(
                    Chunk(
                        chunk_id=len(chunks),
                        text=chunk_text,
                        source=api_reference_path.name,
                        source_type="generated",
                        section_title=title,
                        module=title,
                        kind="module_reference",
                    )
                )
            if end >= len(body):
                break
            start = max(end - 250, start + 1)

    return chunks


def extract_narrative_chunks(
    narratives_path: Path,
    chunk_chars: int = 3500,
    overlap_chars: int = 400,
) -> List[Chunk]:
    """Load narrative sections from module_narratives.jsonl into Chunk objects.

    Long narrative sections are split into overlapping sub-chunks while
    preserving section metadata and cross-reference information.
    """
    if not narratives_path.exists():
        return []

    chunks: List[Chunk] = []
    for raw_line in narratives_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if not isinstance(rec, dict):
            continue

        section_id = _safe_str(rec.get("section_id"))
        module = _safe_str(rec.get("module"))
        section_title = _safe_str(rec.get("section_title"))
        text = _safe_str(rec.get("text"))
        source_file = _safe_str(rec.get("source_file"))
        resolved_refs = rec.get("resolved_symbol_ids", [])
        if not isinstance(resolved_refs, list):
            resolved_refs = []

        if not text or not section_id:
            continue

        # Build chunk text with metadata header for retrieval
        header_parts = [
            f"Narrative: {section_title}",
            f"Module: {module or 'unknown'}",
            f"Source: {source_file}" if source_file else "",
        ]
        if resolved_refs:
            header_parts.append(f"Related symbols: {', '.join(resolved_refs[:15])}")
        header = "\n".join(p for p in header_parts if p)

        full_text = f"{header}\n\n{text}"

        # Split long narratives into overlapping sub-chunks
        if len(full_text) <= chunk_chars:
            chunks.append(
                Chunk(
                    chunk_id=len(chunks),
                    text=full_text,
                    source=narratives_path.name,
                    source_type="narrative",
                    section_title=section_title,
                    symbol_id=section_id,
                    module=module,
                    kind="narrative",
                    file_path=source_file,
                )
            )
        else:
            # Split on paragraph boundaries where possible
            start = 0
            sub_idx = 0
            while start < len(full_text):
                end = min(len(full_text), start + chunk_chars)
                # Try to break at a paragraph boundary
                if end < len(full_text):
                    newline_pos = full_text.rfind("\n\n", start + chunk_chars // 2, end)
                    if newline_pos > start:
                        end = newline_pos
                chunk_text = full_text[start:end].strip()
                if chunk_text:
                    sub_id = f"{section_id}:{sub_idx}" if sub_idx > 0 else section_id
                    chunks.append(
                        Chunk(
                            chunk_id=len(chunks),
                            text=chunk_text,
                            source=narratives_path.name,
                            source_type="narrative",
                            section_title=section_title,
                            symbol_id=sub_id,
                            module=module,
                            kind="narrative",
                            file_path=source_file,
                        )
                    )
                    sub_idx += 1
                if end >= len(full_text):
                    break
                start = max(end - overlap_chars, start + 1)

    return chunks


def chunk_to_result(chunk: Chunk, score: float) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        score=float(score),
        source=chunk.source,
        text=chunk.text,
        source_type=chunk.source_type,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_title=chunk.section_title,
        symbol_id=chunk.symbol_id,
        module=chunk.module,
        qualname=chunk.qualname,
        signature=chunk.signature,
        kind=chunk.kind,
        file_path=chunk.file_path,
        line=chunk.line,
        summary=chunk.summary,
        example_count=chunk.example_count,
    )


def location_label(result: RetrievalResult) -> str:
    if result.file_path:
        if result.line is not None:
            return f"{result.file_path}:{result.line}"
        return result.file_path
    if result.source_type == "pdf":
        if result.page_start is not None and result.page_end is not None:
            return f"pp. {result.page_start}-{result.page_end}"
        if result.page_start is not None:
            return f"p. {result.page_start}"
    return result.source


def build_lexical_payload(chunks: List[Chunk]) -> Dict[str, object]:
    docs_tokens: List[List[str]] = [tokenize(c.text) for c in chunks]
    n_docs = len(docs_tokens)

    df: Dict[str, int] = {}
    for tokens in docs_tokens:
        seen = set(tokens)
        for t in seen:
            df[t] = df.get(t, 0) + 1

    idf: Dict[str, float] = {
        t: math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for t, d in df.items()
    }

    doc_vectors: List[Dict[str, float]] = []
    doc_norms: List[float] = []

    for tokens in docs_tokens:
        tf: Dict[str, int] = {}
        for t in tokens:
            tf[t] = tf.get(t, 0) + 1
        vec: Dict[str, float] = {}
        norm_sq = 0.0
        for t, c in tf.items():
            w = (1.0 + math.log(c)) * idf.get(t, 0.0)
            vec[t] = w
            norm_sq += w * w
        doc_vectors.append(vec)
        doc_norms.append(math.sqrt(norm_sq) if norm_sq > 0 else 1.0)

    return {
        "idf": idf,
        "doc_vectors": doc_vectors,
        "doc_norms": doc_norms,
    }


def lexical_search(
    index_payload: Dict[str, object],
    chunks: List[Chunk],
    query: str,
    k: int,
) -> List[RetrievalResult]:
    lexical = index_payload["lexical"]  # type: ignore[index]
    idf: Dict[str, float] = lexical["idf"]  # type: ignore[index]
    doc_vectors: List[Dict[str, float]] = lexical["doc_vectors"]  # type: ignore[index]
    doc_norms: List[float] = lexical["doc_norms"]  # type: ignore[index]

    q_tokens = tokenize(query)
    q_tf: Dict[str, int] = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1

    q_vec: Dict[str, float] = {}
    q_norm_sq = 0.0
    for t, c in q_tf.items():
        if t not in idf:
            continue
        w = (1.0 + math.log(c)) * idf[t]
        q_vec[t] = w
        q_norm_sq += w * w

    q_norm = math.sqrt(q_norm_sq) if q_norm_sq > 0 else 1.0

    scored: List[Tuple[int, float]] = []
    for i, dvec in enumerate(doc_vectors):
        dot = 0.0
        for t, q_w in q_vec.items():
            dot += q_w * dvec.get(t, 0.0)
        score = dot / (q_norm * doc_norms[i])
        if score > 0:
            scored.append((i, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]

    out: List[RetrievalResult] = []
    for idx, score in top:
        c = chunks[idx]
        out.append(chunk_to_result(c, score))
    return out


def has_dense(index_payload: Dict[str, object]) -> bool:
    dense = index_payload.get("dense")
    if not isinstance(dense, dict):
        return False
    has_model = "model" in dense
    has_faiss = "faiss_index_file" in dense
    has_legacy_embeddings = "embeddings_file" in dense
    return has_model and (has_faiss or has_legacy_embeddings)


def require_dense_index(index_payload: Dict[str, object], mode: str) -> Dict[str, object]:
    dense = index_payload.get("dense")
    if not has_dense(index_payload) or not isinstance(dense, dict):
        raise RuntimeError(
            f"{mode} retrieval requires embeddings + vector store metadata. "
            "Rebuild index with: "
            "python3 ore_rag_assistant.py build-index --source-mode both "
            "--pdf data/ore_algebra_guide.pdf --pdf data/ore_algebra_guide_multivariate.pdf "
            "(or use --mode lexical)."
        )
    return dense


@lru_cache(maxsize=4)
def load_sentence_transformer_model(model_name: str):
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers + numpy. "
            "Install with: pip install sentence-transformers numpy"
        ) from exc
    return SentenceTransformer(model_name)


@lru_cache(maxsize=16)
def load_faiss_index_file(index_file: str):
    try:
        import faiss  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dense retrieval requires FAISS for vector-store lookup. "
            "Install with: pip install faiss-cpu"
        ) from exc
    return faiss.read_index(index_file)


@lru_cache(maxsize=16)
def load_dense_embeddings_file(emb_file: str):
    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers + numpy. "
            "Install with: pip install sentence-transformers numpy"
        ) from exc
    return np.load(emb_file)


def dense_search(
    index_payload: Dict[str, object],
    chunks: List[Chunk],
    query: str,
    k: int,
    index_path: Path,
) -> List[RetrievalResult]:
    dense = require_dense_index(index_payload, mode="dense")
    model_name = str(dense["model"])
    faiss_file_name = dense.get("faiss_index_file")
    emb_file_name = dense.get("embeddings_file")

    try:
        import numpy as np  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers + numpy. "
            "Install with: pip install sentence-transformers numpy"
        ) from exc

    model = load_sentence_transformer_model(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_emb, dtype="float32")

    ranked: List[Tuple[int, float]] = []
    if isinstance(faiss_file_name, str) and faiss_file_name:
        faiss_file = index_path.parent / faiss_file_name
        if not faiss_file.exists():
            raise RuntimeError(f"FAISS index file not found: {faiss_file}")

        vs_index = load_faiss_index_file(str(faiss_file))
        scores, ids = vs_index.search(q_vec, k)
        for idx, score in zip(ids[0], scores[0]):
            if int(idx) < 0:
                continue
            ranked.append((int(idx), float(score)))
    else:
        # Backward compatibility for old indexes that only persisted embeddings.
        if not isinstance(emb_file_name, str) or not emb_file_name:
            raise RuntimeError(
                "Dense index metadata is incomplete (missing faiss_index_file/embeddings_file)."
            )
        emb_file = index_path.parent / emb_file_name
        if not emb_file.exists():
            raise RuntimeError(f"Dense embeddings file not found: {emb_file}")
        doc_emb = load_dense_embeddings_file(str(emb_file))
        scores = (doc_emb @ q_vec[0]).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    out: List[RetrievalResult] = []
    for idx, score in ranked:
        c = chunks[idx]
        out.append(chunk_to_result(c, score))
    return out


def _normalize_score_map(score_map: Dict[int, float]) -> Dict[int, float]:
    if not score_map:
        return {}
    values = list(score_map.values())
    lo = min(values)
    hi = max(values)
    if hi - lo < 1e-12:
        return {k: 1.0 for k in score_map}
    return {k: (v - lo) / (hi - lo) for k, v in score_map.items()}


def has_source(chunks: List[Chunk], source_type: str) -> bool:
    return any(c.source_type == source_type for c in chunks)


def dedupe_results(results: List[RetrievalResult]) -> List[RetrievalResult]:
    seen = set()
    out: List[RetrievalResult] = []
    for r in results:
        if r.chunk_id in seen:
            continue
        seen.add(r.chunk_id)
        out.append(r)
    return out


def apply_source_priority(
    results: List[RetrievalResult],
    k: int,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
    index_has_generated: bool,
    index_has_pdf: bool,
) -> List[RetrievalResult]:
    if source_priority not in {"auto", "flat", "symbols-first"}:
        raise RuntimeError(f"Unsupported source-priority: {source_priority}")
    if not (0.0 <= symbols_ratio <= 1.0):
        raise RuntimeError(f"--symbols-ratio must be between 0 and 1, got {symbols_ratio}")
    if max_pdf_extras < 0:
        raise RuntimeError(f"--max-pdf-extras must be >= 0, got {max_pdf_extras}")

    if source_priority == "auto":
        if index_has_generated and index_has_pdf:
            source_priority = "symbols-first"
        else:
            source_priority = "flat"

    ranked = dedupe_results(results)
    if source_priority == "flat":
        return ranked[:k]

    generated = [r for r in ranked if r.source_type == "generated"]
    pdf = [r for r in ranked if r.source_type == "pdf"]
    other = [r for r in ranked if r.source_type not in {"generated", "pdf"}]

    if not generated:
        return ranked[:k]

    target_symbols = int(round(k * symbols_ratio))
    if target_symbols <= 0:
        target_symbols = 1
    target_symbols = min(target_symbols, len(generated), k)

    selected: List[RetrievalResult] = []
    selected.extend(generated[:target_symbols])
    remaining = k - len(selected)

    if remaining > 0 and pdf:
        take_pdf = min(remaining, len(pdf), max_pdf_extras)
        selected.extend(pdf[:take_pdf])
        remaining = k - len(selected)

    if remaining > 0:
        selected.extend(generated[target_symbols:target_symbols + remaining])
        remaining = k - len(selected)

    if remaining > 0 and pdf:
        already = {r.chunk_id for r in selected}
        pdf_rest = [r for r in pdf if r.chunk_id not in already]
        selected.extend(pdf_rest[:remaining])
        remaining = k - len(selected)

    if remaining > 0 and other:
        already = {r.chunk_id for r in selected}
        other_rest = [r for r in other if r.chunk_id not in already]
        selected.extend(other_rest[:remaining])

    return dedupe_results(selected)[:k]


def hybrid_search(
    index_payload: Dict[str, object],
    chunks: List[Chunk],
    query: str,
    k: int,
    index_path: Path,
    alpha: float,
) -> List[RetrievalResult]:
    require_dense_index(index_payload, mode="hybrid")
    fetch_k = max(k * 4, 20)
    dense_results = dense_search(index_payload, chunks, query, fetch_k, index_path)
    lexical_results = lexical_search(index_payload, chunks, query, fetch_k)

    dense_map = _normalize_score_map({r.chunk_id: r.score for r in dense_results})
    lexical_map = _normalize_score_map({r.chunk_id: r.score for r in lexical_results})

    all_ids = set(dense_map.keys()) | set(lexical_map.keys())
    combined: List[Tuple[int, float]] = []
    for chunk_id in all_ids:
        d = dense_map.get(chunk_id, 0.0)
        l = lexical_map.get(chunk_id, 0.0)
        combined_score = alpha * d + (1.0 - alpha) * l
        combined.append((chunk_id, combined_score))

    combined.sort(key=lambda x: x[1], reverse=True)
    top = combined[:k]
    chunk_by_id = {c.chunk_id: c for c in chunks}

    out: List[RetrievalResult] = []
    for idx, score in top:
        c = chunk_by_id.get(idx)
        if c is None:
            continue
        out.append(chunk_to_result(c, score))
    return out


def select_retrieval(
    index_payload: Dict[str, object],
    chunks: List[Chunk],
    query: str,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
    source_priority: str,
    symbols_ratio: float,
    max_pdf_extras: int,
) -> Tuple[str, List[RetrievalResult]]:
    if not (0.0 <= hybrid_alpha <= 1.0):
        raise RuntimeError(f"--hybrid-alpha must be between 0 and 1, got {hybrid_alpha}")
    candidate_k = max(k * 5, 25)
    requested_mode = mode
    if mode == "auto":
        if has_dense(index_payload):
            mode = "hybrid"
        else:
            mode = "lexical"
    mode_used = mode
    try:
        if mode == "lexical":
            candidates = lexical_search(index_payload, chunks, query, candidate_k)
        elif mode == "dense":
            candidates = dense_search(index_payload, chunks, query, candidate_k, index_path)
        elif mode == "hybrid":
            candidates = hybrid_search(
                index_payload=index_payload,
                chunks=chunks,
                query=query,
                k=candidate_k,
                index_path=index_path,
                alpha=hybrid_alpha,
            )
        else:
            raise RuntimeError(f"Unsupported retrieval mode: {mode}")
    except RuntimeError as exc:
        dense_runtime_markers = (
            "requires embeddings + vector store metadata",
            "Dense retrieval requires sentence-transformers + numpy",
            "Dense retrieval requires FAISS",
        )
        if requested_mode in {"auto", "hybrid"} and any(marker in str(exc) for marker in dense_runtime_markers):
            mode_used = "lexical"
            candidates = lexical_search(index_payload, chunks, query, candidate_k)
        else:
            raise

    results = apply_source_priority(
        results=candidates,
        k=k,
        source_priority=source_priority,
        symbols_ratio=symbols_ratio,
        max_pdf_extras=max_pdf_extras,
        index_has_generated=has_source(chunks, "generated"),
        index_has_pdf=has_source(chunks, "pdf"),
    )
    return mode_used, results


def build_context_block(
    results: List[RetrievalResult],
    max_chars_per_pdf_chunk: int = 2200,
    max_chars_per_generated_chunk: int = 0,
) -> str:
    blocks = []
    for r in results:
        if r.source_type == "generated":
            if max_chars_per_generated_chunk > 0:
                snippet = r.text[:max_chars_per_generated_chunk]
            else:
                snippet = r.text
        else:
            snippet = r.text[:max_chars_per_pdf_chunk]
        meta_lines = [
            f"[CHUNK {r.chunk_id}]",
            f"Source: {r.source} ({r.source_type})",
        ]
        if r.source_type == "pdf":
            section = r.section_title or "(section unknown)"
            if r.page_start is not None and r.page_end is not None:
                meta_lines.append(f"Pages: {r.page_start}-{r.page_end}")
            elif r.page_start is not None:
                meta_lines.append(f"Page: {r.page_start}")
            meta_lines.append(f"Section: {section}")
        else:
            symbol = r.qualname or r.symbol_id
            if symbol:
                meta_lines.append(f"Symbol: {symbol}")
            if r.signature:
                meta_lines.append(f"Signature: {r.signature}")
            if r.module:
                meta_lines.append(f"Module: {r.module}")
            if r.kind:
                meta_lines.append(f"Kind: {r.kind}")
            meta_lines.append(f"Location: {location_label(r)}")

        blocks.append(
            "\n".join(
                meta_lines
                + [
                    "Text:",
                    snippet,
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def build_generation_prompt(question: str, results: List[RetrievalResult]) -> str:
    context = build_context_block(results)
    return f"""You write runnable SageMath code for ore_algebra.
Use only APIs and syntax that appear in the provided context.
If required details are missing from context, say exactly what is missing.

Question:
{question}

Context:
{context}

Output format:
1) One fenced code block with complete SageMath code.
2) A short note (3-6 lines) listing citations from the context, for example:
   - pp. <start>-<end>, section: <title or unknown>
   - symbol: <name>, source: <file_path:line>
"""


def build_repair_prompt(
    question: str,
    previous_code: str,
    stderr_text: str,
    results: List[RetrievalResult],
) -> str:
    context = build_context_block(results)
    return f"""Fix the SageMath script so it runs successfully with `sage -python`.
Only use ore_algebra syntax and methods supported by the context.

Original question:
{question}

Current code:
```python
{previous_code}
```

Execution error:
{stderr_text}

Context:
{context}

Output format:
1) One fenced code block with corrected complete SageMath code.
2) A short citation list based on the provided context.
"""


def call_openai(prompt: str, model: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    try:
        from openai import OpenAI  # type: ignore
    except Exception as exc:
        raise RuntimeError("Missing dependency `openai`. Install with: pip install openai") from exc

    client = OpenAI(api_key=api_key)

    # New API path
    try:
        resp = client.responses.create(model=model, input=prompt)
        txt = getattr(resp, "output_text", None)
        if txt:
            return txt
    except Exception:
        pass

    # Backward-compatible fallback
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a precise SageMath coding assistant."},
            {"role": "user", "content": prompt},
        ],
    )
    content = resp.choices[0].message.content
    if isinstance(content, str):
        return content
    return str(content)


def extract_code_block(text: str) -> str:
    m = re.search(r"```(?:python|sage|sagemath)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return text.strip()


def run_sage_code(code: str, sage_bin: str, timeout_sec: int = 60) -> Tuple[bool, str, str]:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        script_path = f.name

    try:
        proc = subprocess.run(
            [sage_bin, "-python", script_path],
            text=True,
            capture_output=True,
            timeout=timeout_sec,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    ok = proc.returncode == 0
    return ok, proc.stdout, proc.stderr


def format_citations(results: List[RetrievalResult]) -> str:
    seen = set()
    lines = []
    for r in results:
        if r.source_type == "pdf":
            key = ("pdf", r.source, r.page_start, r.page_end, r.section_title)
            if key in seen:
                continue
            seen.add(key)
            section = r.section_title or "unknown"
            if r.page_start is not None and r.page_end is not None:
                lines.append(f"- pp. {r.page_start}-{r.page_end}, section: {section}")
            elif r.page_start is not None:
                lines.append(f"- p. {r.page_start}, section: {section}")
            else:
                lines.append(f"- source: {r.source}, section: {section}")
            continue

        symbol = r.qualname or r.symbol_id or "unknown_symbol"
        loc = location_label(r)
        key = ("generated", symbol, loc)
        if key in seen:
            continue
        seen.add(key)
        if r.signature:
            lines.append(f"- symbol: {symbol}, signature: {r.signature}, source: {loc}")
        else:
            lines.append(f"- symbol: {symbol}, source: {loc}")
    return "\n".join(lines)


def save_index(index_path: Path, payload: Dict[str, object]) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")


def load_index(index_path: Path) -> Dict[str, object]:
    if not index_path.exists():
        raise FileNotFoundError(f"Index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def parse_chunks(index_payload: Dict[str, object]) -> List[Chunk]:
    raw = index_payload.get("chunks")
    if not isinstance(raw, list):
        raise RuntimeError("Index is missing `chunks`.")
    chunks: List[Chunk] = []
    for idx, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        page_start = _safe_int(item.get("page_start"))
        page_end = _safe_int(item.get("page_end"))
        source_type = _safe_str(item.get("source_type"))
        if not source_type:
            source_type = "pdf" if (page_start is not None or page_end is not None) else "generated"
        chunks.append(
            Chunk(
                chunk_id=_safe_int(item.get("chunk_id")) or idx,
                text=_safe_str(item.get("text")),
                source=_safe_str(item.get("source")) or "unknown",
                source_type=source_type,
                page_start=page_start,
                page_end=page_end,
                section_title=_safe_str(item.get("section_title")),
                symbol_id=_safe_str(item.get("symbol_id")),
                module=_safe_str(item.get("module")),
                qualname=_safe_str(item.get("qualname")),
                signature=_safe_str(item.get("signature")),
                kind=_safe_str(item.get("kind")),
                file_path=_safe_str(item.get("file_path")),
                line=_safe_int(item.get("line")),
                summary=_safe_str(item.get("summary")),
                example_count=_safe_int(item.get("example_count")) or 0,
            )
        )
    return chunks


def _resolve_pdf_inputs(value: object) -> list[Path]:
    raw_values: list[str] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            text = str(item).strip()
            if text:
                raw_values.append(text)
    else:
        text = str(value or "").strip()
        if text:
            raw_values.append(text)

    if not raw_values:
        raw_values = list(default_pdf_inputs())

    resolved: list[Path] = []
    seen: set[str] = set()
    for raw in raw_values:
        path = Path(raw).resolve()
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        resolved.append(path)
    return resolved


def cmd_build_index(args: argparse.Namespace) -> int:
    index_path = Path(args.index_path).resolve()
    source_mode = args.source_mode
    chunks: List[Chunk] = []
    pages_count = 0
    source_meta: Dict[str, object] = {"mode": source_mode}

    # Symbols-first organization: keep generated API docs as the primary layer.
    if source_mode in {"generated", "both"}:
        symbols_path = Path(args.generated_symbols).resolve()
        try:
            generated_chunks = extract_generated_symbol_chunks(symbols_path)
        except FileNotFoundError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        chunks.extend(generated_chunks)
        source_meta["generated_symbols"] = str(symbols_path)

        if args.include_generated_api_md:
            api_path = Path(args.generated_api_md).resolve()
            md_chunks = extract_api_reference_chunks(
                api_reference_path=api_path,
                chunk_chars=args.generated_chunk_chars,
            )
            chunks.extend(md_chunks)
            source_meta["generated_api_md"] = str(api_path)

    # Narrative tutorial chunks (module-level docstrings with cross-references)
    narratives_arg = getattr(args, "narratives", None)
    if narratives_arg and source_mode in {"generated", "both"}:
        narratives_path = Path(narratives_arg).resolve()
        if narratives_path.exists():
            narrative_chunks = extract_narrative_chunks(
                narratives_path,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
            )
            chunks.extend(narrative_chunks)
            source_meta["narratives"] = str(narratives_path)

    if source_mode in {"pdf", "both"}:
        pdf_paths = _resolve_pdf_inputs(getattr(args, "pdf", None))
        pdf_path_strings: list[str] = []
        for pdf_path in pdf_paths:
            if not pdf_path.exists():
                print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
                return 2
            pages = extract_pages(pdf_path)
            pages_count += len(pages)
            pdf_chunks = chunk_pages(
                pages=pages,
                source=pdf_path.name,
                chunk_chars=args.chunk_chars,
                overlap_chars=args.overlap_chars,
            )
            chunks.extend(pdf_chunks)
            pdf_path_strings.append(str(pdf_path))
        source_meta["pdf_paths"] = pdf_path_strings
        if len(pdf_path_strings) == 1:
            source_meta["pdf_path"] = pdf_path_strings[0]

    if not chunks:
        print("ERROR: No chunks were produced for indexing.", file=sys.stderr)
        return 2

    # Keep ids contiguous to make retrieval bookkeeping stable.
    for idx, c in enumerate(chunks):
        c.chunk_id = idx

    payload: Dict[str, object] = {
        "version": 2,
        "created_at": utc_now_iso(),
        "sources": source_meta,
        "retrieval_defaults": {
            "source_priority": "symbols-first",
            "symbols_ratio": 0.75,
            "max_pdf_extras": 2,
        },
        "chunking": {
            "chunk_chars": args.chunk_chars,
            "overlap_chars": args.overlap_chars,
            "generated_chunk_chars": args.generated_chunk_chars,
        },
        "chunks": [asdict(c) for c in chunks],
        "lexical": build_lexical_payload(chunks),
    }

    if not args.no_dense:
        try:
            import numpy as np  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Embedding vector-store build requires sentence-transformers and numpy. "
                "Install with: pip install sentence-transformers numpy faiss-cpu "
                "(or rebuild with --no-dense for lexical-only indexing)."
            ) from exc

        model = SentenceTransformer(args.dense_model)
        texts = [c.text for c in chunks]
        # Large one-shot encodes have been unstable in our local env; keep
        # batches small so dense index rebuilds remain reliable. Also defer the
        # faiss import until after embedding generation, because importing it
        # earlier has caused native-process crashes in this environment.
        batch_size = 32
        emb_batches = []
        for start in range(0, len(texts), batch_size):
            batch = model.encode(
                texts[start:start + batch_size],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            emb_batches.append(np.asarray(batch, dtype="float32"))
        emb = np.concatenate(emb_batches, axis=0) if emb_batches else np.empty((0, 0), dtype="float32")

        emb_name = index_path.with_suffix(index_path.suffix + ".dense.npy").name
        emb_path = index_path.parent / emb_name
        index_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(emb_path), emb)

        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Embedding vector-store build requires faiss-cpu for index writing. "
                "Install with: pip install faiss-cpu "
                "(or rebuild with --no-dense for lexical-only indexing)."
            ) from exc

        dim = emb.shape[1]
        vs_index = faiss.IndexFlatIP(dim)
        vs_index.add(emb)
        faiss_name = index_path.with_suffix(index_path.suffix + ".faiss").name
        faiss_path = index_path.parent / faiss_name
        faiss.write_index(vs_index, str(faiss_path))

        payload["dense"] = {
            "model": args.dense_model,
            "metric": "inner_product",
            "embeddings_file": emb_name,
            "faiss_index_file": faiss_name,
        }

    save_index(index_path, payload)

    print(f"Index written: {index_path}")
    if pages_count:
        print(f"Pages: {pages_count}")
    pdf_chunks_count = sum(1 for c in chunks if c.source_type == "pdf")
    narrative_chunks_count = sum(1 for c in chunks if c.source_type == "narrative")
    generated_chunks_count = len(chunks) - pdf_chunks_count - narrative_chunks_count
    print(
        f"Chunks: {len(chunks)} "
        f"(pdf={pdf_chunks_count}, generated={generated_chunks_count}, narrative={narrative_chunks_count})"
    )
    print(f"Source mode: {source_mode}")
    if "dense" in payload:
        print(f"Dense model: {payload['dense']['model']}")  # type: ignore[index]
        print(f"Vector store: {payload['dense']['faiss_index_file']}")  # type: ignore[index]
    else:
        print("Dense index: disabled (--no-dense); retrieval mode must be lexical.")
    return 0


def cmd_retrieve(args: argparse.Namespace) -> int:
    index_path = Path(args.index_path).resolve()
    payload = load_index(index_path)
    chunks = parse_chunks(payload)

    mode_used, results = select_retrieval(
        index_payload=payload,
        chunks=chunks,
        query=args.question,
        k=args.k,
        mode=args.mode,
        index_path=index_path,
        hybrid_alpha=args.hybrid_alpha,
        source_priority=args.source_priority,
        symbols_ratio=args.symbols_ratio,
        max_pdf_extras=args.max_pdf_extras,
    )

    print(f"Retrieval mode: {mode_used}")
    print(f"Question: {args.question}\n")
    for r in results:
        if r.source_type == "pdf":
            preview = r.text[:300].replace("\n", " ")
            section = r.section_title or "unknown"
            pages = (
                f"{r.page_start}-{r.page_end}"
                if r.page_start is not None and r.page_end is not None
                else str(r.page_start or "?")
            )
            print(
                f"- chunk={r.chunk_id} score={r.score:.4f} source=pdf pages={pages} "
                f"section={section}\n  preview: {preview}\n"
            )
        else:
            preview = r.text.replace("\n", " ")
            symbol = r.qualname or r.symbol_id or "unknown_symbol"
            print(
                f"- chunk={r.chunk_id} score={r.score:.4f} source=generated symbol={symbol} "
                f"location={location_label(r)}\n  preview: {preview}\n"
            )
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    index_path = Path(args.index_path).resolve()
    payload = load_index(index_path)
    chunks = parse_chunks(payload)

    mode_used, results = select_retrieval(
        index_payload=payload,
        chunks=chunks,
        query=args.question,
        k=args.k,
        mode=args.mode,
        index_path=index_path,
        hybrid_alpha=args.hybrid_alpha,
        source_priority=args.source_priority,
        symbols_ratio=args.symbols_ratio,
        max_pdf_extras=args.max_pdf_extras,
    )

    prompt = build_generation_prompt(args.question, results)

    if args.provider != "openai":
        print("ERROR: Only `openai` provider is currently supported.", file=sys.stderr)
        return 2

    try:
        raw = call_openai(prompt=prompt, model=args.model)
    except Exception as exc:
        print("LLM call failed.")
        print(f"Reason: {exc}")
        print("\nYou can still use this prompt manually:\n")
        print(prompt)
        print("\nCitations for this retrieval pass:")
        print(format_citations(results))
        return 1

    code = extract_code_block(raw)
    verify_ok = None
    verify_stdout = ""
    verify_stderr = ""

    if args.verify:
        verify_ok, verify_stdout, verify_stderr = run_sage_code(code, sage_bin=args.sage_bin)

        repairs_done = 0
        while not verify_ok and repairs_done < args.max_repairs:
            repairs_done += 1
            error_for_retrieval = f"{args.question}\n\nError:\n{verify_stderr[-3000:]}"
            _, repair_results = select_retrieval(
                index_payload=payload,
                chunks=chunks,
                query=error_for_retrieval,
                k=args.k,
                mode=args.mode,
                index_path=index_path,
                hybrid_alpha=args.hybrid_alpha,
                source_priority=args.source_priority,
                symbols_ratio=args.symbols_ratio,
                max_pdf_extras=args.max_pdf_extras,
            )
            repair_prompt = build_repair_prompt(
                question=args.question,
                previous_code=code,
                stderr_text=verify_stderr[-3000:],
                results=repair_results,
            )
            raw = call_openai(prompt=repair_prompt, model=args.model)
            code = extract_code_block(raw)
            results = repair_results
            verify_ok, verify_stdout, verify_stderr = run_sage_code(code, sage_bin=args.sage_bin)

    if args.output_script:
        out_path = Path(args.output_script)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(code + "\n", encoding="utf-8")
        print(f"Script written: {out_path}")

    print("\n=== SageMath Code ===\n")
    print(code)
    print("\n=== Citations ===")
    print(format_citations(results))
    print("\n=== Retrieval Mode ===")
    print(mode_used)

    if args.verify:
        print("\n=== Verification ===")
        print("pass" if verify_ok else "fail")
        if verify_stdout.strip():
            print("\n--- stdout ---")
            print(verify_stdout[-3000:])
        if verify_stderr.strip():
            print("\n--- stderr ---")
            print(verify_stderr[-3000:])

    return 0 if (verify_ok is not False) else 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Vector-store organizer for ore_algebra docs")
    sub = p.add_subparsers(dest="cmd")

    kb_profile = load_knowledge_base_profile()
    b = sub.add_parser("build-index", help="Extract and index docs from PDF/generated sources")
    b.add_argument("--source-mode", choices=["pdf", "generated", "both"], default=kb_profile.default_source_mode)
    b.add_argument(
        "--pdf",
        action="append",
        default=None,
        help=(
            "Path to an ore_algebra guide PDF. Repeat to include multiple PDFs; "
            "defaults to both checked-in guides in data/."
        ),
    )
    b.add_argument(
        "--generated-symbols",
        default=default_generated_symbols_path(),
        help="Path to generated symbol JSONL",
    )
    b.add_argument(
        "--generated-api-md",
        default=default_generated_api_md_path(),
        help="Path to generated API markdown",
    )
    b.add_argument(
        "--include-generated-api-md",
        action="store_true",
        help="Include API_REFERENCE markdown sections in the index",
    )
    b.add_argument(
        "--index-path",
        default=default_index_path_for_mode(kb_profile.default_source_mode),
        help="Where to write index JSON",
    )
    b.add_argument(
        "--narratives",
        default=None,
        help="Path to module_narratives.jsonl (narrative tutorial chunks)",
    )
    b.add_argument("--chunk-chars", type=int, default=3500)
    b.add_argument("--overlap-chars", type=int, default=400)
    b.add_argument("--generated-chunk-chars", type=int, default=2400)
    b.add_argument(
        "--dense-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embedding vector-store build",
    )
    b.add_argument(
        "--no-dense",
        action="store_true",
        help="Skip embedding/FAISS build and keep a lexical-only index",
    )
    b.set_defaults(func=cmd_build_index)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help(sys.stderr)
        print(
            "\nExamples:\n"
            "  bash scripts/download_ore_guide.sh\n"
            "  python3 ore_rag_assistant.py build-index --source-mode pdf\n"
            "  python3 ore_rag_assistant.py build-index --source-mode generated\n"
            "  python3 ore_rag_assistant.py build-index --source-mode both --include-generated-api-md\n"
            "  python3 scripts/refresh_knowledge_base.py\n"
            "  streamlit run streamlit_app.py\n",
            file=sys.stderr,
        )
        return 2
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
