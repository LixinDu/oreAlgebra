#!/usr/bin/env python3
"""RAG assistant for the ore_algebra user guide PDF.

Features:
- Extracts PDF text with page numbers.
- Detects likely section titles.
- Chunks pages with overlap while preserving page metadata.
- Builds a persisted hybrid index: FAISS vector store + lexical TF-IDF.
- Uses dense, lexical, or hybrid retrieval (hybrid default).
- Generates SageMath code using an LLM (OpenAI optional).
- Optionally verifies and repairs code by running `sage -python`.
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
from pathlib import Path
from typing import Dict, List, Optional, Tuple


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
    page_start: int
    page_end: int
    section_title: str
    source: str


@dataclass
class RetrievalResult:
    chunk_id: int
    score: float
    page_start: int
    page_end: int
    section_title: str
    source: str
    text: str


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
                page_start=pages[start].page,
                page_end=pages[end - 1].page,
                section_title=section,
                source=source,
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
        out.append(
            RetrievalResult(
                chunk_id=c.chunk_id,
                score=float(score),
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                source=c.source,
                text=c.text,
            )
        )
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
            "python3 ore_rag_assistant.py build-index --pdf data/ore_algebra_guide.pdf"
        )
    return dense


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
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Dense retrieval requires sentence-transformers + numpy. "
            "Install with: pip install sentence-transformers numpy"
        ) from exc

    model = SentenceTransformer(model_name)
    q_emb = model.encode([query], normalize_embeddings=True)
    q_vec = np.asarray(q_emb, dtype="float32")

    ranked: List[Tuple[int, float]] = []
    if isinstance(faiss_file_name, str) and faiss_file_name:
        try:
            import faiss  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "Dense retrieval requires FAISS for vector-store lookup. "
                "Install with: pip install faiss-cpu"
            ) from exc

        faiss_file = index_path.parent / faiss_file_name
        if not faiss_file.exists():
            raise RuntimeError(f"FAISS index file not found: {faiss_file}")

        vs_index = faiss.read_index(str(faiss_file))
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
        doc_emb = np.load(str(emb_file))
        scores = (doc_emb @ q_vec[0]).tolist()
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:k]

    out: List[RetrievalResult] = []
    for idx, score in ranked:
        c = chunks[idx]
        out.append(
            RetrievalResult(
                chunk_id=c.chunk_id,
                score=float(score),
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                source=c.source,
                text=c.text,
            )
        )
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

    out: List[RetrievalResult] = []
    for idx, score in top:
        c = chunks[idx]
        out.append(
            RetrievalResult(
                chunk_id=c.chunk_id,
                score=float(score),
                page_start=c.page_start,
                page_end=c.page_end,
                section_title=c.section_title,
                source=c.source,
                text=c.text,
            )
        )
    return out


def select_retrieval(
    index_payload: Dict[str, object],
    chunks: List[Chunk],
    query: str,
    k: int,
    mode: str,
    index_path: Path,
    hybrid_alpha: float,
) -> Tuple[str, List[RetrievalResult]]:
    if not (0.0 <= hybrid_alpha <= 1.0):
        raise RuntimeError(f"--hybrid-alpha must be between 0 and 1, got {hybrid_alpha}")
    if mode == "lexical":
        return "lexical", lexical_search(index_payload, chunks, query, k)
    if mode == "dense":
        return "dense", dense_search(index_payload, chunks, query, k, index_path)
    if mode == "hybrid" or mode == "auto":
        return "hybrid", hybrid_search(
            index_payload=index_payload,
            chunks=chunks,
            query=query,
            k=k,
            index_path=index_path,
            alpha=hybrid_alpha,
        )
    raise RuntimeError(f"Unsupported retrieval mode: {mode}")


def build_context_block(results: List[RetrievalResult], max_chars_per_chunk: int = 2200) -> str:
    blocks = []
    for r in results:
        snippet = r.text[:max_chars_per_chunk]
        section = r.section_title or "(section unknown)"
        blocks.append(
            "\n".join(
                [
                    f"[CHUNK {r.chunk_id}]",
                    f"Source: {r.source}",
                    f"Pages: {r.page_start}-{r.page_end}",
                    f"Section: {section}",
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
2) A short note (3-6 lines) listing citations in the format:
   - pp. <start>-<end>, section: <title or unknown>
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
2) A short citation list in this format:
   - pp. <start>-<end>, section: <title or unknown>
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
        key = (r.page_start, r.page_end, r.section_title)
        if key in seen:
            continue
        seen.add(key)
        section = r.section_title or "unknown"
        lines.append(f"- pp. {r.page_start}-{r.page_end}, section: {section}")
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
    return [Chunk(**item) for item in raw]


def cmd_build_index(args: argparse.Namespace) -> int:
    pdf_path = Path(args.pdf).resolve()
    index_path = Path(args.index_path).resolve()

    if not pdf_path.exists():
        print(f"ERROR: PDF not found: {pdf_path}", file=sys.stderr)
        return 2

    pages = extract_pages(pdf_path)
    chunks = chunk_pages(
        pages=pages,
        source=pdf_path.name,
        chunk_chars=args.chunk_chars,
        overlap_chars=args.overlap_chars,
    )

    payload: Dict[str, object] = {
        "version": 1,
        "created_at": utc_now_iso(),
        "pdf_path": str(pdf_path),
        "chunking": {
            "chunk_chars": args.chunk_chars,
            "overlap_chars": args.overlap_chars,
        },
        "chunks": [asdict(c) for c in chunks],
        "lexical": build_lexical_payload(chunks),
    }

    try:
        import faiss  # type: ignore
        import numpy as np  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Embedding vector-store build requires sentence-transformers, numpy, and faiss-cpu. "
            "Install with: pip install sentence-transformers numpy faiss-cpu"
        ) from exc

    model = SentenceTransformer(args.dense_model)
    texts = [c.text for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    emb_name = index_path.with_suffix(index_path.suffix + ".dense.npy").name
    emb_path = index_path.parent / emb_name
    index_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(emb_path), emb)

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
    print(f"Pages: {len(pages)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Dense model: {payload['dense']['model']}")  # type: ignore[index]
    print(f"Vector store: {payload['dense']['faiss_index_file']}")  # type: ignore[index]
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
    )

    print(f"Retrieval mode: {mode_used}")
    print(f"Question: {args.question}\n")
    for r in results:
        section = r.section_title or "unknown"
        preview = r.text[:300].replace("\n", " ")
        print(
            f"- chunk={r.chunk_id} score={r.score:.4f} pages={r.page_start}-{r.page_end} "
            f"section={section}\n  preview: {preview}\n"
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
    p = argparse.ArgumentParser(description="RAG assistant for ore_algebra PDF -> SageMath code")
    sub = p.add_subparsers(dest="cmd")

    b = sub.add_parser("build-index", help="Extract and index PDF chunks")
    b.add_argument("--pdf", default="data/ore_algebra_guide.pdf", help="Path to ore_algebra guide PDF")
    b.add_argument("--index-path", default=".rag/ore_algebra_index.json", help="Where to write index JSON")
    b.add_argument("--chunk-chars", type=int, default=3500)
    b.add_argument("--overlap-chars", type=int, default=400)
    b.add_argument(
        "--dense-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for embedding vector-store build",
    )
    b.set_defaults(func=cmd_build_index)

    r = sub.add_parser("retrieve", help="Retrieve relevant chunks for a question")
    r.add_argument("--index-path", default=".rag/ore_algebra_index.json")
    r.add_argument("--question", required=True)
    r.add_argument("--k", type=int, default=6)
    r.add_argument("--mode", choices=["auto", "hybrid", "dense", "lexical"], default="hybrid")
    r.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.7,
        help="Hybrid weight for dense score in [0,1]. Final score = a*dense + (1-a)*lexical",
    )
    r.set_defaults(func=cmd_retrieve)

    a = sub.add_parser("answer", help="Generate Sage code from retrieved PDF context")
    a.add_argument("--index-path", default=".rag/ore_algebra_index.json")
    a.add_argument("--question", required=True)
    a.add_argument("--k", type=int, default=6)
    a.add_argument("--mode", choices=["auto", "hybrid", "dense", "lexical"], default="hybrid")
    a.add_argument(
        "--hybrid-alpha",
        type=float,
        default=0.7,
        help="Hybrid weight for dense score in [0,1]. Final score = a*dense + (1-a)*lexical",
    )
    a.add_argument("--provider", default="openai", choices=["openai"])
    a.add_argument("--model", default="gpt-4.1-mini")
    a.add_argument("--verify", action="store_true", help="Run generated code with sage -python and auto-repair")
    a.add_argument("--max-repairs", type=int, default=2)
    a.add_argument("--sage-bin", default="sage")
    a.add_argument("--output-script", default="")
    a.set_defaults(func=cmd_answer)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "cmd", None):
        parser.print_help(sys.stderr)
        print(
            "\nExamples:\n"
            "  bash scripts/download_ore_guide.sh\n"
            "  python3 ore_rag_assistant.py build-index --pdf data/ore_algebra_guide.pdf\n"
            "  python3 ore_rag_assistant.py retrieve --question \"shift Ore algebra over QQ[n]\"\n"
            "  python3 ore_rag_assistant.py answer --question \"Define Sn over QQ[n]\" --verify\n",
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
