# Architecture Note

## Purpose
This project organizes vector stores for `ore_algebra` documentation and provides a separate retrieval UI. The index builder ingests package docs from `generated/` and optional PDF material, builds hybrid retrieval artifacts (dense + lexical), and stores source-aware metadata for retrieval-time fusion.

## High-Level Architecture
1. Ingestion
- Input PDF is loaded with `pypdf`.
- Each page is extracted and tagged with page number.
- A heuristic section-title detector tracks likely section headers.
- Generated package docs are loaded from `generated/symbols.jsonl` (symbol-level records).
- Optional generated markdown reference chunks can be loaded from `generated/API_REFERENCE.md`.

2. Chunking
- Pages are grouped into overlapping chunks (`chunk_chars`, `overlap_chars`).
- Each chunk stores page range, section title (best effort), and source.
- Symbol docs are chunked as one record per symbol (no page chunking).
- Combined indexes are organized symbols-first, then PDF support chunks.

3. Indexing
- Lexical index: TF-IDF-like sparse vectors built from tokenized chunk text.
- Dense index: sentence-transformers embeddings + FAISS inner-product index.
- Metadata and lexical payload are saved in `.json`; dense artifacts are saved in `.dense.npy` and `.faiss`.
- `--no-dense` is supported for lexical-only index builds.

4. Retrieval
- Modes: `lexical`, `dense`, `hybrid`, `auto`.
- Hybrid combines normalized dense/lexical scores:
  `score = alpha * dense + (1 - alpha) * lexical`.
- Source-priority fusion supports `auto`, `flat`, and `symbols-first`.
- Default mixed-source policy prioritizes generated symbols and adds limited PDF extras.

5. Runtime Apps
- `ore_rag_assistant.py` CLI is restricted to vector-store organization (`build-index`).
- `streamlit_app.py` is a separate retrieval-only UI that runs retrieval in background and displays context/citations.

## Repository Structure
```text
oreAlgebra/
  ore_rag_assistant.py          # Index builder CLI + shared retrieval functions
  streamlit_app.py              # Retrieval-only Streamlit application
  README.md                     # User-facing quickstart
  ARCHITECTURE_NOTE.md          # Architecture + maintenance notes
  WORK_LOG.md                   # Development/change log
  requirements.txt              # Python dependencies
  scripts/
    download_ore_guide.sh       # Downloads ore_algebra guide PDF
  data/
    PDF_SOURCE.txt              # Source URL manifest for PDF
  generated/
    symbols.jsonl               # Extracted ore_algebra symbol docs
    API_REFERENCE.md            # Generated package API summary
  .rag/                         # Generated index artifacts (local runtime output)
```

## Current Workflow
1. Create virtual environment and install dependencies.
2. Prepare source docs:
- generated package docs in `generated/symbols.jsonl` (required for generated mode).
- optional PDF in `data/ore_algebra_guide.pdf`.
3. Build index with `ore_rag_assistant.py build-index` (`generated`, `pdf`, or `both`).
4. Start `streamlit_app.py`.
5. Run retrieval queries in the app and inspect source-aware citations.

## Run Commands
```bash
# 1) setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2) optional: get source PDF
bash scripts/download_ore_guide.sh

# 3) build index (generated docs only)
python3 ore_rag_assistant.py build-index \
  --source-mode generated \
  --generated-symbols generated/symbols.jsonl \
  --index-path .rag/ore_algebra_generated_index.json \
  --dense-model all-MiniLM-L6-v2

# 4) build index (generated + PDF)
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --include-generated-api-md \
  --generated-api-md generated/API_REFERENCE.md \
  --index-path .rag/ore_algebra_both_index.json \
  --dense-model all-MiniLM-L6-v2

# 5) retrieval UI
streamlit run streamlit_app.py

# 6) optional lexical-only index (if dense model download is blocked)
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --index-path .rag/ore_algebra_both_index.json \
  --no-dense
```

## Potential Problems
- Dependency friction: `faiss-cpu` and `sentence-transformers` can be platform-sensitive.
- Dense-model availability: first-time embedding model download may fail in restricted/offline environments.
- Source-quality variance: generated symbol docs may have sparse summaries for some APIs.
- Citation quality limits: section detection is heuristic and may miss or mislabel section titles.
- Retrieval misses: source-priority and `--hybrid-alpha` may need tuning by query type.
- Context volume: full symbol text can be large; UI and downstream consumers should manage token/length budgets.

## Regular Updates
- Last updated: 2026-02-15
- Update frequency: whenever workflow, dependencies, index format, or CLI flags change.
- Update checklist:
  - Confirm run commands still work as written.
  - Confirm default file paths match code defaults.
  - Record major behavior changes in `WORK_LOG.md`.
  - Bump this date and summarize the change in 2-5 bullets.

## Maintainer Edit Policy
- Routine updates go only in `Update Entries` below (append new item at top).
- Do not edit these sections for routine updates: `Purpose`, `High-Level Architecture`, `Repository Structure`, `Current Workflow`, `Run Commands`, `Potential Problems`.
- Edit those core sections only when architecture or behavior has actually changed.
- If core sections change, also update `WORK_LOG.md` in the same commit.

## Update Entries (Append Only)
<!-- UPDATE_ENTRIES_START -->
- 2026-02-15: Reworked architecture to index-builder + retrieval-only Streamlit app; added generated-doc source model and symbols-first retrieval policy notes.
- 2026-02-14: Added maintainer edit policy and append-only update log markers.
<!-- UPDATE_ENTRIES_END -->
