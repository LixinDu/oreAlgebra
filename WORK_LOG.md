# Work Log  

## Friday, 13 February 2026

## Scope
- Build a local RAG assistant over `kauers14b.pdf` to generate runnable SageMath `ore_algebra` code with citations.

## Completed
- Implemented CLI in `ore_rag_assistant.py` with subcommands:
  - `build-index`
  - `retrieve`
  - `answer`
- Added PDF ingestion with page tracking and section-title heuristics.
- Added chunking with overlap and page metadata.
- Added retrieval and citation plumbing for generation prompts.
- Added Sage execution/repair loop (`sage -python`) for runtime verification.
- Improved no-argument CLI behavior to print help + examples.

## Retrieval Upgrade (Embedding Required)
- Added embedding-based indexing using `sentence-transformers`.
- Added FAISS vector store persistence (`.faiss`) and embedding array persistence (`.dense.npy`).
- Added dense retrieval via FAISS.
- Added hybrid retrieval (dense + lexical TF-IDF weighted merge).
- Set hybrid as default retrieval mode for `retrieve` and `answer`.
- Added `--hybrid-alpha` tuning option.

## Files Added/Updated
- Added: `ore_rag_assistant.py`
- Added: `README.md`
- Added: `requirements.txt`
- Added: `WORK_LOG.md`
- Generated locally during testing: `.rag/mock_index.json`

## Dependency/Runtime Notes
- Required packages now include:
  - `pypdf`
  - `openai`
  - `sentence-transformers`
  - `numpy`
  - `faiss-cpu`
- In the restricted environment used during development, online `pip install` failed (network resolution), so full end-to-end run against real PDF indexing was blocked there.

## Current Recommended Run
```bash
pip install -r requirements.txt

bash scripts/download_ore_guide.sh

python3 ore_rag_assistant.py build-index --pdf data/ore_algebra_guide.pdf --index-path .rag/ore_algebra_index.json --dense-model all-MiniLM-L6-v2

python3 ore_rag_assistant.py retrieve --index-path .rag/ore_algebra_index.json --question "how to compute LCLM" --mode hybrid --hybrid-alpha 0.25 --k 10

python3 ore_rag_assistant.py answer --index-path .rag/ore_algebra_index.json --question "how to compute LCLM" --mode hybrid --hybrid-alpha 0.25 --verify
```

## Next
- If needed, add automatic query expansion for domain terms (e.g., `LCLM`, `least common left multiple`) before retrieval.

## Saturday, 14 February 2026
## Update (PDF Handling)
- Added `.gitignore` rules to keep external PDFs out of git:
  - `data/*.pdf`
- Added downloader script: `scripts/download_ore_guide.sh`
- Added PDF source manifest: `data/PDF_SOURCE.txt`
- Added repo placeholders:
  - `data/.gitkeep`
  - `indexes/.gitkeep`
- Updated defaults/docs to use `data/ore_algebra_guide.pdf` instead of a committed root PDF.
