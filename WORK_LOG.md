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

## Sunday, 15 February 2026
## Update (Source Organization + Retrieval App Split)
- Reworked source organization to treat generated package docs as primary:
  - Added ingestion for `generated/symbols.jsonl` (symbol-level chunks).
  - Added optional ingestion for `generated/API_REFERENCE.md`.
  - Added source modes: `pdf`, `generated`, `both`.
- Added source-aware chunk metadata and citations:
  - Generated chunks now carry symbol/module/signature/location fields.
  - PDF chunks keep page/section metadata.
- Added mixed-source retrieval policy controls:
  - `--source-priority {auto,flat,symbols-first}`
  - `--symbols-ratio`
  - `--max-pdf-extras`
- Added symbols-first organization/retrieval defaults for combined indexes.
- Updated retrieval/context behavior:
  - Generated symbol content is preserved in full.
  - PDF content remains truncated for compact support context.

## Update (App/CLI Responsibilities)
- Restricted `ore_rag_assistant.py` CLI to vector-store organization only:
  - Kept `build-index`.
  - Removed `retrieve` and `answer` subcommands from CLI surface.
- Added separate retrieval UI app:
  - `streamlit_app.py` performs retrieval in background from built indexes.
  - Displays retrieved contexts and citations.
- Removed OpenAI requirement from retrieval app workflow.

## Dependency Notes (Current)
- Requirements now focus on indexing + retrieval:
  - `pypdf`
  - `openai`
  - `google-generativeai`
  - `python-dotenv`
  - `sentence-transformers`
  - `numpy`
  - `faiss-cpu`
  - `streamlit`

## Current Recommended Run (Latest)
```bash
source .venv/bin/activate
pip install -r requirements.txt

# Build combined index (generated docs + PDF support)
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --include-generated-api-md \
  --generated-api-md generated/API_REFERENCE.md \
  --index-path .rag/ore_algebra_both_index.json

# Start retrieval UI
streamlit run streamlit_app.py
```

## Update (LLM Chat App + Multi-Provider)
- Added LLM service layer: `llm_service.py`
  - Structured request/response schema for question + retrieved context.
  - Prompt template enforces context-grounded answering and citation IDs.
  - Added ore_algebra generator naming/commutation rule block in prompt.
- Added chat UI: `streamlit_chat_app.py`
  - Retrieves top-k context from built index.
  - Sends context + question to selected provider/model.
  - Renders answer, optional code, citations, and missing-info notes.
- Added provider choice:
  - `openai`
  - `gemini`
- Added key fallback behavior:
  - UI key is optional.
  - If blank, app uses environment/.env keys.
  - Supported key names:
    - OpenAI: `OPENAI_API_KEY`
    - Gemini: `GEMINI_API_KEY` (fallback: `GOOGLE_API_KEY`)
- Added `.env` template for local key management.

## Current Recommended Run (Chat)
```bash
source .venv/bin/activate
pip install -r requirements.txt

# Build combined index (generated docs + PDF support)
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --include-generated-api-md \
  --generated-api-md generated/API_REFERENCE.md \
  --index-path .rag/ore_algebra_both_index.json

# Start LLM chat UI
streamlit run streamlit_chat_app.py
```

## Update (Agentic Step Loop + Final Synthesis Streaming)
- Updated `streamlit_chat_app.py` to execute the step workflow:
  - plan subtasks,
  - retrieve per step,
  - decide next action,
  - then run final synthesis for answer/code.
- Added final synthesis context aggregation:
  - deduplicate retrieved chunks across steps,
  - cap by configurable final context count.
- Added visible streaming status for final synthesis in Streamlit:
  - shows live provider output while final answer is being generated.
- Added question persistence in UI:
  - latest submitted question stays visible after input.

## Update (LLM Runtime Controls and Defaults)
- Added UI temperature control in `streamlit_chat_app.py` (default `0.1`).
- Propagated temperature through planning, step-decision, and final synthesis calls.
- Changed chat defaults to:
  - provider: `openai`
  - model: `gpt-4o-mini`
- Important runtime note:
  - no automatic retry/fallback is implemented when a model rejects explicit `temperature`;
    choose a compatible model/temperature in the UI.

## Current Recommended Run (Planner + Synthesis Chat)
```bash
source .venv/bin/activate
pip install -r requirements.txt

# Build combined index (generated docs + PDF support)
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --include-generated-api-md \
  --generated-api-md generated/API_REFERENCE.md \
  --index-path .rag/ore_algebra_both_index.json

# Start planner/synthesis chat UI
streamlit run streamlit_chat_app.py
```

## Update (Docs Sync for Current Runtime)
- Updated `ARCHITECTURE_NOTE.md` and `WORK_LOG.md` to match current behavior.
- Clarified temperature handling:
  - default UI temperature is `0.1`,
  - model compatibility is user-controlled from UI,
  - there is no automatic provider temperature fallback in code.
