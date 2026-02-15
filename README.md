# ore_algebra Agentic RAG Assistant

This project builds a local index over `ore_algebra` documentation sources and provides two Streamlit apps:
- retrieval-only inspection (`streamlit_app.py`),
- agentic chat (`streamlit_chat_app.py`) with the loop:
  `plan subtasks -> retrieve per step -> decide next action -> final synthesis`.

Primary source is `generated/symbols.jsonl` (package symbol docs), with optional support from `data/ore_algebra_guide.pdf`.

## Repository Roles

- `ore_rag_assistant.py`: index builder CLI (`build-index`) + shared retrieval/index utilities.
- `streamlit_app.py`: retrieval UI over prebuilt indexes.
- `streamlit_chat_app.py`: retrieval + LLM synthesis UI (OpenAI/Gemini).
- `llm_service.py`: provider calls, prompts, parsing for planning/decision/final synthesis.
- `scripts/download_ore_guide.sh`: fetches external guide PDF.

## Install

Use a virtual environment (recommended, especially on macOS/Homebrew Python):

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Data Inputs

- Generated symbol docs (recommended): `generated/symbols.jsonl`
- Optional generated API markdown: `generated/API_REFERENCE.md`
- Optional PDF guide: `data/ore_algebra_guide.pdf`

Download PDF (optional):

```bash
bash scripts/download_ore_guide.sh
```

## Build Index

### Generated docs only

```bash
python3 ore_rag_assistant.py build-index \
  --source-mode generated \
  --generated-symbols generated/symbols.jsonl \
  --index-path .rag/ore_algebra_generated_index.json
```

### Generated + PDF (recommended)

```bash
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --include-generated-api-md \
  --generated-api-md generated/API_REFERENCE.md \
  --index-path .rag/ore_algebra_both_index.json
```

### PDF only

```bash
python3 ore_rag_assistant.py build-index \
  --source-mode pdf \
  --pdf data/ore_algebra_guide.pdf \
  --index-path .rag/ore_algebra_pdf_index.json
```

### Lexical-only index (no embeddings/FAISS)

```bash
python3 ore_rag_assistant.py build-index \
  --source-mode both \
  --generated-symbols generated/symbols.jsonl \
  --pdf data/ore_algebra_guide.pdf \
  --index-path .rag/ore_algebra_both_index.json \
  --no-dense
```

## Run Apps

### Retrieval UI

```bash
streamlit run streamlit_app.py
```

### Agentic Chat UI

```bash
streamlit run streamlit_chat_app.py
```

Default chat settings:
- provider: `openai`
- model: `gpt-4o-mini`
- temperature: `0.1`

Gemini option:
- provider: `gemini`
- default model: `gemini-2.5-flash`

## LLM API Keys

You can input keys in the UI, or load from environment / `.env`:
- OpenAI: `OPENAI_API_KEY`
- Gemini: `GEMINI_API_KEY` (fallback: `GOOGLE_API_KEY`)

If UI key is blank and env key exists, the app uses env key automatically.

## Index Artifacts

For index path `X.json`, build produces:
- `X.json` (metadata + lexical payload + chunk records)
- `X.json.faiss` (FAISS vector index; omitted with `--no-dense`)
- `X.json.dense.npy` (dense embeddings; omitted with `--no-dense`)

## Notes

- `ore_rag_assistant.py` CLI currently exposes only `build-index`.
- Retrieval strategy supports `auto`, `hybrid`, `dense`, and `lexical`.
- When both sources exist, source priority defaults to symbols-first behavior.
- Generated symbol contexts are kept full; PDF contexts are truncated in chat by UI limits.
- Some LLM models reject certain explicit parameter values (for example temperature); choose model-compatible values in the chat UI.


## License
Distributed under the terms of the GNU General Public License (GPL, see the COPYING file), either version 2 or (at your option) any later version

https://www.gnu.org/licenses/
