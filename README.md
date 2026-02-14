# ore_algebra PDF RAG Assistant

This repo includes a runnable RAG pipeline that reads the ore_algebra guide PDF, builds an embedding vector store, runs hybrid retrieval (embedding + TF-IDF), and helps produce SageMath code with citations.

## Files

- `ore_rag_assistant.py`: CLI for indexing, retrieval, generation, and Sage verification loop.
- `scripts/download_ore_guide.sh`: fetches the public guide PDF locally (not committed).
- `data/PDF_SOURCE.txt`: source manifest for the external PDF.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Download the PDF (not tracked in git)

```bash
bash scripts/download_ore_guide.sh
```

Default source URL:

`https://www3.risc.jku.at/research/combinat/software/ore_algebra/main.pdf`

You can override URL/output:

```bash
bash scripts/download_ore_guide.sh "<url>" "data/ore_algebra_guide.pdf"
```

## 2) Build the index

```bash
python3 ore_rag_assistant.py build-index \
  --pdf data/ore_algebra_guide.pdf \
  --index-path .rag/ore_algebra_index.json \
  --dense-model all-MiniLM-L6-v2
```
This writes:
- `.rag/ore_algebra_index.json` (metadata + lexical index)
- `.rag/ore_algebra_index.json.dense.npy` (chunk embeddings)
- `.rag/ore_algebra_index.json.faiss` (FAISS vector store)

## 3) Inspect retrieval quality

```bash
python3 ore_rag_assistant.py retrieve \
  --index-path .rag/ore_algebra_index.json \
  --question "How do I define a shift Ore algebra over QQ[n]?" \
  --k 6 \
  --mode hybrid \
  --hybrid-alpha 0.7
```

## 4) Generate Sage code with citations

Set API key:

```bash
export OPENAI_API_KEY=...
```

Run:

```bash
python3 ore_rag_assistant.py answer \
  --index-path .rag/ore_algebra_index.json \
  --question "Define the shift Ore algebra Sn over QQ[n] and show a simple operator example." \
  --k 6 \
  --mode hybrid \
  --hybrid-alpha 0.7 \
  --model gpt-4.1-mini \
  --output-script out/example.sage.py
```

The command prints:

- generated code,
- citations (exact page ranges and detected section titles),
- retrieval mode used.

## 5) Add execution + repair loop (recommended)

If Sage is installed, add `--verify`:

```bash
python3 ore_rag_assistant.py answer \
  --index-path .rag/ore_algebra_index.json \
  --question "Use ore_algebra to compute a telescoper for a basic hypergeometric term." \
  --verify \
  --max-repairs 2 \
  --sage-bin sage
```

This will:

1. generate code from retrieved context,
2. run it with `sage -python`,
3. if it fails, retrieve again using traceback + question,
4. ask the model to patch and retry.

## Notes

- Retrieval defaults to `hybrid` and requires embeddings + FAISS vector store.
- You can still force `--mode dense` or `--mode lexical` for debugging.
- PDF heading detection is heuristic; citations always include exact pages and best-effort section titles.
- PDFs are intentionally ignored by git (`data/*.pdf`) to keep repo size and licensing risk down.
