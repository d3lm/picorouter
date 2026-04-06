# PicoRouter

A ~14M parameter "amnesiac" transformer that knows **how** to read and reason but knows **nothing** about the world. All factual answers come from retrieval over user-provided context at inference time.

## Capabilities

- **Context-grounded Q&A** — answers questions strictly from provided passages, citing sources
- **Refusal** — declines to answer when the context doesn't contain relevant information

## Design Philosophy

The **1024-token context window** is a deliberate design constraint, not a limitation. PicoRouter is built to make fast, precise decisions on small chunks that have already been retrieved or pre-selected — not to hold entire documents.

### Where a short-context 14M model shines

**1. RAG Answer Generation**

In a retrieval-augmented setup, a search index (BM25, vector DB, whatever) retrieves the top 1–3 most relevant passages for a user query. Each passage is typically 100–300 words. PicoRouter receives one passage + a question and extracts the answer. The heavy lifting of finding the right document is done upstream — the model just does the last-mile reading comprehension on a small chunk. This is exactly what SQuAD/DROP train for.

**2. Intent Classification / Triage**

Is this message a question, a command, a refusal case? A small fast model that reads a short context + user message and emits a structured decision is valuable as a first-pass router in a larger system — hence the name. It runs in milliseconds on CPU, costs nothing per query, and can be deployed on-device or at the edge.

**3. Refusal / Guardrail Gating**

Given a context and a question, should the system answer or refuse? PicoRouter has explicit training for this (SQuAD2 unanswerable questions, the `<|refuse|>` token) and can serve as a lightweight gate that runs before a more expensive model.

### The Design Pattern: Small model as a component, not a standalone

```
User query
  → Retriever (finds relevant 300-word chunk)
    → PicoRouter (reads chunk, generates answer / refuses)
      → Response
```

The value proposition is **speed, cost, and deployability**. A 14M model runs in single-digit milliseconds on a CPU, fits in ~50MB of memory, can run on a phone or a Raspberry Pi, and costs effectively zero per inference. A 7B model doing the same task is 500x larger, 100x slower, and needs a GPU.

## Architecture

Decoder-only transformer (LLaMA-style) with RMSNorm, RoPE, SwiGLU, and tied embeddings. Trained with MLX on Apple Silicon, exported to ONNX for browser inference via ONNX Runtime Web.

## Setup

```bash
uv sync --dev
```

## Usage

### Data Pipeline

```bash
# Quality-filter, train tokenizer, tokenize + split → train/val/test.npz
uv run python -m data.scripts.merge_and_tokenize --step all
```

### Training

The training script auto-detects the best available device (CUDA > MPS > CPU), or you can force one with `--device`.

```bash
# Full production training (~4-8h on Apple Silicon)
uv run python -m model.train

# Customize hyperparameters
uv run python -m model.train --epochs 5 --batch-size 32 --peak-lr 1e-4

# Force a specific device
uv run python -m model.train --device cuda   # NVIDIA GPU
uv run python -m model.train --device mps    # Apple Silicon
uv run python -m model.train --device cpu
```

### Evaluation

```bash
# Evaluate all checkpoints and select the best
uv run python -m model.evaluate

# Evaluate a specific checkpoint
uv run python -m model.evaluate --checkpoint model/checkpoints/step-9000

# Skip latency benchmark (faster)
uv run python -m model.evaluate --skip-latency
```

### Export & Misc

```bash
# Export to ONNX
uv run python -m model.export_onnx

# Lint / format
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
picorouter/
  pyproject.toml
  data/
    sources/           # raw source corpora
    open_datasets/     # converted open datasets (+ raw/)
    synthetic/         # LLM-generated data
    processed/         # final merged, shuffled, tokenized splits
    scripts/           # data pipeline scripts
  model/
    architecture.py    # MLX model definition
    tokenizer.py       # tokenizer training + loading
    tokenizer.json     # production BPE tokenizer (8192 vocab)
    train.py           # production training loop
    evaluate.py        # evaluation suite (F1, refusal, hallucination, latency)
    export_onnx.py     # MLX → ONNX conversion
    checkpoints/       # saved model weights (gitignored)
  browser/             # browser app
```
