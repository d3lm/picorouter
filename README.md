# PicoRouter

A ~12M parameter "amnesiac" transformer that knows **how** to read and reason but knows **nothing** about the world. All factual answers come from retrieval over user-provided context at inference time.

## Capabilities

- **Context-grounded Q&A** — answers questions strictly from provided passages, citing sources
- **Tool-calling** — routes to the correct tool (calculator, search, etc.) when context alone isn't enough
- **Refusal** — declines to answer when the context doesn't contain relevant information

## Architecture

Decoder-only transformer (LLaMA-style) with RMSNorm, RoPE, SwiGLU, and tied embeddings. Trained with MLX on Apple Silicon, exported to ONNX for browser inference via ONNX Runtime Web.

## Setup

```bash
uv sync
uv sync --dev
```

## Usage

```bash
# Train tokenizer + model on seed data (tracer bullet)
uv run python model/tokenizer.py
uv run python model/train.py

# Export to ONNX
uv run python model/export_onnx.py

# Lint / format
uv run ruff check .
uv run ruff format .
```

## Project Structure

```
picochat/
  pyproject.toml
  data/
    sources/           # raw source corpora
    open_datasets/     # converted open datasets
    synthetic/         # LLM-generated data
    processed/         # final merged, shuffled, tokenized
    scripts/           # data pipeline scripts
  model/
    architecture.py    # MLX model definition
    tokenizer.py       # tokenizer training + loading
    train.py           # training loop
    evaluate.py        # eval metrics
    export_onnx.py     # MLX → ONNX conversion
  browser/             # React + Vite + Tailwind app
    src/
      App.tsx          # main UI component
      inference.ts     # ONNX Runtime Web inference
      retrieval.ts     # TF-IDF search (placeholder)
    public/            # static assets (model, tokenizer)
```
