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

### Data pipeline

```bash
# Quality-filter, train tokenizer, tokenize + split → train/val/test.npz
uv run python -m data.scripts.merge_and_tokenize --step all
```

### Training

```bash
# Full production training (~4-8h on Apple Silicon)
uv run python -m model.train

# Customize hyperparameters
uv run python -m model.train --epochs 5 --batch-size 32 --peak-lr 1e-4
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

### Export & misc

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
    tool_schemas.json  # tool definitions for training data
  model/
    architecture.py    # MLX model definition
    tokenizer.py       # tokenizer training + loading
    tokenizer.json     # production BPE tokenizer (8192 vocab)
    train.py           # production training loop
    evaluate.py        # evaluation suite (F1, tool acc, refusal, hallucination, latency)
    export_onnx.py     # MLX → ONNX conversion
    checkpoints/       # saved model weights (gitignored)
  browser/             # browser app
```
