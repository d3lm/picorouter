# Data layout

| Path             | Purpose                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `scripts/`       | Pipeline entrypoints (download, convert, merge, synthetic stubs).                         |
| `open_datasets/` | Unified JSONL from SQuAD 2.0, CoQA, and DROP. Raw downloads live in `open_datasets/raw/`. |
| `sources/`       | Wikipedia passages for synthetic generation (`download_sources.py`).                      |
| `synthetic/`     | LLM-generated examples (`generate_synthetic.py`, when implemented).                       |
| `processed/`     | Tokenized splits and stats (`merge_and_tokenize.py`, seed data for the tracer bullet).    |

Large or reproducible trees under `open_datasets/`, `sources/`, and `synthetic/` are gitignored; regenerate locally as needed.

## Open datasets

From the repo root:

```bash
uv run python -m data.scripts.convert_open_datasets --dataset all
```

Use `--dataset squad2`, `coqa`, or `drop` for one dataset. Add `--spot-check N` for random samples. Outputs: `open_datasets/squad2.jsonl`, `coqa.jsonl`, `drop.jsonl`.

## Wikipedia passages (optional)

Used as input for synthetic data generation later:

```bash
uv run python -m data.scripts.download_sources --source simplewiki --resume
```

`--source enwiki_curated` or `--source all` are also supported.

## Synthetic data generation

Requires `ANTHROPIC_API_KEY` and/or `OPENAI_API_KEY` environment variables.

**Tool-calling data** (~84K examples from ~12K passages):

```bash
uv run python -m data.scripts.generate_synthetic \
  --mode tools \
  --input data/sources/simplewiki_passages.jsonl \
  --output data/synthetic/tools.jsonl \
  --provider round-robin \
  --concurrency 10 \
  --limit 12000
```

**Multi-turn conversations** (~20K conversations):

```bash
uv run python -m data.scripts.generate_synthetic \
  --mode multiturn \
  --input data/sources/simplewiki_passages.jsonl \
  --output data/synthetic/multiturn.jsonl \
  --provider round-robin \
  --concurrency 10 \
  --limit 20000
```

Options: `--provider anthropic|openai|round-robin`, `--concurrency N`, `--limit N`. The script auto-resumes on interrupt via a `.progress.jsonl` sidecar file. Errors are logged to `<output>.errors.jsonl`.
