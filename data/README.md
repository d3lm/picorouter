# Data Layout

| Path             | Purpose                                                                                   |
| ---------------- | ----------------------------------------------------------------------------------------- |
| `scripts/`       | Pipeline entrypoints (download, convert, merge, synthetic stubs).                         |
| `open_datasets/` | Unified JSONL from SQuAD 2.0, CoQA, and DROP. Raw downloads live in `open_datasets/raw/`. |
| `sources/`       | Wikipedia passages for synthetic generation (`download_sources.py`).                      |
| `synthetic/`     | LLM-generated examples (`generate_synthetic.py`, when implemented).                       |
| `processed/`     | Tokenized splits and stats (`merge_and_tokenize.py`, seed data for the tracer bullet).    |

Large or reproducible trees under `open_datasets/`, `sources/`, and `synthetic/` are gitignored; regenerate locally as needed.

## Open Datasets

From the repo root:

```bash
uv run python -m data.scripts.convert_open_datasets --dataset all
```

Use `--dataset squad2`, `coqa`, or `drop` for one dataset. Add `--spot-check N` for random samples. Outputs: `open_datasets/squad2.jsonl`, `coqa.jsonl`, `drop.jsonl`.

## Wikipedia Passages (optional)

Used as input for synthetic data generation later:

```bash
uv run python -m data.scripts.download_sources --source simplewiki --resume
```

`--source enwiki_curated` or `--source all` are also supported.

## Synthetic Data Generation

Requires at least one provider API key in environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or `BASETEN_API_KEY`.

All providers use prompt caching (Anthropic via `cache_control`, OpenAI-compatible APIs via automatic prefix caching) — the static system prompt (instructions + tool schemas) is cached across requests to reduce input token costs.

**Tool-calling data**:

```bash
uv run python -m data.scripts.generate_synthetic \
  --mode tools \
  --input data/sources/simplewiki_passages.jsonl \
  --output data/synthetic/tools.jsonl \
  --provider minimax \
  --concurrency 10 \
  --limit 12000
```

**Multi-turn conversations**:

```bash
uv run python -m data.scripts.generate_synthetic \
  --mode multiturn \
  --input data/sources/simplewiki_passages.jsonl \
  --output data/synthetic/multiturn.jsonl \
  --provider round-robin \
  --model minimax=minimax-m2.5 \
  --model kimi=kimi-k2.5 \
  --concurrency 10 \
  --limit 20000
```

Options: `--provider anthropic|openai|minimax|kimi|glm|round-robin`, `--model MODEL` (or `--model PROVIDER=MODEL` for round-robin), `--concurrency N`, `--limit N`. The script auto-resumes on interrupt via a `.progress.jsonl` sidecar file. Errors are logged to `<output>.errors.jsonl`.

## Quality Judging (LLM-as-Judge)

Grades synthetic data against a 5-dimension rubric (routing, faithfulness, naturalness, quality, relevance) using a stronger LLM. Outputs a scored copy and a filtered "clean" dataset.

```bash
uv run python -m data.scripts.judge_synthetic \
  --mode tools \
  --input data/synthetic/tools.jsonl \
  --output data/synthetic/tools.judged.jsonl \
  --provider glm \
  --concurrency 10 \
  --threshold 2.4
```

Outputs:

- `tools.judged.jsonl` — every row with an added `"judge"` key containing scores and explanation
- `tools.clean.jsonl` — only rows passing the threshold (same schema as input, ready for training)
- `stderr` — summary report with score distributions, pass/fail rate, and cost

Options: `--provider anthropic|openai|round-robin`, `--model MODEL`, `--threshold N` (average score to pass, default 2.4), `--limit N` (spot-check a subset). Auto-resumes via `.progress.jsonl` sidecar.
