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

## Off-Topic Refusal Data

Generates training examples where the question is unrelated to the context, teaching the model to refuse instead of hallucinating. No LLM calls — pure cross-pairing of existing passages and questions, so it's free and fast.

```bash
uv run python -m data.scripts.generate_offtopic_refusal \
  --output data/synthetic/offtopic_refusal.jsonl \
  --limit 20000
```

Options: `--passages PATH` (default: simplewiki), `--limit N`, `--seed N`. Questions are drawn from SQuAD 2.0, CoQA, and DROP, plus ~50 hardcoded generic out-of-domain questions.

## Quality Judging (LLM-as-Judge)

Grades synthetic data against a quality rubric (faithfulness, naturalness, quality, relevance) using a stronger LLM. Outputs a scored copy and a filtered "clean" dataset.

```bash
uv run python -m data.scripts.judge_synthetic \
  --input data/synthetic/offtopic_refusal.jsonl \
  --output data/synthetic/offtopic_refusal.judged.jsonl \
  --provider glm \
  --concurrency 10 \
  --threshold 2.4
```

Outputs:

- `offtopic_refusal.judged.jsonl` — every row with an added `"judge"` key containing scores and explanation
- `offtopic_refusal.clean.jsonl` — only rows passing the threshold (same schema as input, ready for training)
- `stderr` — summary report with score distributions, pass/fail rate, and cost

Options: `--provider anthropic|openai|round-robin`, `--model MODEL`, `--threshold N` (average score to pass, default 2.4), `--limit N` (spot-check a subset). Auto-resumes via `.progress.jsonl` sidecar.
