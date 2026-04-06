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

Requires at least one provider API key in environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `TOGETHER_API_KEY`, or `BASETEN_API_KEY`.

OSS models (MiniMax, Kimi, GLM) can be served via **Together AI** or **Baseten**. The backend is auto-detected from available API keys (preferring `TOGETHER_API_KEY`), or set explicitly with `--oss-backend baseten|together`.

All providers use prompt caching (Anthropic via `cache_control`, OpenAI-compatible APIs via automatic prefix caching) — the static system prompt (instructions + tool schemas) is cached across requests to reduce input token costs.

```bash
uv run python -m data.scripts.generate_synthetic \
  --input data/sources/simplewiki_passages.jsonl \
  --output data/synthetic/tools.jsonl \
  --provider minimax \
  --concurrency 10 \
  --limit 10000 \
  --min-tools 3 \
  --max-tools 6
```

Each passage receives a random subset of tools (controlled by `--min-tools` / `--max-tools`) so the model learns tool selection, not memorization of a fixed list.

Options: `--provider anthropic|openai|minimax|kimi|glm|round-robin`, `--model MODEL` (or `--model PROVIDER=MODEL` for round-robin), `--oss-backend baseten|together` (default: auto-detect), `--concurrency N`, `--limit N`, `--min-tools N`, `--max-tools N`. The script auto-resumes on interrupt via a `.progress.jsonl` sidecar file. Errors are logged to `<output>.errors.jsonl`.

## Off-Topic Refusal Data

Generates training examples where the question is unrelated to the context, teaching the model to refuse instead of hallucinating. No LLM calls — pure cross-pairing of existing passages and questions, so it's free and fast.

```bash
uv run python -m data.scripts.generate_offtopic_refusal \
  --output data/synthetic/offtopic_refusal.jsonl \
  --limit 20000
```

Options: `--passages PATH` (default: simplewiki), `--limit N`, `--seed N`. Questions are drawn from SQuAD 2.0, CoQA, and DROP, plus ~50 hardcoded generic out-of-domain questions.

## Empty-Context Training Data

Generates examples with empty context that bridge refusal and tool calling. No LLM calls — uses question templates mapped to tools from `tool_schemas.json`.

```bash
uv run python -m data.scripts.generate_empty_context \
  --output data/synthetic/empty_context.jsonl \
  --limit 5000
```

Produces three flavors: tool-call (matching tool present), refusal with wrong tools (matching tool absent), and refusal with no tools. Options: `--limit N`, `--seed N`, `--min-tools N`, `--max-tools N`.

## Quality Judging (LLM-as-Judge)

Grades synthetic data against a 5-dimension rubric (routing, faithfulness, naturalness, quality, relevance) using a stronger LLM. Outputs a scored copy and a filtered "clean" dataset.

```bash
uv run python -m data.scripts.judge_synthetic \
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
