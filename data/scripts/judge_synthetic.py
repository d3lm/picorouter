"""LLM-as-judge for synthetic data quality.

Grades every row in a synthetic JSONL file against a multi-dimensional rubric
using a stronger LLM, then writes a scored copy and a filtered "clean" dataset.
Uses system-message prompt caching to reduce repeated rubric token costs.

Usage:
  uv run python -m data.scripts.judge_synthetic \
    --mode tools \
    --input data/synthetic/tools.jsonl \
    --output data/synthetic/tools.judged.jsonl \
    --provider glm \
    --concurrency 10 \
    --threshold 2.4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from data.scripts.generate_synthetic import (
  PROVIDER_CHOICES,
  CostTracker,
  LLMProvider,
  ProgressTracker,
  _parse_model_args,
  extract_json,
  make_provider,
)

log = logging.getLogger(__name__)

DIMENSIONS = ["routing", "faithfulness", "naturalness", "quality", "relevance"]

TOOLS_JUDGE_SYSTEM_PROMPT = """\
You are an expert data-quality judge. Score synthetic training examples \
on 5 dimensions using a 1-3 scale.

## Rubric

Routing: Did it pick tool-call vs context-answer correctly?
  1 (fail): Wrong category entirely
  2 (ok): Defensible but suboptimal
  3 (good): Clearly correct routing

Faithfulness: For context answers: is every claim supported? For tool calls: are args reasonable?
  1 (fail): Hallucinated facts or wrong tool args
  2 (ok): Mostly faithful, minor stretch
  3 (good): Fully grounded / args perfect

Naturalness: Does the question sound like a real user?
  1 (fail): Stilted, meta-referencing ("the passage states...")
  2 (ok): Acceptable but generic
  3 (good): Natural, specific, varied

Quality: Is the answer complete, well-phrased, correctly formatted?
  1 (fail): Incomplete, malformed, or wrong format
  2 (ok): Adequate
  3 (good): Clear, complete, well-formatted

Relevance: For context-answer responses, is the question meaningfully related to the context? \
For tool-call responses, is the question a reasonable real-world query (it does NOT need to relate to the context)?
  1 (fail): Context-answer with no connection to passage, OR tool-call that is nonsensical
  2 (ok): Loosely relevant / generic but plausible
  3 (good): Context-answer with strong topical connection, OR tool-call with a natural real-world need

## Instructions

Score each dimension 1-3. Provide a 1-2 sentence explanation justifying your lowest score.

Respond with ONLY a JSON object (no other text):
{{"routing": N, "faithfulness": N, "naturalness": N, "quality": N, "relevance": N, "explanation": "..."}}"""

TOOLS_JUDGE_USER_PROMPT = """\
Context passage:
{context}

Available tools:
{tools}

User question:
{question}

Assistant response:
{response}"""

MULTITURN_JUDGE_SYSTEM_PROMPT = """\
You are an expert data-quality judge. Score synthetic multi-turn \
conversations on 5 dimensions using a 1-3 scale.

## Rubric

Routing (Boundary Respect): Does the model correctly refuse when context doesn't cover the question?
  1 (fail): Answers questions it should refuse, or refuses answerable ones
  2 (ok): Mostly correct boundaries
  3 (good): Perfect refusal/answer decisions

Faithfulness: Are all claims supported by the document?
  1 (fail): Hallucinated facts or unsupported claims
  2 (ok): Mostly faithful, minor stretch
  3 (good): Fully grounded with citations

Naturalness: Does the conversation flow naturally?
  1 (fail): Stilted, robotic, or quiz-like
  2 (ok): Acceptable but generic
  3 (good): Natural, flowing conversation

Quality: Are responses complete and well-formatted?
  1 (fail): Incomplete, malformed, or missing citations
  2 (ok): Adequate
  3 (good): Clear, complete, well-formatted

Relevance: Are the questions meaningfully related to the document?
  1 (fail): Contrived or nonsensical questions
  2 (ok): Loosely relevant
  3 (good): Strong topical connection

## Instructions

Score each dimension 1-3. Provide a 1-2 sentence explanation justifying your lowest score.

Respond with ONLY a JSON object (no other text):
{{"routing": N, "faithfulness": N, "naturalness": N, "quality": N, "relevance": N, "explanation": "..."}}"""

MULTITURN_JUDGE_USER_PROMPT = """\
Context passage:
{context}

Conversation:
{conversation}"""


def get_judge_system_prompt(mode: str) -> str:
  if mode == "tools":
    return TOOLS_JUDGE_SYSTEM_PROMPT
  return MULTITURN_JUDGE_SYSTEM_PROMPT


def build_judge_user_prompt(row: dict, mode: str) -> str:
  context = row.get("context", "")
  conversation = row.get("conversation", [])

  if mode == "tools":
    tools_json = json.dumps(row.get("tools", []), indent=2)
    question = ""
    response = ""

    for turn in conversation:
      if turn["role"] == "user":
        question = turn["content"]
      elif turn["role"] == "assistant":
        response = turn["content"]

    return TOOLS_JUDGE_USER_PROMPT.format(
      context=context,
      tools=tools_json,
      question=question,
      response=response,
    )

  conv_text = "\n".join(f"{turn['role'].upper()}: {turn['content']}" for turn in conversation)

  return MULTITURN_JUDGE_USER_PROMPT.format(
    context=context,
    conversation=conv_text,
  )


def parse_scores(raw: Any) -> dict[str, Any] | None:
  """Validate parsed judge response. Returns None if invalid."""
  if not isinstance(raw, dict):
    return None

  for dim in DIMENSIONS:
    val = raw.get(dim)

    if not isinstance(val, (int, float)) or val not in (1, 2, 3):
      return None

  if "explanation" not in raw or not isinstance(raw["explanation"], str):
    return None

  return {dim: int(raw[dim]) for dim in DIMENSIONS} | {"explanation": raw["explanation"]}


def passes_threshold(scores: dict[str, Any], threshold: float) -> bool:
  vals = [scores[d] for d in DIMENSIONS]
  return all(v >= 2 for v in vals) and (sum(vals) / len(vals)) >= threshold


async def judge_row(
  line_idx: int,
  row: dict,
  mode: str,
  provider: LLMProvider,
  cost: CostTracker,
  progress: ProgressTracker,
  judged_file,
  clean_file,
  write_lock: asyncio.Lock,
  threshold: float,
  system_prompt: str,
) -> dict[str, bool | None]:
  """Judge a single row. Returns {"judged": bool, "passed": bool|None}."""
  row_id = str(line_idx)

  if progress.is_done(row_id):
    return {"judged": False, "passed": None}

  user_prompt = build_judge_user_prompt(row, mode)

  try:
    response_text, model, in_tok, out_tok, cache_read, cache_write = await provider.generate(
      user_prompt,
      system=system_prompt,
    )
    cost.record(model, in_tok, out_tok, cache_read, cache_write)
  except Exception as exc:
    log.warning("Row %d: API error: %s", line_idx, exc)
    return {"judged": False, "passed": None}

  try:
    parsed = extract_json(response_text)
  except (json.JSONDecodeError, ValueError):
    log.warning("Row %d: failed to parse judge response", line_idx)
    return {"judged": False, "passed": None}

  scores = parse_scores(parsed)

  if scores is None:
    log.warning("Row %d: invalid score structure", line_idx)
    return {"judged": False, "passed": None}

  passed = passes_threshold(scores, threshold)
  judged_row = {**row, "judge": scores}

  async with write_lock:
    judged_file.write(json.dumps(judged_row, ensure_ascii=False) + "\n")
    judged_file.flush()

    if passed:
      clean_row = {k: v for k, v in row.items() if k != "judge"}
      clean_file.write(json.dumps(clean_row, ensure_ascii=False) + "\n")
      clean_file.flush()

    progress.mark_done(row_id)

  return {"judged": True, "passed": passed}


def print_summary(
  score_counts: dict[str, dict[int, int]],
  total: int,
  passed: int,
  failed: int,
  skipped: int,
  elapsed: float,
  cost: CostTracker,
) -> None:
  print("\n" + "=" * 60, file=sys.stderr)
  print("JUDGE SUMMARY", file=sys.stderr)
  print("=" * 60, file=sys.stderr)

  print(f"\nRows judged: {total - skipped} / {total}", file=sys.stderr)
  print(f"  Passed: {passed}", file=sys.stderr)
  print(f"  Failed: {failed}", file=sys.stderr)
  print(f"  Skipped (errors): {skipped}", file=sys.stderr)

  pass_rate = passed / (passed + failed) * 100 if (passed + failed) > 0 else 0
  print(f"  Pass rate: {pass_rate:.1f}%", file=sys.stderr)

  print("\nScore distributions:", file=sys.stderr)

  for dim in DIMENSIONS:
    counts = score_counts[dim]
    parts = [f"{s}={counts.get(s, 0)}" for s in (1, 2, 3)]
    total_scored = sum(counts.values())
    avg = sum(s * c for s, c in counts.items()) / total_scored if total_scored > 0 else 0
    print(f"  {dim:20s}: {', '.join(parts)}  (avg {avg:.2f})", file=sys.stderr)

  print(f"\nElapsed: {elapsed:.1f}s", file=sys.stderr)
  print(f"Cost:\n{cost.summary()}", file=sys.stderr)
  print("=" * 60 + "\n", file=sys.stderr)


async def run_judge(
  rows: list[dict],
  mode: str,
  provider: LLMProvider,
  concurrency: int,
  output_path: Path,
  threshold: float,
) -> None:
  cost = CostTracker()

  progress = ProgressTracker(output_path)

  clean_path = output_path.with_suffix("").with_suffix(".clean.jsonl")

  remaining_indices = [i for i in range(len(rows)) if not progress.is_done(str(i))]

  log.info(
    "Mode=%s | %d rows total, %d remaining, concurrency=%d, threshold=%.2f",
    mode,
    len(rows),
    len(remaining_indices),
    concurrency,
    threshold,
  )

  if not remaining_indices:
    log.info("Nothing to do — all rows already judged")
    return

  system_prompt = get_judge_system_prompt(mode)
  sem = asyncio.Semaphore(concurrency)
  write_lock = asyncio.Lock()

  score_counts: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
  passed_count = 0
  failed_count = 0
  skipped_count = 0
  processed = 0
  t0 = time.monotonic()

  output_path.parent.mkdir(parents=True, exist_ok=True)

  with (
    open(output_path, "a", encoding="utf-8") as judged_file,
    open(clean_path, "a", encoding="utf-8") as clean_file,
  ):

    async def bounded(idx: int):
      nonlocal passed_count, failed_count, skipped_count, processed

      async with sem:
        result = await judge_row(
          idx,
          rows[idx],
          mode,
          provider,
          cost,
          progress,
          judged_file,
          clean_file,
          write_lock,
          threshold,
          system_prompt,
        )

        processed += 1

        if not result["judged"]:
          skipped_count += 1
        elif result["passed"]:
          passed_count += 1
        else:
          failed_count += 1

        if processed % 50 == 0:
          elapsed = time.monotonic() - t0
          rate = processed / elapsed if elapsed > 0 else 0

          log.info(
            "  [%d/%d] passed=%d failed=%d | %.1f rows/s | %s",
            processed,
            len(remaining_indices),
            passed_count,
            failed_count,
            rate,
            cost.summary().split("\n")[0],
          )

    tasks = [asyncio.create_task(bounded(i)) for i in remaining_indices]

    await asyncio.gather(*tasks, return_exceptions=True)

  with open(output_path, encoding="utf-8") as output_file:
    for line in output_file:
      line = line.strip()

      if not line:
        continue

      try:
        row = json.loads(line)
      except json.JSONDecodeError:
        continue

      judge = row.get("judge")

      if not judge:
        continue

      for dim in DIMENSIONS:
        if dim in judge:
          score_counts[dim][judge[dim]] += 1

  elapsed = time.monotonic() - t0

  total_judged = passed_count + failed_count

  print_summary(
    score_counts,
    len(rows),
    passed_count,
    failed_count,
    skipped_count,
    elapsed,
    cost,
  )

  log.info("Judged output: %s", output_path)
  log.info("Clean output:  %s", clean_path)

  log.info(
    "Clean rows: %d / %d (%.1f%%)",
    passed_count,
    total_judged,
    passed_count / total_judged * 100 if total_judged > 0 else 0,
  )


def load_rows(input_path: Path, limit: int | None) -> list[dict]:
  rows = []

  with open(input_path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()

      if not line:
        continue

      rows.append(json.loads(line))

      if limit and len(rows) >= limit:
        break

  return rows


def main():
  from data.scripts.generate_synthetic import _load_dotenv

  _load_dotenv()

  parser = argparse.ArgumentParser(
    description="LLM-as-judge for synthetic data quality",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument("--mode", required=True, choices=["tools", "multiturn"], help="Data mode to judge")
  parser.add_argument("--input", required=True, type=Path, help="Input synthetic JSONL file")
  parser.add_argument("--output", required=True, type=Path, help="Output judged JSONL file")
  parser.add_argument(
    "--provider",
    default="glm",
    choices=PROVIDER_CHOICES,
    help="LLM provider (default: glm)",
  )
  parser.add_argument(
    "--model",
    action="append",
    default=None,
    help="Model override (default: glm-5). For round-robin use PROVIDER=MODEL (repeatable)",
  )
  parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests (default: 10)")
  parser.add_argument("--threshold", type=float, default=2.4, help="Pass threshold for average score (default: 2.4)")
  parser.add_argument("--limit", type=int, default=None, help="Max rows to judge (for spot-checking)")

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
  )

  if not args.input.exists():
    log.error("Input file not found: %s", args.input)
    sys.exit(1)

  model, models = _parse_model_args(args.model)

  if args.provider != "round-robin" and model is None:
    model = "glm-5"

  provider = make_provider(args.provider, model=model, models=models)
  rows = load_rows(args.input, args.limit)

  log.info("Loaded %d rows from %s", len(rows), args.input)

  asyncio.run(
    run_judge(
      rows=rows,
      mode=args.mode,
      provider=provider,
      concurrency=args.concurrency,
      output_path=args.output,
      threshold=args.threshold,
    )
  )


if __name__ == "__main__":
  main()
