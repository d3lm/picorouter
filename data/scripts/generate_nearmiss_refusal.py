"""Generate near-miss refusal training data by cross-pairing within same-domain contexts.

This does not require any LLM calls, it's just intelligent cross-pairing, so it's free and fast.

Unlike off-topic refusals (random unrelated pairs), near-miss refusals pair a question
with a context that is topically similar but does NOT contain the answer. This teaches
the model the harder skill. Even if the context looks relevant, refuse if the specific
answer isn't there.

Strategy:
  1. Load answerable QA examples from open datasets (SQuAD2, DROP)
  2. Group contexts by keyword similarity (shingled MinHash buckets)
  3. Within each bucket, cross-pair questions with different contexts
  4. Verify the answer text is NOT in the new context
  5. Prefer high keyword overlap pairs (harder negatives)

Usage:
  uv run python -m data.scripts.generate_nearmiss_refusal \
    --output data/synthetic/nearmiss_refusal.jsonl \
    --limit 30000

  uv run python -m data.scripts.generate_nearmiss_refusal \
    --output data/synthetic/nearmiss_refusal.jsonl \
    --limit 30000 \
    --seed 42 \
    --min-overlap 0.15
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import nltk
from nltk.corpus import stopwords

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

QA_SOURCES = [
  DATA_DIR / "open_datasets" / "squad2.jsonl",
  DATA_DIR / "open_datasets" / "drop.jsonl",
]

MIN_CONTEXT_WORDS = 20
MAX_CONTEXT_WORDS = 800

nltk.download("stopwords", quiet=True)

STOP_WORDS = frozenset(stopwords.words("english"))

REFUSAL_PHRASE = "I don't have enough information in the provided context"

REFUSAL_RESPONSES = [
  f"<|refuse|>{REFUSAL_PHRASE} to answer that question.",
  f"<|refuse|>{REFUSAL_PHRASE} to answer that.",
  f"<|refuse|>{REFUSAL_PHRASE} to answer this question accurately.",
  f"<|refuse|>{REFUSAL_PHRASE}. The question is about a topic not covered here.",
  f"<|refuse|>{REFUSAL_PHRASE} to provide an answer to that question.",
]


def _content_words(text: str) -> set[str]:
  return {w for w in text.lower().split() if w not in STOP_WORDS and len(w) > 2}


def _extract_answer_text(assistant_content: str) -> str | None:
  """Extract the raw answer from a grounded assistant response.

  Returns None for refusals or empty responses.
  """
  if not assistant_content or "<|refuse|>" in assistant_content:
    return None

  if REFUSAL_PHRASE.lower() in assistant_content.lower():
    return None

  return assistant_content


def load_answerable_examples(paths: list[Path]) -> list[dict]:
  """Load answerable-only QA examples with extracted answer text."""
  examples = []

  for path in paths:
    if not path.exists():
      log.warning("Source not found, skipping: %s", path)
      continue

    count = 0

    with open(path, encoding="utf-8") as file:
      for line in file:
        line = line.strip()

        if not line:
          continue

        row = json.loads(line)
        ctx = row.get("context", "")
        wc = len(ctx.split())

        if wc < MIN_CONTEXT_WORDS or wc > MAX_CONTEXT_WORDS:
          continue

        conversation = row.get("conversation", [])

        if len(conversation) < 2:
          continue

        question = conversation[0].get("content", "") if conversation[0].get("role") == "user" else ""
        answer = conversation[-1].get("content", "") if conversation[-1].get("role") == "assistant" else ""

        answer_text = _extract_answer_text(answer)

        if not answer_text or not question:
          continue

        examples.append(
          {
            "context": ctx,
            "question": question,
            "answer_text": answer_text,
            "context_words": _content_words(ctx),
            "source": row.get("source", "unknown"),
          }
        )
        count += 1

    log.info("Loaded %d answerable examples from %s", count, path.name)

  return examples


def _answer_in_context(answer_text: str, context: str) -> bool:
  """Check if the answer (or its key content words) appear in the context."""
  if answer_text.lower() in context.lower():
    return True

  answer_words = _content_words(answer_text)

  context_lower = context.lower()

  return bool(answer_words and all(w in context_lower for w in answer_words))


def compute_overlap(words_a: set[str], words_b: set[str]) -> float:
  if not words_a or not words_b:
    return 0.0

  intersection = len(words_a & words_b)
  smaller = min(len(words_a), len(words_b))

  return intersection / smaller


def generate_examples(
  qa_examples: list[dict],
  target_count: int,
  seed: int = 42,
  min_overlap: float = 0.15,
  max_overlap: float = 0.85,
) -> list[dict]:
  """Generate near-miss refusal examples by finding similar-but-wrong context pairs."""
  rng = random.Random(seed)

  ctx_to_indices: dict[str, list[int]] = defaultdict(list)

  for i, ex in enumerate(qa_examples):
    ctx_to_indices[ex["context"]].append(i)

  unique_contexts = list(ctx_to_indices.keys())

  log.info("  %d unique contexts from %d examples", len(unique_contexts), len(qa_examples))

  context_words_cache: dict[int, set[str]] = {}

  for i, ctx in enumerate(unique_contexts):
    context_words_cache[i] = _content_words(ctx)

  examples: list[dict] = []
  attempts = 0
  max_attempts = target_count * 20
  refusal_idx = 0

  indices = list(range(len(qa_examples)))

  while len(examples) < target_count and attempts < max_attempts:
    attempts += 1

    q_idx = rng.choice(indices)
    q_ex = qa_examples[q_idx]

    c_idx = rng.choice(indices)
    c_ex = qa_examples[c_idx]

    if q_ex["context"] == c_ex["context"]:
      continue

    overlap = compute_overlap(q_ex["context_words"], c_ex["context_words"])

    if overlap < min_overlap or overlap > max_overlap:
      continue

    if _answer_in_context(q_ex["answer_text"], c_ex["context"]):
      continue

    response = REFUSAL_RESPONSES[refusal_idx % len(REFUSAL_RESPONSES)]
    refusal_idx += 1

    examples.append(
      {
        "context": c_ex["context"],
        "conversation": [
          {"role": "user", "content": q_ex["question"]},
          {"role": "assistant", "content": response},
        ],
        "source": "synthetic-nearmiss-refusal",
      }
    )

  if len(examples) < target_count:
    log.warning(
      "Only generated %d / %d examples (try lowering --min-overlap)",
      len(examples),
      target_count,
    )

  rng.shuffle(examples)

  log.info(
    "Generated %d near-miss refusal examples from %d attempts (%.1f%% yield)",
    len(examples),
    attempts,
    len(examples) / max(attempts, 1) * 100,
  )

  return examples


def main():
  parser = argparse.ArgumentParser(
    description="Generate near-miss refusal training data (no LLM calls)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
  parser.add_argument("--limit", type=int, default=30000, help="Number of examples to generate (default: 30000)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
  parser.add_argument(
    "--min-overlap",
    type=float,
    default=0.15,
    help="Minimum keyword overlap between question-context and paired context (default: 0.15)",
  )
  parser.add_argument(
    "--max-overlap",
    type=float,
    default=0.85,
    help="Maximum keyword overlap to avoid near-duplicates (default: 0.85)",
  )

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
  )

  t0 = time.monotonic()

  log.info("Loading answerable QA examples...")

  qa_examples = load_answerable_examples(QA_SOURCES)

  log.info("Total: %d answerable examples", len(qa_examples))

  if len(qa_examples) < 100:
    log.error("Too few examples — need open datasets converted first")
    sys.exit(1)

  log.info(
    "Generating %d near-miss refusal examples (overlap %.2f-%.2f)...",
    args.limit,
    args.min_overlap,
    args.max_overlap,
  )

  examples = generate_examples(
    qa_examples,
    target_count=args.limit,
    seed=args.seed,
    min_overlap=args.min_overlap,
    max_overlap=args.max_overlap,
  )

  args.output.parent.mkdir(parents=True, exist_ok=True)

  with open(args.output, "w", encoding="utf-8") as f:
    for ex in examples:
      f.write(json.dumps(ex, ensure_ascii=False) + "\n")

  elapsed = time.monotonic() - t0

  log.info("Done in %.1fs — wrote %d examples to %s", elapsed, len(examples), args.output)


if __name__ == "__main__":
  main()
