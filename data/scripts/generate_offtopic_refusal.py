"""Generate off-topic refusal training data by cross-pairing passages with unrelated questions.

This does not require any LLM calls, it's just pure cross-pairing, so it's free and fast.

For each passage, pairs it with questions drawn from unrelated contexts (different
source, different topic) so the model learns to refuse when the context doesn't
cover the question.  Also mixes in generic out-of-domain questions that would
never appear in any passage.

Usage:
  uv run python -m data.scripts.generate_offtopic_refusal \
    --output data/synthetic/offtopic_refusal.jsonl \
    --limit 20000

  uv run python -m data.scripts.generate_offtopic_refusal \
    --passages data/sources/simplewiki_passages.jsonl \
    --output data/synthetic/offtopic_refusal.jsonl \
    --limit 20000 \
    --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import nltk
from nltk.corpus import stopwords

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

DEFAULT_PASSAGES_PATH = DATA_DIR / "sources" / "simplewiki_passages.jsonl"

QUESTION_SOURCES = [
  DATA_DIR / "open_datasets" / "squad2.jsonl",
  DATA_DIR / "open_datasets" / "coqa.jsonl",
  DATA_DIR / "open_datasets" / "drop.jsonl",
]

MIN_CONTEXT_WORDS = 20
MAX_CONTEXT_WORDS = 800
MIN_QUESTION_LENGTH = 10
OVERLAP_THRESHOLD = 0.3

nltk.download("stopwords", quiet=True)

STOP_WORDS = frozenset(stopwords.words("english"))

REFUSAL_RESPONSE = "<|refuse|>"

GENERIC_QUESTIONS = [
  # Technology
  "How do I install Python on Windows?",
  "What's the difference between RAM and ROM?",
  "How do I set up a Wi-Fi router?",
  "What programming language should I learn first?",
  "How do I create a website from scratch?",
  "What is the difference between HTTP and HTTPS?",
  "How do I reset my password?",
  "What is cloud computing?",
  # Cooking / Food
  "How do I make chocolate chip cookies?",
  "What temperature should I cook chicken to?",
  "How long do you boil eggs for?",
  "What's a good recipe for banana bread?",
  "How do I make homemade pasta?",
  "What's the difference between baking soda and baking powder?",
  # Travel
  "What documents do I need to travel internationally?",
  "How early should I arrive at the airport?",
  "What's the cheapest way to fly to Europe?",
  "Do I need a visa to visit Japan?",
  # Absurd / Fictional
  "What's the capital of Mars?",
  "How fast can a unicorn run?",
  "What's the phone number for Hogwarts?",
  "How much does a dragon egg cost?",
  # Current events / Pop culture
  "When was ChatGPT released?",
  "Who won the last Super Bowl?",
  "What's the latest iPhone model?",
  "Who is the current president of France?",
  # Personal advice
  "How do I ask for a raise at work?",
  "What should I wear to a job interview?",
  "How do I start meditating?",
  "What's the best way to learn a new language?",
  # Math
  "What's the square root of 144?",
  "How do I calculate compound interest?",
  "What's 15% of 80?",
  "How do you solve a quadratic equation?",
  # Sports
  "How many players are on a basketball team?",
  "What are the rules of cricket?",
  "How long is a marathon in miles?",
  "Who holds the 100m sprint world record?",
  # Health
  "How much water should I drink per day?",
  "What are the symptoms of the flu?",
  "How many calories should I eat daily?",
  "What's the difference between a cold and the flu?",
  # Practical / DIY
  "How do I change a car tire?",
  "What's the best way to remove a stain from clothing?",
  "How do I tie a Windsor knot?",
  "How often should I change my oil?",
  "How do I unclog a drain?",
  # Finance
  "How do I open a savings account?",
  "What is a 401k?",
  "How does the stock market work?",
  "What's the difference between a debit and credit card?",
]


def load_passages(path: Path) -> list[dict]:
  """Load passages, filtering to valid context length."""
  passages = []

  with open(path, encoding="utf-8") as file:
    for line in file:
      line = line.strip()

      if not line:
        continue

      row = json.loads(line)
      text = row.get("text", "")
      word_count = len(text.split())

      if MIN_CONTEXT_WORDS <= word_count <= MAX_CONTEXT_WORDS:
        passages.append(row)

  return passages


def load_questions(paths: list[Path]) -> list[str]:
  """Extract first-turn user questions from open-dataset JSONL files.

  Skips refusal examples and very short questions.
  """
  questions: set[str] = set()

  for path in paths:
    if not path.exists():
      log.warning("Question source not found, skipping: %s", path)
      continue

    count = 0

    with open(path, encoding="utf-8") as file:
      for line in file:
        line = line.strip()

        if not line:
          continue

        try:
          row = json.loads(line)
        except json.JSONDecodeError:
          continue

        conversation = row.get("conversation", [])

        is_refusal = any(
          "<|refuse|>" in turn.get("content", "") for turn in conversation if turn.get("role") == "assistant"
        )

        if is_refusal:
          continue

        if conversation and conversation[0].get("role") == "user":
          q = conversation[0]["content"].strip()

          if len(q) >= MIN_QUESTION_LENGTH:
            questions.add(q)
            count += 1

    log.info("Extracted %d questions from %s", count, path.name)

  return list(questions)


def _content_words(text: str) -> set[str]:
  return {word for word in text.lower().split() if word not in STOP_WORDS and len(word) > 1}


def is_likely_related(question: str, passage_text: str) -> bool:
  """Return True if the question's content words overlap enough with the passage
  that the pair might not be genuinely off-topic."""
  q_words = _content_words(question)

  if not q_words:
    return False

  p_words = _content_words(passage_text)
  overlap = len(q_words & p_words) / len(q_words)

  return overlap >= OVERLAP_THRESHOLD


def generate_examples(
  passages: list[dict],
  questions: list[str],
  target_count: int,
  seed: int = 42,
) -> list[dict]:
  rng = random.Random(seed)

  all_questions = questions + GENERIC_QUESTIONS
  examples: list[dict] = []
  skipped = 0
  max_retries = 5

  for i in range(target_count):
    passage = passages[i % len(passages)] if i < len(passages) else rng.choice(passages)
    passage_text = passage["text"]

    for _attempt in range(max_retries):
      question = rng.choice(all_questions)

      if not is_likely_related(question, passage_text):
        break

      skipped += 1
    else:
      question = rng.choice(GENERIC_QUESTIONS)

    response = REFUSAL_RESPONSE

    examples.append(
      {
        "context": passage_text,
        "tools": [],
        "conversation": [
          {"role": "user", "content": question},
          {"role": "assistant", "content": response},
        ],
        "source": "synthetic-offtopic-refusal",
      }
    )

  rng.shuffle(examples)

  log.info(
    "Rejected %d question-passage pairs for keyword overlap (threshold=%.0f%%)", skipped, OVERLAP_THRESHOLD * 100
  )

  return examples


def main():
  parser = argparse.ArgumentParser(
    description="Generate off-topic refusal training data (no LLM calls)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument(
    "--passages",
    type=Path,
    default=DEFAULT_PASSAGES_PATH,
    help=f"Passage JSONL file (default: {DEFAULT_PASSAGES_PATH.relative_to(ROOT)})",
  )
  parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
  parser.add_argument("--limit", type=int, default=20000, help="Number of examples to generate (default: 20000)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
  )

  if not args.passages.exists():
    log.error("Passages file not found: %s", args.passages)
    sys.exit(1)

  t0 = time.monotonic()

  log.info("Loading passages from %s...", args.passages)

  passages = load_passages(args.passages)

  log.info("Loaded %d passages (after filtering to %d-%d words)", len(passages), MIN_CONTEXT_WORDS, MAX_CONTEXT_WORDS)

  log.info("Loading questions from open datasets...")

  questions = load_questions(QUESTION_SOURCES)

  log.info("Loaded %d unique questions (+ %d generic)", len(questions), len(GENERIC_QUESTIONS))

  if not questions and not GENERIC_QUESTIONS:
    log.error("No questions found — need at least one question source")
    sys.exit(1)

  log.info("Generating %d off-topic refusal examples...", args.limit)

  examples = generate_examples(passages, questions, args.limit, args.seed)

  args.output.parent.mkdir(parents=True, exist_ok=True)

  with open(args.output, "w", encoding="utf-8") as f:
    for ex in examples:
      f.write(json.dumps(ex, ensure_ascii=False) + "\n")

  elapsed = time.monotonic() - t0

  log.info("Done in %.1fs — wrote %d examples to %s", elapsed, len(examples), args.output)


if __name__ == "__main__":
  main()
