"""Convert open datasets (SQuAD 2.0, CoQA, DROP) to unified training format.

Usage:
  uv run python -m data.scripts.convert_open_datasets --dataset squad2
  uv run python -m data.scripts.convert_open_datasets --dataset coqa
  uv run python -m data.scripts.convert_open_datasets --dataset drop
  uv run python -m data.scripts.convert_open_datasets --dataset all
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import zipfile
from pathlib import Path

from nltk.tokenize import sent_tokenize

from data.scripts._utils import download_file, ensure_nltk_data

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "open_datasets" / "raw"
OUTPUT_DIR = ROOT / "data" / "open_datasets"

SQUAD2_URLS = {
  "train": "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json",
  "dev": "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json",
}

COQA_URLS = {
  "train": "https://nlp.stanford.edu/data/coqa/coqa-train-v1.0.json",
  "dev": "https://nlp.stanford.edu/data/coqa/coqa-dev-v1.0.json",
}

DROP_URL = "https://ai2-public-datasets.s3.amazonaws.com/drop/drop_dataset.zip"

REFUSAL_TEXT = "<|refuse|>"

GROUNDING_PREFIXES = [
  "Based on the context, ",
  "According to the context, ",
]

LOWERCASE_STARTERS = {"the", "a", "an", "it", "its", "he", "she", "they", "this", "that", "these", "those"}


def _grounding_prefix(example_id: str) -> str:
  """Deterministic prefix selection based on example ID hash."""
  return GROUNDING_PREFIXES[hash(example_id) % len(GROUNDING_PREFIXES)]


def _find_answer_sentence(context: str, answer_text: str) -> str:
  """Find the sentence in *context* that contains *answer_text*."""
  for sentence in sent_tokenize(context):
    if answer_text in sentence:
      return sentence

  return answer_text


def format_grounded_answer(example_id: str, context: str, answer_text: str) -> str:
  """Build a grounded answer with citation from an extractive span."""
  prefix = _grounding_prefix(example_id)
  sentence = _find_answer_sentence(context, answer_text).strip().rstrip(".")

  first_word = sentence.split()[0] if sentence.split() else ""

  if first_word.lower() in LOWERCASE_STARTERS:
    sentence = sentence[0].lower() + sentence[1:]

  return f"{prefix}{sentence}."


def convert_squad2(output_path: Path) -> dict:
  """Download and convert SQuAD 2.0 to unified format."""
  stats = {"total": 0, "answerable": 0, "unanswerable": 0}

  with open(output_path, "w", encoding="utf-8") as output_file:
    for split, url in SQUAD2_URLS.items():
      raw_path = download_file(url, RAW_DIR / f"squad2-{split}.json")
      data = json.loads(raw_path.read_text())

      for article in data["data"]:
        for paragraph in article["paragraphs"]:
          context = paragraph["context"]

          for qa_pair in paragraph["qas"]:
            qa_id = qa_pair["id"]
            example_id = f"squad2-{split}-{qa_id}"
            question = qa_pair["question"]

            if qa_pair.get("is_impossible", False) or not qa_pair.get("answers"):
              assistant_content = REFUSAL_TEXT
              stats["unanswerable"] += 1
            else:
              answer_text = qa_pair["answers"][0]["text"]
              assistant_content = format_grounded_answer(example_id, context, answer_text)
              stats["answerable"] += 1

            example = {
              "id": example_id,
              "source": "squad2",
              "context": context,
              "tools": [],
              "conversation": [
                {"role": "user", "content": question},
                {"role": "assistant", "content": assistant_content},
              ],
            }
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
            stats["total"] += 1

  log.info(
    "SQuAD 2.0: %d total (%d answerable, %d unanswerable)",
    stats["total"],
    stats["answerable"],
    stats["unanswerable"],
  )

  return stats


def convert_coqa(output_path: Path) -> dict:
  """Download and convert CoQA to unified multi-turn format."""
  stats = {"total": 0, "conversations": 0, "turns": 0, "unknown": 0}

  with open(output_path, "w", encoding="utf-8") as output_file:
    for split, url in COQA_URLS.items():
      raw_path = download_file(url, RAW_DIR / f"coqa-{split}.json")
      data = json.loads(raw_path.read_text())

      for story in data["data"]:
        story_id = story["id"]
        context = story["story"]
        questions = story["questions"]
        answers = story["answers"]

        conversation: list[dict] = []

        for question_entry, answer_entry in zip(questions, answers, strict=False):
          question_text = question_entry["input_text"]
          answer_text = answer_entry["input_text"]
          turn_id = question_entry["turn_id"]
          example_id = f"coqa-{split}-{story_id}-t{turn_id}"

          conversation.append({"role": "user", "content": question_text})

          if answer_text.lower().strip() == "unknown":
            conversation.append({"role": "assistant", "content": REFUSAL_TEXT})
            stats["unknown"] += 1
          else:
            grounded = format_grounded_answer(example_id, context, answer_text)
            conversation.append({"role": "assistant", "content": grounded})

          stats["turns"] += 1

        if not conversation:
          continue

        example = {
          "id": f"coqa-{split}-{story_id}",
          "source": "coqa",
          "context": context,
          "tools": [],
          "conversation": conversation,
        }

        output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

        stats["total"] += 1
        stats["conversations"] += 1

  stats["answerable"] = stats["total"]
  stats["unanswerable"] = 0

  log.info(
    "CoQA: %d conversations (%d turns total, %d unknown answers)",
    stats["conversations"],
    stats["turns"],
    stats["unknown"],
  )

  return stats


def _format_drop_answer(answer: dict) -> str | None:
  """Extract a text answer from a DROP answer object."""
  if answer.get("number") and answer["number"].strip():
    return answer["number"].strip()

  date = answer.get("date", {})
  date_parts = [date.get("month", ""), date.get("day", ""), date.get("year", "")]
  date_parts = [part for part in date_parts if part.strip()]

  if date_parts:
    return " ".join(date_parts)

  spans = answer.get("spans", [])

  if spans:
    return "; ".join(spans)

  return None


def convert_drop(output_path: Path) -> dict:
  """Download and convert DROP to unified format."""
  stats = {"total": 0, "number": 0, "date": 0, "span": 0, "skipped": 0}

  zip_path = download_file(DROP_URL, RAW_DIR / "drop_dataset.zip")
  extract_dir = RAW_DIR / "drop_dataset"

  if not extract_dir.exists():
    log.info("Extracting DROP zip...")

    with zipfile.ZipFile(zip_path) as zip_ref:
      zip_ref.extractall(RAW_DIR)

  with open(output_path, "w", encoding="utf-8") as output_file:
    for split in ("train", "dev"):
      raw_path = extract_dir / f"drop_dataset_{split}.json"

      if not raw_path.exists():
        log.warning("Missing %s, skipping", raw_path.name)
        continue

      data = json.loads(raw_path.read_text())

      for _passage_id, entry in data.items():
        context = entry["passage"]

        for qa_pair in entry["qa_pairs"]:
          query_id = qa_pair["query_id"]
          question = qa_pair["question"]
          answer_obj = qa_pair["answer"]
          example_id = f"drop-{split}-{query_id}"

          answer_text = _format_drop_answer(answer_obj)

          if answer_text is None:
            stats["skipped"] += 1
            continue

          if answer_obj.get("number", "").strip():
            stats["number"] += 1
          elif any(answer_obj.get("date", {}).get(field, "").strip() for field in ("day", "month", "year")):
            stats["date"] += 1
          else:
            stats["span"] += 1

          assistant_content = format_grounded_answer(example_id, context, answer_text)

          example = {
            "id": example_id,
            "source": "drop",
            "context": context,
            "tools": [],
            "conversation": [
              {"role": "user", "content": question},
              {"role": "assistant", "content": assistant_content},
            ],
          }

          output_file.write(json.dumps(example, ensure_ascii=False) + "\n")

          stats["total"] += 1

  stats["answerable"] = stats["total"]
  stats["unanswerable"] = 0

  log.info(
    "DROP: %d total (%d number, %d date, %d span, %d skipped)",
    stats["total"],
    stats["number"],
    stats["date"],
    stats["span"],
    stats["skipped"],
  )

  return stats


def validate_jsonl(path: Path) -> int:
  """Validate unified-format JSONL. Returns count of valid records."""
  valid_count, error_count = 0, 0
  required_keys = {"id", "source", "context", "tools", "conversation"}

  with open(path, encoding="utf-8") as input_file:
    for line_number, line in enumerate(input_file, 1):
      try:
        obj = json.loads(line)
        missing = required_keys - obj.keys()

        if missing:
          raise ValueError(f"missing keys: {missing}")

        conv = obj["conversation"]

        if len(conv) < 2:
          raise ValueError(f"conversation has {len(conv)} turns, need >= 2")

        if conv[0]["role"] != "user":
          raise ValueError("first turn must be user")

        valid_count += 1
      except (json.JSONDecodeError, ValueError, KeyError) as exc:
        error_count += 1

        if error_count <= 3:
          log.warning("  line %d: %s", line_number, exc)

  log.info("Validated %s: %d valid, %d errors", path.name, valid_count, error_count)

  return valid_count


def spot_check(path: Path, num_examples: int = 5) -> None:
  """Print *num_examples* random examples from a JSONL file for manual review."""
  lines = path.read_text().splitlines()

  if not lines:
    log.warning("  (empty file)")
    return

  sample = random.sample(lines, min(num_examples, len(lines)))

  print(f"\n--- Spot check: {path.name} ({num_examples} random examples) ---")

  for index, line in enumerate(sample, 1):
    obj = json.loads(line)

    print(f"\n[{index}] id={obj['id']}  source={obj['source']}")

    context_preview = obj["context"][:120].replace("\n", " ")

    print(f"    context: {context_preview}...")

    for turn in obj["conversation"]:
      role = turn["role"].upper()
      content = turn["content"][:150].replace("\n", " ")
      print(f"    {role}: {content}")

  print()


def print_summary(all_stats: dict[str, dict]) -> None:
  """Print a summary table of all converted datasets."""
  print("\n" + "=" * 72)
  print(f"{'Dataset':<12} {'Total':>10} {'Answerable':>12} {'Unanswerable':>14}")
  print("-" * 72)

  grand = {"total": 0, "answerable": 0, "unanswerable": 0}

  for name, stats in all_stats.items():
    total = stats["total"]
    answerable = stats["answerable"]
    unanswerable = stats["unanswerable"]

    print(f"{name:<12} {total:>10,} {answerable:>12,} {unanswerable:>14,}")

    grand["total"] += total
    grand["answerable"] += answerable
    grand["unanswerable"] += unanswerable

  print("-" * 72)
  print(f"{'TOTAL':<12} {grand['total']:>10,} {grand['answerable']:>12,} {grand['unanswerable']:>14,}")
  print("=" * 72 + "\n")


CONVERTERS = {
  "squad2": ("squad2.jsonl", convert_squad2),
  "coqa": ("coqa.jsonl", convert_coqa),
  "drop": ("drop.jsonl", convert_drop),
}


def main():
  parser = argparse.ArgumentParser(description="Convert open datasets to unified training format")
  parser.add_argument("--dataset", choices=["squad2", "coqa", "drop", "all"], default="all")
  parser.add_argument("--spot-check", type=int, default=5, metavar="N", help="Print N random examples per dataset")

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  random.seed(42)

  ensure_nltk_data()

  RAW_DIR.mkdir(parents=True, exist_ok=True)
  OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

  datasets = list(CONVERTERS) if args.dataset == "all" else [args.dataset]
  all_stats: dict[str, dict] = {}

  for dataset_name in datasets:
    filename, converter = CONVERTERS[dataset_name]
    output_path = OUTPUT_DIR / filename

    log.info("Converting %s ...", dataset_name)

    stats = converter(output_path)
    all_stats[dataset_name] = stats

    validate_jsonl(output_path)

    if args.spot_check > 0:
      spot_check(output_path, args.spot_check)

  if len(all_stats) > 1:
    print_summary(all_stats)


if __name__ == "__main__":
  main()
