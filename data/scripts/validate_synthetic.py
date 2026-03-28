"""Validate synthetic training data files produced by generate_synthetic.py.

Runs structural and semantic checks on JSONL output, reporting errors
(hard failures that indicate data corruption) and warnings (soft issues
worth investigating at scale).

Usage:
  uv run python -m data.scripts.validate_synthetic \
    --mode tools \
    --input data/synthetic/tools.jsonl

  uv run python -m data.scripts.validate_synthetic \
    --mode multiturn \
    --input data/synthetic/multiturn.jsonl

Can also be called programmatically:

  from data.scripts.validate_synthetic import validate
  report = validate(Path("data/synthetic/tools.jsonl"), mode="tools")
  report.print()
  assert report.ok
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
TOOL_SCHEMAS_PATH = ROOT / "data" / "tool_schemas.json"

REFUSAL_PHRASE = "I don't have enough information in the provided context"
STOP_WORDS = frozenset(
  [
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "in",
    "on",
    "at",
    "to",
    "of",
    "and",
    "or",
    "it",
    "that",
    "this",
    "for",
    "with",
    "as",
    "by",
    "from",
    "be",
    "has",
    "have",
    "had",
    "not",
    "but",
    "can",
    "do",
  ]
)


@dataclass
class ValidationReport:
  errors: list[str] = field(default_factory=list)
  warnings: list[str] = field(default_factory=list)
  total: int = 0
  tool_calls: int = 0
  context_answers: int = 0
  unique_contexts: int = 0
  tools_used: dict[str, int] = field(default_factory=lambda: collections.Counter())
  duplicate_questions: dict[str, int] = field(default_factory=dict)

  @property
  def ok(self) -> bool:
    return len(self.errors) == 0

  def print(self, file=sys.stdout) -> None:
    write = file.write

    write("=" * 60 + "\n")
    write("VALIDATION REPORT\n")
    write("=" * 60 + "\n\n")

    write(f"Total examples:      {self.total}\n")
    write(f"Unique contexts:     {self.unique_contexts}\n")

    if self.unique_contexts:
      write(f"Examples/context:    {self.total / self.unique_contexts:.1f}\n")

    if self.tool_calls or self.context_answers:
      tool_call_percent = self.tool_calls / self.total * 100 if self.total else 0
      context_answer_percent = self.context_answers / self.total * 100 if self.total else 0

      write(f"Tool calls:          {self.tool_calls} ({tool_call_percent:.1f}%, target ~57.1%)\n")
      write(f"Context answers:     {self.context_answers} ({context_answer_percent:.1f}%, target ~42.9%)\n")

    if self.tools_used:
      write("\nTool usage:\n")

      for name, count in sorted(self.tools_used.items(), key=lambda item: -item[1]):
        write(f"  {name}: {count}\n")

    if self.duplicate_questions:
      write(f"\nDuplicate questions: {len(self.duplicate_questions)}\n")

      for question, count in list(self.duplicate_questions.items())[:5]:
        write(f"  ({count}x) {question[:80]}\n")

    write(f"\nErrors ({len(self.errors)}):\n")
    if self.errors:
      for error in self.errors[:20]:
        write(f"  {error}\n")

      if len(self.errors) > 20:
        write(f"  ... and {len(self.errors) - 20} more\n")
    else:
      write("  None\n")

    write(f"\nWarnings ({len(self.warnings)}):\n")

    if self.warnings:
      for warning in self.warnings[:20]:
        write(f"  {warning}\n")

      if len(self.warnings) > 20:
        write(f"  ... and {len(self.warnings) - 20} more\n")
    else:
      write("  None\n")

    write("\n" + "=" * 60 + "\n")

    status = "PASS" if self.ok else "FAIL"

    write(f"RESULT: {status} ({len(self.errors)} errors, {len(self.warnings)} warnings)\n")


def _load_canonical_tools() -> tuple[list[dict], str]:
  with open(TOOL_SCHEMAS_PATH, encoding="utf-8") as file:
    schemas = json.load(file)

  return schemas, json.dumps(schemas, sort_keys=True)


def _validate_tools_row(
  row: dict,
  lineno: int,
  canonical_json: str,
  schema_names: set[str],
  schema_params: dict[str, set[str]],
  report: ValidationReport,
) -> str | None:
  """Validate a single row from --mode tools output."""
  row_tools_json = json.dumps(row.get("tools", []), sort_keys=True)

  if row_tools_json != canonical_json:
    report.errors.append(f"L{lineno}: tools array doesn't match canonical tool_schemas.json")

  conversation = row.get("conversation", [])

  if not isinstance(conversation, list) or len(conversation) != 2:
    report.errors.append(
      f"L{lineno}: conversation should have exactly 2 turns, "
      f"got {len(conversation) if isinstance(conversation, list) else type(conversation).__name__}"
    )

    return

  if conversation[0].get("role") != "user":
    report.errors.append(f"L{lineno}: first turn role should be 'user', got '{conversation[0].get('role')}'")

  if conversation[1].get("role") != "assistant":
    report.errors.append(f"L{lineno}: second turn role should be 'assistant', got '{conversation[1].get('role')}'")

  user_question = conversation[0].get("content", "")
  assistant_response = conversation[1].get("content", "")

  if not user_question or not isinstance(user_question, str):
    report.errors.append(f"L{lineno}: empty or non-string user question")

  if not assistant_response or not isinstance(assistant_response, str):
    report.errors.append(f"L{lineno}: empty or non-string assistant response")
    return

  if "<|tool_call|>" in assistant_response:
    report.tool_calls += 1
    tool_call_text = assistant_response.split("<|tool_call|>", 1)[1]

    try:
      tool_call = json.loads(tool_call_text)
    except json.JSONDecodeError as exc:
      report.errors.append(f"L{lineno}: tool_call JSON parse error: {exc}")
      return

    if "name" not in tool_call or "arguments" not in tool_call:
      report.errors.append(f"L{lineno}: tool_call missing 'name' or 'arguments'")
      return

    if tool_call["name"] not in schema_names:
      report.errors.append(f"L{lineno}: unknown tool '{tool_call['name']}'")
      return

    report.tools_used[tool_call["name"]] += 1

    actual_params = set(tool_call["arguments"].keys())
    expected_params = schema_params.get(tool_call["name"], set())
    extra = actual_params - expected_params

    if extra:
      report.errors.append(f"L{lineno}: tool '{tool_call['name']}' has unexpected args: {extra}")

    missing = expected_params - actual_params

    if missing:
      report.warnings.append(f"L{lineno}: tool '{tool_call['name']}' missing params: {missing}")

    if "[source:" in assistant_response:
      report.warnings.append(f"L{lineno}: tool_call also contains [source:] citation")
  else:
    report.context_answers += 1

    if "[source:" not in assistant_response:
      report.errors.append(f"L{lineno}: context_answer missing [source:] citation")

    context = row.get("context", "")
    context_words = set(context.lower().split()) - STOP_WORDS
    answer_words = set(assistant_response.lower().split()) - STOP_WORDS

    if len(context_words & answer_words) < 3:
      report.warnings.append(f"L{lineno}: context_answer has very little overlap with context")

  return user_question


def _validate_multiturn_row(row: dict, lineno: int, report: ValidationReport) -> None:
  """Validate a single row from --mode multiturn output."""
  conversation = row.get("conversation", [])

  if not isinstance(conversation, list) or len(conversation) < 4:
    report.errors.append(
      f"L{lineno}: conversation should have >=4 turns, got {len(conversation) if isinstance(conversation, list) else 0}"
    )

    return

  if len(conversation) % 2 != 0:
    report.warnings.append(
      f"L{lineno}: odd number of turns ({len(conversation)}), expected even (user/assistant pairs)"
    )

  for index, turn in enumerate(conversation):
    if "role" not in turn or "content" not in turn:
      report.errors.append(f"L{lineno}: turn {index} missing 'role' or 'content'")
      continue

    expected_role = "user" if index % 2 == 0 else "assistant"
    if turn["role"] != expected_role:
      report.errors.append(f"L{lineno}: turn {index} role is '{turn['role']}', expected '{expected_role}'")

  has_refusal = any(
    turn.get("role") == "assistant" and REFUSAL_PHRASE.lower() in turn.get("content", "").lower()
    for turn in conversation
  )

  has_citation = any(turn.get("role") == "assistant" and "[source:" in turn.get("content", "") for turn in conversation)

  has_refuse_tag = any(
    turn.get("role") == "assistant" and turn.get("content", "").startswith("<|refuse|>") for turn in conversation
  )

  if not has_citation:
    report.errors.append(f"L{lineno}: no assistant turn contains a [source:] citation")

  if not has_refusal:
    report.errors.append(f"L{lineno}: no assistant turn contains the refusal phrase")

  if has_refusal and not has_refuse_tag:
    report.errors.append(f"L{lineno}: refusal turn missing <|refuse|> prefix")

  tools = row.get("tools", [])

  if tools:
    report.warnings.append(f"L{lineno}: multiturn row has non-empty tools array")


def validate(input_path: Path, mode: str) -> ValidationReport:
  """Run all validation checks against a synthetic data JSONL file.

  Returns a ValidationReport with errors and warnings.
  """
  report = ValidationReport()

  canonical_schemas = None
  canonical_json = None
  schema_names: set[str] = set()
  schema_params: dict[str, set[str]] = {}

  if mode == "tools":
    if not TOOL_SCHEMAS_PATH.exists():
      report.errors.append(f"Tool schemas file not found: {TOOL_SCHEMAS_PATH}")
      return report

    canonical_schemas, canonical_json = _load_canonical_tools()
    schema_names = {schema["name"] for schema in canonical_schemas}
    schema_params = {schema["name"]: set(schema.get("parameters", {}).keys()) for schema in canonical_schemas}

  if not input_path.exists():
    report.errors.append(f"Input file not found: {input_path}")
    return report

  contexts: set[str] = set()
  all_questions: list[str] = []

  with open(input_path, encoding="utf-8") as file:
    for lineno, line in enumerate(file, 1):
      line = line.strip()

      if not line:
        continue

      try:
        row = json.loads(line)
      except json.JSONDecodeError as exc:
        report.errors.append(f"L{lineno}: invalid JSON: {exc}")
        continue

      report.total += 1

      for key in ("context", "tools", "conversation", "source"):
        if key not in row:
          report.errors.append(f"L{lineno}: missing required key '{key}'")

      expected_source = f"synthetic-{mode}"

      if row.get("source") != expected_source:
        report.errors.append(f"L{lineno}: source is '{row.get('source')}', expected '{expected_source}'")

      context = row.get("context", "")

      if not isinstance(context, str) or len(context) < 50:
        report.errors.append(
          f"L{lineno}: context too short or wrong type (len={len(context) if isinstance(context, str) else 'N/A'})"
        )

      contexts.add(context[:200])

      if mode == "tools":
        question = _validate_tools_row(row, lineno, canonical_json, schema_names, schema_params, report)

        if question:
          all_questions.append(question)
      elif mode == "multiturn":
        _validate_multiturn_row(row, lineno, report)

  report.unique_contexts = len(contexts)

  if mode == "tools":
    question_counts = collections.Counter(all_questions)
    report.duplicate_questions = {question: count for question, count in question_counts.items() if count > 1}

    if report.duplicate_questions:
      report.warnings.append(f"{len(report.duplicate_questions)} duplicate question(s)")

    if report.total > 0:
      tool_call_ratio = report.tool_calls / report.total

      if not (0.45 <= tool_call_ratio <= 0.70):
        report.warnings.append(f"Tool-call ratio {tool_call_ratio:.1%} is outside expected range [45%-70%]")

    if report.unique_contexts > 0:
      avg_yield = report.total / report.unique_contexts

      if avg_yield < 6.5:
        report.warnings.append(
          f"Average yield {avg_yield:.1f} examples/context is below 7 (some LLM responses may have been filtered)"
        )

    if canonical_schemas:
      unused = schema_names - set(report.tools_used.keys())

      if unused:
        report.warnings.append(f"Unused tools: {unused}")

  progress_path = input_path.with_suffix(".progress.jsonl")
  errors_path = input_path.with_suffix(".errors.jsonl")

  if progress_path.exists():
    with open(progress_path) as progress_file:
      progress_count = sum(1 for line in progress_file if line.strip())

    if progress_count != report.unique_contexts:
      report.warnings.append(
        f"Progress file has {progress_count} entries but data has {report.unique_contexts} unique contexts"
      )

  if errors_path.exists():
    with open(errors_path) as errors_file:
      error_count = sum(1 for line in errors_file if line.strip())

    if error_count > 0:
      report.warnings.append(f"Errors file has {error_count} entries — check {errors_path.name}")

  return report


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Validate synthetic training data",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument("--mode", required=True, choices=["tools", "multiturn"], help="Data mode to validate")
  parser.add_argument("--input", required=True, type=Path, help="JSONL file to validate")

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)

  report = validate(args.input, args.mode)

  report.print()

  sys.exit(0 if report.ok else 1)


if __name__ == "__main__":
  main()
