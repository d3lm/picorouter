"""Generate synthetic training data via LLM APIs.

Supports tool-calling and multi-turn conversation generation using
Anthropic (Claude 3.5 Haiku) and OpenAI (GPT-4o-mini) as providers.

Usage:
  uv run python -m data.scripts.generate_synthetic \
    --mode tools \
    --input data/sources/simplewiki_passages.jsonl \
    --output data/synthetic/tools.jsonl \
    --provider round-robin \
    --concurrency 10 \
    --limit 12000

  uv run python -m data.scripts.generate_synthetic \
    --mode multiturn \
    --input data/sources/simplewiki_passages.jsonl \
    --output data/synthetic/multiturn.jsonl \
    --provider anthropic \
    --concurrency 10 \
    --limit 20000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import sys
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]

TOOL_SCHEMAS_PATH = ROOT / "data" / "tool_schemas.json"

PRICING = {
  "claude-haiku-4-5-20251001": {"input": 1.00 / 1_000_000, "output": 5.00 / 1_000_000},
  "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}


@dataclass
class CostTracker:
  """Accumulates token usage and estimated cost across requests."""

  input_tokens: int = 0
  output_tokens: int = 0
  requests: int = 0
  cost_usd: float = 0.0
  _by_model: dict[str, dict[str, float]] = field(default_factory=dict)

  def record(self, model: str, input_tokens: int, output_tokens: int) -> None:
    self.input_tokens += input_tokens
    self.output_tokens += output_tokens
    self.requests += 1
    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})
    cost = input_tokens * pricing["input"] + output_tokens * pricing["output"]
    self.cost_usd += cost
    bucket = self._by_model.setdefault(model, {"input": 0, "output": 0, "cost": 0.0})
    bucket["input"] += input_tokens
    bucket["output"] += output_tokens
    bucket["cost"] += cost

  def summary(self) -> str:
    lines = [f"Total: {self.requests} reqs, {self.input_tokens} in / {self.output_tokens} out, ${self.cost_usd:.4f}"]

    for model, stats in self._by_model.items():
      lines.append(f"  {model}: {int(stats['input'])} in / {int(stats['output'])} out, ${stats['cost']:.4f}")

    return "\n".join(lines)


class ProgressTracker:
  """Track completed passage IDs in a sidecar file so we can resume after interrupt."""

  def __init__(self, output_path: Path):
    self.sidecar = output_path.with_suffix(".progress.jsonl")
    self.completed: set[str] = set()

    if self.sidecar.exists():
      import contextlib

      with open(self.sidecar, encoding="utf-8") as file:
        for line in file:
          line = line.strip()

          if line:
            with contextlib.suppress(json.JSONDecodeError, KeyError):
              self.completed.add(json.loads(line)["id"])

      log.info("Resuming: %d passages already completed", len(self.completed))

  def is_done(self, passage_id: str) -> bool:
    return passage_id in self.completed

  def mark_done(self, passage_id: str) -> None:
    self.completed.add(passage_id)

    with open(self.sidecar, "a", encoding="utf-8") as f:
      f.write(json.dumps({"id": passage_id}) + "\n")


class LLMProvider(ABC):
  """Base class for LLM API providers."""

  @abstractmethod
  async def generate(self, prompt: str) -> tuple[str, str, int, int]:
    """Send prompt, return (response_text, model_name, input_tokens, output_tokens)."""


class AnthropicProvider(LLMProvider):
  def __init__(self):
    import anthropic

    self.client = anthropic.AsyncAnthropic()
    self.model = "claude-haiku-4-5-20251001"

  async def generate(self, prompt: str) -> tuple[str, str, int, int]:
    import anthropic

    for attempt in range(6):
      try:
        response = await self.client.messages.create(
          model=self.model,
          max_tokens=2048,
          messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text

        return text, self.model, response.usage.input_tokens, response.usage.output_tokens
      except anthropic.RateLimitError:
        wait = min(2**attempt + random.random(), 60)
        log.warning("Anthropic rate limit, retrying in %.1fs (attempt %d/6)", wait, attempt + 1)
        await asyncio.sleep(wait)
      except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
          wait = min(2**attempt + random.random(), 60)
          log.warning("Anthropic %d error, retrying in %.1fs", exc.status_code, wait)
          await asyncio.sleep(wait)
        else:
          raise

    raise RuntimeError("Anthropic: max retries exceeded")


class OpenAIProvider(LLMProvider):
  def __init__(self):
    import openai

    self.client = openai.AsyncOpenAI()
    self.model = "gpt-4o-mini"

  async def generate(self, prompt: str) -> tuple[str, str, int, int]:
    import openai

    for attempt in range(6):
      try:
        response = await self.client.chat.completions.create(
          model=self.model,
          max_tokens=2048,
          messages=[{"role": "user", "content": prompt}],
        )

        text = response.choices[0].message.content or ""
        usage = response.usage

        return text, self.model, usage.prompt_tokens, usage.completion_tokens
      except openai.RateLimitError:
        wait = min(2**attempt + random.random(), 60)
        log.warning("OpenAI rate limit, retrying in %.1fs (attempt %d/6)", wait, attempt + 1)
        await asyncio.sleep(wait)
      except openai.APIStatusError as exc:
        if exc.status_code >= 500:
          wait = min(2**attempt + random.random(), 60)
          log.warning("OpenAI %d error, retrying in %.1fs", exc.status_code, wait)
          await asyncio.sleep(wait)
        else:
          raise

    raise RuntimeError("OpenAI: max retries exceeded")


class RoundRobinProvider(LLMProvider):
  """Alternates between Anthropic and OpenAI to stay under per-provider rate limits."""

  def __init__(self):
    self._providers: list[LLMProvider] = []
    self._idx = 0

    if os.environ.get("ANTHROPIC_API_KEY"):
      self._providers.append(AnthropicProvider())
    if os.environ.get("OPENAI_API_KEY"):
      self._providers.append(OpenAIProvider())

    if not self._providers:
      raise RuntimeError("Round-robin requires at least one of ANTHROPIC_API_KEY or OPENAI_API_KEY")

    log.info("Round-robin with %d providers: %s", len(self._providers), [type(p).__name__ for p in self._providers])

  async def generate(self, prompt: str) -> tuple[str, str, int, int]:
    provider = self._providers[self._idx % len(self._providers)]
    self._idx += 1
    return await provider.generate(prompt)


def make_provider(name: str) -> LLMProvider:
  if name == "anthropic":
    return AnthropicProvider()
  elif name == "openai":
    return OpenAIProvider()
  elif name == "round-robin":
    return RoundRobinProvider()
  else:
    raise ValueError(f"Unknown provider: {name}")


def extract_json(text: str) -> Any:
  """Extract a JSON array or object from LLM response text, handling markdown fences."""
  text = text.strip()

  if text.startswith("```"):
    lines = text.split("\n")
    lines = lines[1:]  # drop opening fence

    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]

    text = "\n".join(lines).strip()

  return json.loads(text)


TOOLS_PROMPT = """\
You are generating training data for a model that can either answer from context
or call tools. Given the context passage and available tools below, generate
exactly 7 examples:

- 4 questions where the correct response is a tool_call (the context does NOT contain the answer, but a tool can help)
- 3 questions where the answer IS in the context (no tool needed)

For tool calls, the response must be a JSON object: {{"tool_call": {{"name": "...", "arguments": {{...}}}}}}
For context answers, respond with a grounded answer citing the context with [source: sentence N].

Context:
{passage_text}

Available tools:
{tool_schemas_json}

Respond with ONLY a JSON array (no other text):
[
  {{
    "question": "...",
    "response_type": "tool_call",
    "response": {{"tool_call": {{"name": "...", "arguments": {{...}}}}}}
  }},
  {{
    "question": "...",
    "response_type": "context_answer",
    "response": "Based on the context, ... [source: sentence N]"
  }}
]"""


def _load_tool_schemas() -> list[dict]:
  with open(TOOL_SCHEMAS_PATH, encoding="utf-8") as file:
    return json.load(file)


def _validate_tool_call(tool_call: dict, schema_names: set[str], schema_params: dict[str, set[str]]) -> str | None:
  """Return an error string if the tool call is invalid, else None."""
  if "name" not in tool_call or "arguments" not in tool_call:
    return "missing name or arguments"

  if tool_call["name"] not in schema_names:
    return f"unknown tool: {tool_call['name']}"

  expected = schema_params.get(tool_call["name"], set())

  if expected and not set(tool_call["arguments"].keys()).issubset(expected):
    extra = set(tool_call["arguments"].keys()) - expected
    return f"unexpected args for {tool_call['name']}: {extra}"

  return None


def process_tools_response(
  raw: list[dict],
  passage: dict,
  tool_schemas: list[dict],
) -> list[dict]:
  """Convert parsed LLM response into unified training format for tool mode."""
  schema_names = {schema["name"] for schema in tool_schemas}
  schema_params = {schema["name"]: set(schema.get("parameters", {}).keys()) for schema in tool_schemas}

  examples = []

  for item in raw:
    rtype = item.get("response_type")
    question = item.get("question", "")
    response = item.get("response")

    if not question or response is None:
      continue

    if rtype == "tool_call":
      tool_call = response.get("tool_call") if isinstance(response, dict) else None

      if tool_call is None:
        continue

      err = _validate_tool_call(tool_call, schema_names, schema_params)

      if err:
        log.debug("Skipping invalid tool call: %s", err)
        continue

      examples.append(
        {
          "context": passage["text"],
          "tools": tool_schemas,
          "conversation": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"<|tool_call|>{json.dumps(tool_call)}"},
          ],
          "source": "synthetic-tools",
        }
      )
    elif rtype == "context_answer":
      if isinstance(response, str) and "[source:" in response:
        examples.append(
          {
            "context": passage["text"],
            "tools": tool_schemas,
            "conversation": [
              {"role": "user", "content": question},
              {"role": "assistant", "content": response},
            ],
            "source": "synthetic-tools",
          }
        )

  return examples


MULTITURN_PROMPT = """\
You are generating training data for a conversational reading comprehension model.
Given the document below, simulate a realistic 4-6 turn conversation between a
user and an assistant.

Rules:
- The assistant ALWAYS grounds responses in the document with [source: sentence N] citations
- Include follow-up questions that refer to earlier answers ("you mentioned...",
  "what about the other...", "can you elaborate on...")
- Include at least ONE turn where the user asks something NOT in the document —
  the assistant should respond: "I don't have enough information in the provided
  context to answer that question."
- Make the conversation feel natural, not like a quiz

Document:
{passage_text}

Respond with ONLY a JSON array of conversation turns (no other text):
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]"""

REFUSAL_PHRASE = "I don't have enough information in the provided context"


def process_multiturn_response(raw: list[dict], passage: dict) -> dict | None:
  """Convert parsed LLM response into unified training format for multiturn mode."""
  if len(raw) < 4:
    return None

  for i, turn in enumerate(raw):
    if "role" not in turn or "content" not in turn:
      return None

    expected_role = "user" if i % 2 == 0 else "assistant"

    if turn["role"] != expected_role:
      return None

  has_refusal = any(turn["role"] == "assistant" and REFUSAL_PHRASE.lower() in turn["content"].lower() for turn in raw)

  has_citation = any(turn["role"] == "assistant" and "[source:" in turn["content"] for turn in raw)

  if not has_refusal or not has_citation:
    return None

  conversation = []

  for turn in raw:
    content = turn["content"]

    if turn["role"] == "assistant" and REFUSAL_PHRASE.lower() in content.lower():
      content = f"<|refuse|>{content}"

    conversation.append({"role": turn["role"], "content": content})

  return {
    "context": passage["text"],
    "tools": [],
    "conversation": conversation,
    "source": "synthetic-multiturn",
  }


async def process_passage(
  passage: dict,
  mode: str,
  provider: LLMProvider,
  cost: CostTracker,
  progress: ProgressTracker,
  output_file,
  write_lock: asyncio.Lock,
  tool_schemas: list[dict] | None,
  errors_file,
) -> int:
  """Process a single passage. Returns number of examples written."""
  pid = passage["id"]

  if progress.is_done(pid):
    return 0

  if mode == "tools":
    prompt = TOOLS_PROMPT.format(
      passage_text=passage["text"],
      tool_schemas_json=json.dumps(tool_schemas, indent=2),
    )
  else:
    prompt = MULTITURN_PROMPT.format(passage_text=passage["text"])

  try:
    response_text, model, in_tok, out_tok = await provider.generate(prompt)
    cost.record(model, in_tok, out_tok)
  except Exception as exc:
    async with write_lock:
      errors_file.write(json.dumps({"id": pid, "error": f"api: {exc}", "mode": mode}) + "\n")
      errors_file.flush()

    return 0

  try:
    parsed = extract_json(response_text)
  except (json.JSONDecodeError, ValueError) as exc:
    async with write_lock:
      errors_file.write(
        json.dumps(
          {
            "id": pid,
            "error": f"json_parse: {exc}",
            "mode": mode,
            "raw": response_text[:500],
          }
        )
        + "\n"
      )

      errors_file.flush()

    return 0

  if not isinstance(parsed, list):
    async with write_lock:
      errors_file.write(
        json.dumps(
          {
            "id": pid,
            "error": "expected JSON array",
            "mode": mode,
          }
        )
        + "\n"
      )

      errors_file.flush()

    return 0

  if mode == "tools":
    examples = process_tools_response(parsed, passage, tool_schemas)
  else:
    result = process_multiturn_response(parsed, passage)
    examples = [result] if result else []

  count = 0

  async with write_lock:
    for ex in examples:
      output_file.write(json.dumps(ex, ensure_ascii=False) + "\n")
      count += 1

    output_file.flush()

    progress.mark_done(pid)

  return count


async def run_pipeline(
  passages: list[dict],
  mode: str,
  provider: LLMProvider,
  concurrency: int,
  output_path: Path,
  errors_path: Path,
) -> None:
  cost = CostTracker()
  progress = ProgressTracker(output_path)

  remaining = [p for p in passages if not progress.is_done(p["id"])]

  log.info(
    "Mode=%s | %d passages total, %d remaining, concurrency=%d",
    mode,
    len(passages),
    len(remaining),
    concurrency,
  )

  if not remaining:
    log.info("Nothing to do — all passages already processed")
    return

  sem = asyncio.Semaphore(concurrency)
  write_lock = asyncio.Lock()
  total_examples = 0
  processed = 0
  t0 = time.monotonic()

  output_path.parent.mkdir(parents=True, exist_ok=True)
  errors_path.parent.mkdir(parents=True, exist_ok=True)

  with (
    open(output_path, "a", encoding="utf-8") as output_file,
    open(errors_path, "a", encoding="utf-8") as errors_file,
  ):

    async def bounded(passage):
      nonlocal total_examples, processed

      async with sem:
        n = await process_passage(
          passage,
          mode,
          provider,
          cost,
          progress,
          output_file,
          write_lock,
          tool_schemas,
          errors_file,
        )

        total_examples += n
        processed += 1

        if processed % 100 == 0:
          elapsed = time.monotonic() - t0

          rate = processed / elapsed if elapsed > 0 else 0

          log.info(
            "  [%d/%d] %d examples | %.1f passages/s | %s",
            processed,
            len(remaining),
            total_examples,
            rate,
            cost.summary().split("\n")[0],
          )

    tasks = [asyncio.create_task(bounded(p)) for p in remaining]

    await asyncio.gather(*tasks, return_exceptions=True)

  elapsed = time.monotonic() - t0

  log.info("Done in %.1fs. %d examples from %d passages.", elapsed, total_examples, processed)
  log.info("Cost breakdown:\n%s", cost.summary())

  from data.scripts.validate_synthetic import validate

  log.info("Running post-generation validation...\n")

  report = validate(output_path, mode)

  report.print(file=sys.stderr)

  if not report.ok:
    log.error("Validation FAILED — review errors above before scaling up")
  else:
    log.info("Validation passed (%d warnings)", len(report.warnings))


tool_schemas: list[dict] | None = None


def load_passages(input_path: Path, limit: int | None) -> list[dict]:
  passages = []

  with open(input_path, encoding="utf-8") as file:
    for line in file:
      line = line.strip()

      if not line:
        continue

      passages.append(json.loads(line))

      if limit and len(passages) >= limit:
        break

  return passages


def _load_dotenv() -> None:
  """Load .env from project root into os.environ (no third-party dependency)."""
  env_path = ROOT / ".env"

  if not env_path.exists():
    return

  with open(env_path, encoding="utf-8") as file:
    for line in file:
      line = line.strip()

      if not line or line.startswith("#") or "=" not in line:
        continue

      key, _, value = line.partition("=")
      key = key.strip()

      value = value.strip().strip("\"'")

      if key and key not in os.environ:
        os.environ[key] = value


def main():
  global tool_schemas

  _load_dotenv()

  parser = argparse.ArgumentParser(
    description="Generate synthetic training data via LLM APIs",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument("--mode", required=True, choices=["tools", "multiturn"], help="Generation mode")
  parser.add_argument("--input", required=True, type=Path, help="Input passage JSONL file")
  parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
  parser.add_argument(
    "--provider",
    default="round-robin",
    choices=["anthropic", "openai", "round-robin"],
    help="LLM provider (default: round-robin)",
  )
  parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent API requests (default: 10)")
  parser.add_argument("--limit", type=int, default=None, help="Max passages to process")
  parser.add_argument("--errors", type=Path, default=None, help="Error log file (default: <output>.errors.jsonl)")

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
  )

  if not args.input.exists():
    log.error("Input file not found: %s", args.input)
    sys.exit(1)

  if args.mode == "tools":
    if not TOOL_SCHEMAS_PATH.exists():
      log.error("Tool schemas not found: %s", TOOL_SCHEMAS_PATH)
      sys.exit(1)

    tool_schemas = _load_tool_schemas()

    log.info("Loaded %d tool schemas", len(tool_schemas))

  errors_path = args.errors or args.output.with_suffix(".errors.jsonl")

  provider = make_provider(args.provider)
  passages = load_passages(args.input, args.limit)

  log.info("Loaded %d passages from %s", len(passages), args.input)

  asyncio.run(
    run_pipeline(
      passages=passages,
      mode=args.mode,
      provider=provider,
      concurrency=args.concurrency,
      output_path=args.output,
      errors_path=errors_path,
    )
  )


if __name__ == "__main__":
  main()
