"""Generate synthetic training data via LLM APIs.

Supports tool-calling and multi-turn conversation generation using
Anthropic, OpenAI, MiniMax, Kimi, and GLM as providers.  All providers
use prompt caching (Anthropic via explicit cache_control, OpenAI-compatible
APIs via automatic prefix caching) to reduce input token costs.

Usage:
  uv run python -m data.scripts.generate_synthetic \
    --mode tools \
    --input data/sources/simplewiki_passages.jsonl \
    --output data/synthetic/tools.jsonl \
    --provider minimax \
    --concurrency 10 \
    --limit 12000

  uv run python -m data.scripts.generate_synthetic \
    --mode multiturn \
    --input data/sources/simplewiki_passages.jsonl \
    --output data/synthetic/multiturn.jsonl \
    --provider round-robin \
    --model anthropic=claude-haiku-4-5 \
    --model minimax=minimax-m2.5 \
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
  # Anthropic — cache_read = 10% of input, cache_write = 125% of input
  "claude-haiku-4-5": {"input": 1.00 / 1e6, "output": 5.00 / 1e6, "cache_read": 0.10 / 1e6, "cache_write": 1.25 / 1e6},
  "claude-sonnet-4-6": {
    "input": 3.00 / 1e6,
    "output": 15.00 / 1e6,
    "cache_read": 0.30 / 1e6,
    "cache_write": 3.75 / 1e6,
  },
  # OpenAI — cache_read = 50% of input (automatic prefix caching)
  "gpt-4o-mini": {"input": 0.15 / 1e6, "output": 0.60 / 1e6, "cache_read": 0.075 / 1e6},
  "gpt-5.4": {"input": 2.50 / 1e6, "output": 15.00 / 1e6, "cache_read": 1.25 / 1e6},
  # MiniMax — cached input $0.03/MTok
  "minimax-m2.5": {"input": 0.19 / 1e6, "output": 0.95 / 1e6, "cache_read": 0.03 / 1e6},
  # Kimi / Moonshot — cached input $0.10/MTok
  "kimi-k2.5": {"input": 0.60 / 1e6, "output": 2.00 / 1e6, "cache_read": 0.10 / 1e6},
  # GLM / Zhipu
  "glm-5": {"input": 0.72 / 1e6, "output": 2.30 / 1e6},
}


@dataclass
class CostTracker:
  """Accumulates token usage and estimated cost across requests."""

  input_tokens: int = 0
  output_tokens: int = 0
  requests: int = 0
  cost_usd: float = 0.0
  _by_model: dict[str, dict[str, float]] = field(default_factory=dict)

  def record(
    self,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
  ) -> None:
    self.input_tokens += input_tokens
    self.output_tokens += output_tokens
    self.requests += 1

    pricing = PRICING.get(model, {"input": 0.0, "output": 0.0})

    regular_input = input_tokens - cache_read_tokens - cache_creation_tokens

    cost = (
      max(0, regular_input) * pricing["input"]
      + output_tokens * pricing["output"]
      + cache_read_tokens * pricing.get("cache_read", pricing["input"])
      + cache_creation_tokens * pricing.get("cache_write", pricing["input"])
    )

    self.cost_usd += cost

    bucket = self._by_model.setdefault(
      model,
      {"input": 0, "output": 0, "cost": 0.0, "cache_read": 0, "cache_write": 0},
    )

    bucket["input"] += input_tokens
    bucket["output"] += output_tokens
    bucket["cost"] += cost
    bucket["cache_read"] += cache_read_tokens
    bucket["cache_write"] += cache_creation_tokens

  def summary(self) -> str:
    lines = [f"Total: {self.requests} reqs, {self.input_tokens} in / {self.output_tokens} out, ${self.cost_usd:.4f}"]

    for model, stats in self._by_model.items():
      cache_info = ""

      cache_read, cache_write = int(stats.get("cache_read", 0)), int(stats.get("cache_write", 0))

      if cache_read or cache_write:
        cache_info = f" (cache: {cache_read} read, {cache_write} write)"

      lines.append(
        f"  {model}: {int(stats['input'])} in / {int(stats['output'])} out, ${stats['cost']:.4f}{cache_info}"
      )

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

  model: str

  @abstractmethod
  async def generate(
    self,
    prompt: str,
    *,
    system: str | None = None,
  ) -> tuple[str, str, int, int, int, int]:
    """Send prompt, return (text, model, input_tok, output_tok, cache_read_tok, cache_create_tok)."""


DEFAULT_MODELS = {
  "anthropic": "claude-sonnet-4-6",
  "openai": "gpt-5.4",
  "minimax": "MiniMaxAI/MiniMax-M2.5",
  "kimi": "moonshotai/Kimi-K2.5",
  "glm": "zai-org/GLM-5",
}


class AnthropicProvider(LLMProvider):
  def __init__(self, model: str | None = None):
    import anthropic

    self.client = anthropic.AsyncAnthropic()
    self.model = model or DEFAULT_MODELS["anthropic"]

  async def generate(
    self,
    prompt: str,
    *,
    system: str | None = None,
  ) -> tuple[str, str, int, int, int, int]:
    import anthropic

    kwargs: dict[str, Any] = {
      "model": self.model,
      "max_tokens": 4096,
      "messages": [{"role": "user", "content": prompt}],
    }
    if system:
      kwargs["system"] = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}},
      ]

    for attempt in range(6):
      try:
        response = await self.client.messages.create(**kwargs)

        text = response.content[0].text
        usage = response.usage
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
        total_in = usage.input_tokens + cache_read + cache_create

        return text, self.model, total_in, usage.output_tokens, cache_read, cache_create
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


class OpenAICompatibleProvider(LLMProvider):
  """Provider for OpenAI and any OpenAI-compatible API (MiniMax, Kimi, GLM, …)."""

  def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None):
    import openai

    kwargs: dict[str, Any] = {}

    if base_url:
      kwargs["base_url"] = base_url

    if api_key is not None:
      kwargs["api_key"] = api_key

    self.client = openai.AsyncOpenAI(**kwargs)
    self.model = model

  async def generate(
    self,
    prompt: str,
    *,
    system: str | None = None,
  ) -> tuple[str, str, int, int, int, int]:
    import openai

    messages: list[dict[str, str]] = []

    if system:
      messages.append({"role": "system", "content": system})

    messages.append({"role": "user", "content": prompt})

    for attempt in range(6):
      try:
        response = await self.client.chat.completions.create(
          model=self.model,
          max_completion_tokens=4096,
          messages=messages,
        )

        text = response.choices[0].message.content or ""
        usage = response.usage

        cache_read = 0

        if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
          cache_read = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

        return text, self.model, usage.prompt_tokens, usage.completion_tokens, cache_read, 0
      except openai.RateLimitError:
        wait = min(2**attempt + random.random(), 60)
        log.warning("%s rate limit, retrying in %.1fs (attempt %d/6)", self.model, wait, attempt + 1)
        await asyncio.sleep(wait)
      except openai.APIStatusError as exc:
        if exc.status_code >= 500:
          wait = min(2**attempt + random.random(), 60)
          log.warning("%s %d error, retrying in %.1fs", self.model, exc.status_code, wait)
          await asyncio.sleep(wait)
        else:
          raise

    raise RuntimeError(f"{self.model}: max retries exceeded")


class OpenAIProvider(OpenAICompatibleProvider):
  def __init__(self, model: str | None = None):
    super().__init__(model=model or DEFAULT_MODELS["openai"])


class MiniMaxProvider(OpenAICompatibleProvider):
  def __init__(self, model: str | None = None):
    super().__init__(
      model=model or DEFAULT_MODELS["minimax"],
      base_url="https://inference.baseten.co/v1",
      api_key=os.environ.get("BASETEN_API_KEY", ""),
    )


class KimiProvider(OpenAICompatibleProvider):
  def __init__(self, model: str | None = None):
    super().__init__(
      model=model or DEFAULT_MODELS["kimi"],
      base_url="https://inference.baseten.co/v1",
      api_key=os.environ.get("BASETEN_API_KEY", ""),
    )


class GLMProvider(OpenAICompatibleProvider):
  def __init__(self, model: str | None = None):
    super().__init__(
      model=model or DEFAULT_MODELS["glm"],
      base_url="https://inference.baseten.co/v1",
      api_key=os.environ.get("BASETEN_API_KEY", ""),
    )


class RoundRobinProvider(LLMProvider):
  """Alternates between available providers to stay under per-provider rate limits."""

  def __init__(self, models: dict[str, str] | None = None):
    self._providers: list[LLMProvider] = []
    self._idx = 0
    models = models or {}

    provider_env_keys: list[tuple[str, type[LLMProvider], str]] = [
      ("ANTHROPIC_API_KEY", AnthropicProvider, "anthropic"),
      ("OPENAI_API_KEY", OpenAIProvider, "openai"),
      ("BASETEN_API_KEY", MiniMaxProvider, "minimax"),
      ("BASETEN_API_KEY", KimiProvider, "kimi"),
      ("BASETEN_API_KEY", GLMProvider, "glm"),
    ]

    for env_key, cls, name in provider_env_keys:
      if os.environ.get(env_key):
        self._providers.append(cls(model=models.get(name)))

    if not self._providers:
      raise RuntimeError("Round-robin requires at least one provider API key in env")

    self.model = f"round-robin({','.join(p.model for p in self._providers)})"

    log.info("Round-robin with %d providers: %s", len(self._providers), [p.model for p in self._providers])

  async def generate(
    self,
    prompt: str,
    *,
    system: str | None = None,
  ) -> tuple[str, str, int, int, int, int]:
    provider = self._providers[self._idx % len(self._providers)]
    self._idx += 1
    return await provider.generate(prompt, system=system)


def _parse_model_args(raw: list[str] | None) -> tuple[str | None, dict[str, str]]:
  """Parse --model values into (single_model, per_provider_models).

  Plain values like "gpt-5.4" become the single model.
  Values like "openai=gpt-5.4" populate the per-provider dict.
  """
  if not raw:
    return None, {}

  single: str | None = None
  per_provider: dict[str, str] = {}

  for entry in raw:
    if "=" in entry:
      provider, _, model = entry.partition("=")
      per_provider[provider.strip()] = model.strip()
    else:
      single = entry.strip()

  return single, per_provider


PROVIDER_FACTORIES: dict[str, type[LLMProvider]] = {
  "anthropic": AnthropicProvider,
  "openai": OpenAIProvider,
  "minimax": MiniMaxProvider,
  "kimi": KimiProvider,
  "glm": GLMProvider,
}

PROVIDER_CHOICES = [*PROVIDER_FACTORIES, "round-robin"]


def make_provider(name: str, model: str | None = None, models: dict[str, str] | None = None) -> LLMProvider:
  if name == "round-robin":
    return RoundRobinProvider(models=models)

  if name not in PROVIDER_FACTORIES:
    raise ValueError(f"Unknown provider: {name}. Choose from: {', '.join(PROVIDER_CHOICES)}")

  return PROVIDER_FACTORIES[name](model=model)


def extract_json(text: str) -> Any:
  """Extract a JSON array or object from LLM response text, handling markdown fences."""
  text = text.strip()

  if text.startswith("```"):
    lines = text.split("\n")

    # drop opening fence
    lines = lines[1:]

    if lines and lines[-1].strip() == "```":
      lines = lines[:-1]

    text = "\n".join(lines).strip()

  return json.loads(text)


TOOLS_SYSTEM_PROMPT = """\
You are generating training data for a model that can either answer from context
or call tools. Given a context passage provided by the user and the available
tools below, generate exactly 7 examples:

- 4 questions where the correct response is a tool_call (the context does NOT contain the answer, but a tool can help)
- 3 questions where the answer IS in the context (no tool needed)

CRITICAL RULES:
- Before making a tool_call, verify the context does NOT already answer the question.
  If ANY part of the context addresses the question, it MUST be a context_answer, not a tool_call.
- Tool-call questions should be things a user might naturally ask that genuinely require
  external data (live weather, calculations, translations) — not questions the passage covers.

For tool calls, the response must be a JSON object: {{"tool_call": {{"name": "...", "arguments": {{...}}}}}}
For context answers, respond with a grounded answer citing the context with [source: sentence N].

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

TOOLS_USER_PROMPT = "Context:\n{passage_text}"


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


MULTITURN_SYSTEM_PROMPT = """\
You are generating training data for a conversational reading comprehension model.
Given a document provided by the user, simulate a realistic 4-6 turn conversation
between a user and an assistant.

Rules:
- The assistant ALWAYS grounds responses in the document with [source: sentence N] citations
- Include follow-up questions that refer to earlier answers ("you mentioned...",
  "what about the other...", "can you elaborate on...")
- Include at least ONE turn where the user asks something NOT in the document —
  the assistant should respond: "I don't have enough information in the provided
  context to answer that question."
- Make the conversation feel natural, not like a quiz

Respond with ONLY a JSON array of conversation turns (no other text):
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}},
  ...
]"""

MULTITURN_USER_PROMPT = "Document:\n{passage_text}"

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
  system_prompt: str | None,
) -> int:
  """Process a single passage. Returns number of examples written."""
  pid = passage["id"]

  if progress.is_done(pid):
    return 0

  if mode == "tools":
    user_prompt = TOOLS_USER_PROMPT.format(passage_text=passage["text"])
  else:
    user_prompt = MULTITURN_USER_PROMPT.format(passage_text=passage["text"])

  try:
    response_text, model, in_tok, out_tok, cache_read, cache_write = await provider.generate(
      user_prompt,
      system=system_prompt,
    )

    cost.record(model, in_tok, out_tok, cache_read, cache_write)
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

  if mode == "tools":
    system_prompt = TOOLS_SYSTEM_PROMPT.format(
      tool_schemas_json=json.dumps(tool_schemas, indent=2),
    )
  else:
    system_prompt = MULTITURN_SYSTEM_PROMPT

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
          system_prompt,
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
    choices=PROVIDER_CHOICES,
    help="LLM provider (default: round-robin)",
  )
  parser.add_argument(
    "--model",
    action="append",
    default=None,
    help="Model override. Plain value for single providers (e.g. gpt-5.4). "
    "For round-robin use PROVIDER=MODEL (repeatable, e.g. --model anthropic=claude-sonnet-4-6 --model openai=gpt-5.4)",
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

  model, models = _parse_model_args(args.model)
  provider = make_provider(args.provider, model=model, models=models)
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
