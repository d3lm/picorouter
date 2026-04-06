"""Merge all data sources, quality filter, train tokenizer, tokenize, and create splits.

Usage:
  uv run python -m data.scripts.merge_and_tokenize --step filter
  uv run python -m data.scripts.merge_and_tokenize --step tokenizer
  uv run python -m data.scripts.merge_and_tokenize --step process
  uv run python -m data.scripts.merge_and_tokenize --step all

Steps run sequentially: filter → tokenizer → process
Each step reads the output of the previous one.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "model"
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"

REFUSAL_PHRASE = "I don't have enough information in the provided context"

_CITATION_RE = re.compile(r"\s*\[source:[^\]]*\]\.?")

SPECIAL_TOKENS = [
  "<|pad|>",
  "<|eos|>",
  "<|context|>",
  "<|user|>",
  "<|assistant|>",
  "<|refuse|>",
]

ALL_SOURCES = {
  "squad2": DATA_DIR / "open_datasets" / "squad2.jsonl",
  "coqa": DATA_DIR / "open_datasets" / "coqa.jsonl",
  "drop": DATA_DIR / "open_datasets" / "drop.jsonl",
  "synthetic-offtopic-refusal": DATA_DIR / "synthetic" / "offtopic_refusal.jsonl",
}

FILTERED_PATH = PROCESSED_DIR / "filtered.jsonl"
FILTER_REPORT_PATH = PROCESSED_DIR / "filter_report.json"
STATS_PATH = PROCESSED_DIR / "stats.json"

MAX_SEQ_LEN = 1024
VOCAB_SIZE = 8192
RANDOM_SEED = 42
JACCARD_THRESHOLD = 0.85
MAX_QUESTION_DUPLICATES = 10
MIN_CONTEXT_WORDS = 20
MAX_CONTEXT_WORDS = 800


def strip_source_citations(rows: list[dict]) -> tuple[list[dict], int]:
  """Remove [source: ...] citations from context-answer assistant turns."""
  stripped = 0

  for row in rows:
    for turn in row.get("conversation", []):
      if turn["role"] != "assistant":
        continue

      content = turn["content"]

      if "<|refuse|>" in content:
        continue

      cleaned = _CITATION_RE.sub("", content).strip()
      cleaned = re.sub(r"  +", " ", cleaned)

      if not cleaned:
        continue

      if cleaned != content:
        turn["content"] = cleaned
        stripped += 1

  return rows, stripped


def load_all_sources() -> list[dict]:
  """Load every JSONL data source into a single list."""
  rows: list[dict] = []

  for source_name, path in ALL_SOURCES.items():
    if not path.exists():
      log.warning("Source not found, skipping: %s", path)
      continue

    count = 0

    with open(path, encoding="utf-8") as f:
      for line in f:
        line = line.strip()

        if not line:
          continue

        try:
          row = json.loads(line)
        except json.JSONDecodeError as exc:
          log.warning("Bad JSON in %s: %s", path.name, exc)
          continue

        if "source" not in row:
          row["source"] = source_name

        rows.append(row)
        count += 1

    log.info("Loaded %d rows from %s (%s)", count, source_name, path.name)

  return rows


def _count_words(text: str) -> int:
  return len(text.split())


def _check_row(row: dict) -> list[str]:
  """Run all automated checks on one row. Empty list = pass."""
  errors: list[str] = []

  for key in ("context", "conversation"):
    if key not in row:
      errors.append(f"missing '{key}'")

  if errors:
    return errors

  word_count = _count_words(row.get("context", ""))

  if word_count > MAX_CONTEXT_WORDS:
    errors.append(f"context {word_count} words exceeds {MAX_CONTEXT_WORDS}")
    return errors

  if word_count < MIN_CONTEXT_WORDS:
    errors.append(f"context {word_count} words below {MIN_CONTEXT_WORDS}")
    return errors

  conversation = row.get("conversation", [])

  if not isinstance(conversation, list) or len(conversation) < 2:
    errors.append(f"conversation too short ({len(conversation) if isinstance(conversation, list) else 0})")
    return errors

  if conversation[0].get("role") != "user":
    errors.append("doesn't start with 'user'")

  if conversation[-1].get("role") != "assistant":
    errors.append("doesn't end with 'assistant'")

  for i, turn in enumerate(conversation):
    expected = "user" if i % 2 == 0 else "assistant"

    if turn.get("role") != expected:
      errors.append(f"turn {i}: '{turn.get('role')}' != '{expected}'")
      break

  for turn in conversation:
    if turn.get("role") != "assistant":
      continue

    content = turn.get("content", "")

    if not content:
      errors.append("empty assistant content")
      continue

    if content.startswith("<|refuse|>") and REFUSAL_PHRASE.lower() not in content.lower():
      errors.append("refusal missing standard phrase")

  return errors


def run_automated_checks(rows: list[dict]) -> tuple[list[dict], dict]:
  """Run quality checks on all rows. Returns (passing, report)."""
  source_input = Counter(r.get("source", "unknown") for r in rows)
  reasons: Counter[str] = Counter()
  passing: list[dict] = []

  for row in rows:
    errs = _check_row(row)

    if errs:
      reasons[errs[0]] += 1
    else:
      passing.append(row)

  report = {
    "total_input": len(rows),
    "per_source_input": dict(source_input),
    "passed_checks": len(passing),
    "failed_checks": len(rows) - len(passing),
    "failure_reasons": dict(reasons.most_common(20)),
  }

  return passing, report


def exact_dedup(rows: list[dict]) -> tuple[list[dict], int]:
  """Exact-match dedup on (context, first_question). Returns (deduped, n_removed)."""
  seen: set[tuple[str, ...]] = set()
  deduped: list[dict] = []

  for row in rows:
    conv = row.get("conversation", [])
    ctx = row.get("context", "")
    question = conv[0].get("content", "") if conv else ""
    key = (ctx, question)

    if key in seen:
      continue

    seen.add(key)
    deduped.append(row)

  return deduped, len(rows) - len(deduped)


def question_cap_dedup(rows: list[dict], max_per_question: int = MAX_QUESTION_DUPLICATES) -> tuple[list[dict], int]:
  """Cap identical questions to at most *max_per_question* examples, keeping diverse contexts."""
  question_counts: Counter[str] = Counter()
  deduped: list[dict] = []

  for row in rows:
    conv = row.get("conversation", [])
    question = conv[0].get("content", "") if conv else ""

    if question_counts[question] >= max_per_question:
      continue

    question_counts[question] += 1
    deduped.append(row)

  return deduped, len(rows) - len(deduped)


def fuzzy_dedup(rows: list[dict], threshold: float = JACCARD_THRESHOLD) -> tuple[list[dict], int]:
  """Fuzzy dedup on context text via MinHash/LSH.

  Groups rows by context, deduplicates at the context level, and keeps all
  rows for each surviving context. This preserves multiple questions about
  the same passage while removing near-duplicate passages across sources.
  """
  from datasketch import MinHash, MinHashLSH

  ctx_to_indices: dict[str, list[int]] = {}

  for i, row in enumerate(rows):
    ctx = row.get("context", "")

    if ctx not in ctx_to_indices:
      ctx_to_indices[ctx] = []

    ctx_to_indices[ctx].append(i)

  unique_contexts = list(ctx_to_indices.keys())

  log.info("  %d unique contexts from %d rows", len(unique_contexts), len(rows))

  num_perm = 128
  lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
  removed_contexts: set[int] = set()

  for ci, ctx in enumerate(unique_contexts):
    words = ctx.lower().split()
    min_hash = MinHash(num_perm=num_perm)
    shingles = {" ".join(words[j : j + 3]) for j in range(max(1, len(words) - 2))}

    for shingle in shingles:
      min_hash.update(shingle.encode("utf-8"))

    if lsh.query(min_hash):
      removed_contexts.add(ci)
    else:
      try:
        lsh.insert(str(ci), min_hash)
      except ValueError:
        removed_contexts.add(ci)

    if (ci + 1) % 10_000 == 0:
      log.info("  fuzzy dedup: %d / %d contexts, removed %d", ci + 1, len(unique_contexts), len(removed_contexts))

  surviving = set()

  for ci, ctx in enumerate(unique_contexts):
    if ci not in removed_contexts:
      surviving.update(ctx_to_indices[ctx])

  removed_count = len(rows) - len(surviving)
  deduped = [rows[i] for i in sorted(surviving)]

  return deduped, removed_count


VALIDATION_PROMPT = """\
Is this a valid reading comprehension training example?
Check: (1) Is the answer actually supported by the context?
(2) Is the answer complete and well-phrased?

Context:
{context}

Conversation:
{conversation}

Respond with ONLY a JSON object (no other text):
{{"valid": true, "reason": "..."}} or {{"valid": false, "reason": "..."}}"""


async def _validate_one(row: dict, provider, cost, sem) -> tuple[str, bool | None]:
  async with sem:
    from data.scripts.generate_synthetic import extract_json

    conversation = "\n".join(f"{t['role'].upper()}: {t['content']}" for t in row.get("conversation", []))
    prompt = VALIDATION_PROMPT.format(
      context=row.get("context", "")[:1500],
      conversation=conversation[:1000],
    )

    try:
      text, model, in_tok, out_tok, cache_read, cache_write = await provider.generate(prompt)
      cost.record(model, in_tok, out_tok, cache_read, cache_write)
      parsed = extract_json(text)

      if isinstance(parsed, dict) and "valid" in parsed:
        return row.get("source", "unknown"), bool(parsed["valid"])
    except Exception:
      pass

    return row.get("source", "unknown"), None


async def _run_llm_validation(
  rows: list[dict],
  sample_pct: float,
  provider_name: str,
  model_args: list[str] | None,
  oss_backend: str | None,
  concurrency: int,
) -> dict:
  from tqdm import tqdm

  from data.scripts.generate_synthetic import CostTracker, _load_dotenv, _parse_model_args, make_provider

  _load_dotenv()

  rng = random.Random(RANDOM_SEED)
  n = max(1, int(len(rows) * sample_pct))
  sample = rng.sample(rows, min(n, len(rows)))
  log.info("LLM validation: %d / %d rows (%.1f%%)", len(sample), len(rows), sample_pct * 100)

  model, models = _parse_model_args(model_args)
  provider = make_provider(provider_name, model=model, models=models, oss_backend=oss_backend)
  cost = CostTracker()
  sem = asyncio.Semaphore(concurrency)
  tasks = [asyncio.create_task(_validate_one(r, provider, cost, sem)) for r in sample]
  results: list[tuple[str, bool | None]] = []

  class _TqdmHandler(logging.Handler):
    """Routes log records through tqdm.write() so they don't break the bar."""

    def __init__(self, formatter: logging.Formatter | None = None):
      super().__init__()

      if formatter:
        self.setFormatter(formatter)

    def emit(self, record: logging.LogRecord) -> None:
      try:
        tqdm.write(self.format(record), file=sys.stderr)
      except Exception:
        self.handleError(record)

  root = logging.getLogger()

  original_handlers = root.handlers[:]

  fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

  tqdm_handler = _TqdmHandler(fmt)

  tqdm_handler.setLevel(logging.WARNING)

  root.handlers = [tqdm_handler]

  try:
    with tqdm(total=len(sample), desc="LLM validation", unit="row", file=sys.stderr) as pbar:
      for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)
        pbar.update(1)
  finally:
    root.handlers = original_handlers

  by_source: dict[str, dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0, "error": 0})

  for source, valid in results:
    if valid is None:
      by_source[source]["error"] += 1
    elif valid:
      by_source[source]["valid"] += 1
    else:
      by_source[source]["invalid"] += 1

  report: dict = {"sample_size": len(sample), "per_source": {}, "all_pass": True}

  for source, c in by_source.items():
    total = c["valid"] + c["invalid"]
    rate = c["valid"] / total if total else 0
    flagged = rate < 0.85

    if flagged:
      report["all_pass"] = False

    report["per_source"][source] = {
      "sampled": total + c["error"],
      "valid": c["valid"],
      "invalid": c["invalid"],
      "errors": c["error"],
      "pass_rate": round(rate, 4),
      "flagged": flagged,
    }

  log.info("LLM validation done | %s", cost.summary().split("\n")[0])

  return report


def step_filter(args: argparse.Namespace) -> None:
  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

  log.info("Loading all data sources...")

  rows = load_all_sources()

  log.info("Total: %d rows", len(rows))

  log.info("Stripping [source:] citations from context answers...")

  rows, n_stripped = strip_source_citations(rows)

  log.info("Stripped citations from %d assistant turns", n_stripped)

  log.info("Running automated quality checks...")

  passing, report = run_automated_checks(rows)

  log.info("Passed checks: %d / %d", len(passing), len(rows))

  log.info("Exact dedup on (context, first_question)...")

  passing, exact_n = exact_dedup(passing)
  report["exact_dedup_removed"] = exact_n

  log.info("Exact dedup removed %d (%d remaining)", exact_n, len(passing))

  log.info("Question-cap dedup (max %d per question)...", MAX_QUESTION_DUPLICATES)

  passing, question_cap_n = question_cap_dedup(passing, MAX_QUESTION_DUPLICATES)

  report["question_cap_dedup_removed"] = question_cap_n

  log.info("Question-cap dedup removed %d (%d remaining)", question_cap_n, len(passing))

  log.info("Fuzzy dedup (MinHash/LSH, threshold=%.2f)...", JACCARD_THRESHOLD)

  passing, fuzzy_n = fuzzy_dedup(passing, JACCARD_THRESHOLD)
  report["fuzzy_dedup_removed"] = fuzzy_n

  log.info("Fuzzy dedup removed %d (%d remaining)", fuzzy_n, len(passing))

  report["after_filtering"] = len(passing)
  report["per_source_after"] = dict(Counter(r.get("source", "unknown") for r in passing))

  if args.llm_validate:
    report["llm_validation"] = asyncio.run(
      _run_llm_validation(
        passing,
        args.sample_pct,
        args.llm_provider,
        args.llm_model,
        args.llm_oss_backend,
        args.concurrency,
      )
    )
  else:
    log.info("Skipping LLM validation (use --llm-validate to enable)")
    report["llm_validation"] = "skipped"

  log.info("Writing %d rows → %s", len(passing), FILTERED_PATH)

  with open(FILTERED_PATH, "w", encoding="utf-8") as filtered_file:
    for row in passing:
      filtered_file.write(json.dumps(row, ensure_ascii=False) + "\n")

  with open(FILTER_REPORT_PATH, "w", encoding="utf-8") as report_file:
    json.dump(report, report_file, indent=2, ensure_ascii=False)

  _print_filter_report(report)


def _print_filter_report(report: dict) -> None:
  def _print(s: str) -> None:
    print(s, file=sys.stderr)

  _print("\n" + "=" * 60)
  _print("FILTER REPORT")
  _print("=" * 60)
  _print(f"  Total input:         {report['total_input']:,}")
  _print(f"  Passed checks:       {report['passed_checks']:,}")
  _print(f"  Failed checks:       {report['failed_checks']:,}")
  _print(f"  Exact dedup removed: {report['exact_dedup_removed']:,}")
  _print(f"  Question-cap dedup:  {report['question_cap_dedup_removed']:,}")
  _print(f"  Fuzzy dedup removed: {report['fuzzy_dedup_removed']:,}")
  _print(f"  After filtering:     {report['after_filtering']:,}")
  _print("\n  Per-source after filtering:")

  for src, n in sorted(report["per_source_after"].items(), key=lambda x: -x[1]):
    total = report["per_source_input"].get(src, 0)
    pct = n / total * 100 if total else 0
    _print(f"    {src}: {n:,} / {total:,} ({pct:.1f}%)")

  if isinstance(report.get("llm_validation"), dict):
    _print("\n  LLM validation:")

    for src, v in report["llm_validation"]["per_source"].items():
      flag = " FLAGGED" if v["flagged"] else ""
      _print(f"    {src}: {v['pass_rate']:.1%} ({v['valid']}/{v['sampled']}){flag}")

  _print("\n  Top failure reasons:")

  for reason, n in list(report.get("failure_reasons", {}).items())[:10]:
    _print(f"    {reason}: {n:,}")

  _print("=" * 60)


def step_tokenizer(args: argparse.Namespace) -> None:
  from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

  if not FILTERED_PATH.exists():
    log.error("Run --step filter first: %s not found", FILTERED_PATH)
    sys.exit(1)

  log.info("Collecting texts for tokenizer training...")

  texts: list[str] = []
  passages: list[str] = []

  with open(FILTERED_PATH, encoding="utf-8") as filtered_file:
    for line in filtered_file:
      line = line.strip()

      if not line:
        continue

      row = json.loads(line)
      ctx = row["context"]
      texts.append(ctx)
      passages.append(ctx)

      for turn in row.get("conversation", []):
        texts.append(turn["content"])

  log.info("Collected %d text segments (%d unique passages)", len(texts), len(set(passages)))

  log.info("Training BPE tokenizer (vocab=%d)...", VOCAB_SIZE)

  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  tokenizer.decoder = decoders.ByteLevel()

  trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    min_frequency=2,
  )

  tokenizer.train_from_iterator(texts, trainer=trainer, length=len(texts))

  rng = random.Random(RANDOM_SEED)

  unique_passages = list(set(passages))
  rt_sample = rng.sample(unique_passages, min(1000, len(unique_passages)))
  cov_sample = rng.sample(texts, min(1000, len(texts)))

  rt_fail = 0

  for t in rt_sample:
    enc = tokenizer.encode(t)
    dec = tokenizer.decode(enc.ids)

    if dec != t:
      rt_fail += 1

  total_tok = 0
  unk_tok = 0
  total_words = 0
  unk_id = tokenizer.token_to_id("[UNK]")

  for t in cov_sample:
    enc = tokenizer.encode(t)
    total_tok += len(enc.ids)
    total_words += _count_words(t)

    if unk_id is not None:
      unk_tok += sum(1 for tok_id in enc.ids if tok_id == unk_id)

  unk_rate = unk_tok / total_tok if total_tok else 0
  tpw = total_tok / total_words if total_words else 0

  for i, tok in enumerate(SPECIAL_TOKENS):
    tid = tokenizer.token_to_id(tok)

    if tid != i:
      log.error("Special token %s: got ID %s, expected %d", tok, tid, i)
      sys.exit(1)

  TOKENIZER_PATH.parent.mkdir(parents=True, exist_ok=True)

  tokenizer.save(str(TOKENIZER_PATH))

  def _print(s: str) -> None:
    print(s, file=sys.stderr)

  _print("\n" + "=" * 60)
  _print("TOKENIZER REPORT")
  _print("=" * 60)
  _print(f"  Vocab size:        {tokenizer.get_vocab_size()}")
  _print(f"  Special tokens:    {len(SPECIAL_TOKENS)} (IDs 0-{len(SPECIAL_TOKENS) - 1})")
  _print(f"  Round-trip:        {len(rt_sample) - rt_fail} / {len(rt_sample)} pass")
  _print(f"  Unknown rate:      {unk_rate:.4%}")
  _print(f"  Tokens/word:       {tpw:.2f} (target 1.2-1.5)")
  _print(f"  Saved to:          {TOKENIZER_PATH}")
  _print("=" * 60)

  if rt_fail > 0:
    log.warning("%d / %d round-trip failures", rt_fail, len(rt_sample))

  if unk_rate >= 0.01:
    log.warning("Unknown token rate %.2f%% exceeds 1%% threshold", unk_rate * 100)


def _pack_example(row: dict, tokenizer, sid: dict[str, int]) -> tuple[list[int], list[int], int] | None:
  """Pack one example into token IDs + loss mask. Returns None if >MAX_SEQ_LEN."""
  toks: list[int] = [sid["context"]]
  toks.extend(tokenizer.encode(row["context"]).ids)

  for turn in row["conversation"]:
    role_id = sid["user"] if turn["role"] == "user" else sid["assistant"]
    toks.append(role_id)
    toks.extend(tokenizer.encode(turn["content"]).ids)

  toks.append(sid["eos"])

  if len(toks) > MAX_SEQ_LEN:
    return None

  mask = [0] * len(toks)

  in_asst = False

  for i, t in enumerate(toks):
    if t == sid["assistant"]:
      in_asst = True
    elif t in (sid["user"], sid["context"]):
      in_asst = False
    elif t == sid["eos"] or in_asst:
      mask[i] = 1

  actual_len = len(toks)
  pad_n = MAX_SEQ_LEN - actual_len

  toks.extend([sid["pad"]] * pad_n)
  mask.extend([0] * pad_n)

  return toks, mask, actual_len


def _parse_caps(raw: list[str] | None) -> dict[str, int]:
  """Parse --cap SOURCE=N values into a dict."""
  if not raw:
    return {}

  caps: dict[str, int] = {}

  for entry in raw:
    if "=" not in entry:
      raise ValueError(f"Invalid --cap format: {entry!r} (expected SOURCE=N)")

    source, _, count = entry.partition("=")
    caps[source.strip()] = int(count.strip())

  return caps


def _apply_caps(rows: list[dict], caps: dict[str, int], seed: int = RANDOM_SEED) -> list[dict]:
  """Deterministically sample rows per source to enforce caps."""
  if not caps:
    return rows

  by_source: dict[str, list[dict]] = {}

  for row in rows:
    by_source.setdefault(row.get("source", "unknown"), []).append(row)

  result: list[dict] = []
  rng = random.Random(seed)

  for source, source_rows in by_source.items():
    cap = caps.get(source)

    if cap is not None and len(source_rows) > cap:
      log.info("  Capping %s: %d → %d", source, len(source_rows), cap)
      sampled = rng.sample(source_rows, cap)
      result.extend(sampled)
    else:
      result.extend(source_rows)

  return result


def step_process(args: argparse.Namespace) -> None:
  from tokenizers import Tokenizer

  for path, label in [(FILTERED_PATH, "--step filter"), (TOKENIZER_PATH, "--step tokenizer")]:
    if not path.exists():
      log.error("Run %s first: %s not found", label, path)
      sys.exit(1)

  tokenizer = Tokenizer.from_file(str(TOKENIZER_PATH))

  sid: dict[str, int] = {}

  for tok in SPECIAL_TOKENS:
    tid = tokenizer.token_to_id(tok)

    if tid is None:
      log.error("Special token %s missing from tokenizer", tok)
      sys.exit(1)

    sid[tok.strip("<|>")] = tid

  caps = _parse_caps(args.cap)

  if caps:
    log.info("Per-source caps: %s", caps)

  log.info("Loading rows from %s...", FILTERED_PATH)

  all_rows: list[dict] = []

  with open(FILTERED_PATH, encoding="utf-8") as f:
    for line in f:
      line = line.strip()

      if line:
        all_rows.append(json.loads(line))

  log.info("Loaded %d rows", len(all_rows))

  if caps:
    all_rows = _apply_caps(all_rows, caps)
    log.info("After caps: %d rows", len(all_rows))

  log.info("Packing and tokenizing (max_seq_len=%d)...", MAX_SEQ_LEN)

  all_toks: list[list[int]] = []
  all_mask: list[list[int]] = []
  lengths: list[int] = []
  source_counts: Counter[str] = Counter()
  discarded = 0

  for n, row in enumerate(all_rows, 1):
    result = _pack_example(row, tokenizer, sid)

    if result is None:
      discarded += 1
      continue

    t, m, alen = result
    all_toks.append(t)
    all_mask.append(m)
    lengths.append(alen)
    source_counts[row.get("source", "unknown")] += 1

    if n % 50_000 == 0:
      log.info("  %d rows...", n)

  total = len(all_toks)

  log.info("Tokenized %d examples, discarded %d (>%d tokens)", total, discarded, MAX_SEQ_LEN)

  # deterministic shuffle
  rng = random.Random(RANDOM_SEED)
  idx = list(range(total))

  rng.shuffle(idx)

  all_toks = [all_toks[i] for i in idx]
  all_mask = [all_mask[i] for i in idx]
  lengths = [lengths[i] for i in idx]

  # 90 / 5 / 5 split
  n_val = int(total * 0.05)
  n_test = int(total * 0.05)
  n_train = total - n_val - n_test

  PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

  splits = [
    ("train", 0, n_train),
    ("val", n_train, n_train + n_val),
    ("test", n_train + n_val, total),
  ]

  for name, lo, hi in splits:
    path = PROCESSED_DIR / f"{name}.npz"

    np.savez(
      str(path),
      input_ids=np.array(all_toks[lo:hi], dtype=np.int32),
      loss_mask=np.array(all_mask[lo:hi], dtype=np.int32),
    )

    mb = path.stat().st_size / 1e6

    log.info("Saved %s: %d examples → %s (%.1f MB)", name, hi - lo, path, mb)

  avg_len = sum(lengths) / len(lengths) if lengths else 0

  stats = {
    "total_filtered": total + discarded,
    "after_truncation": total,
    "discarded_too_long": discarded,
    "train": n_train,
    "val": n_val,
    "test": n_test,
    "avg_sequence_length": round(avg_len, 1),
    "max_sequence_length": MAX_SEQ_LEN,
    "vocab_size": tokenizer.get_vocab_size(),
    "source_breakdown": dict(source_counts),
  }

  with open(STATS_PATH, "w", encoding="utf-8") as stats_file:
    json.dump(stats, stats_file, indent=2, ensure_ascii=False)

  def _print(s: str) -> None:
    print(s, file=sys.stderr)

  _print("\n" + "=" * 60)
  _print("Dataset Statistics:")
  _print(f"  Total examples:       {total + discarded:,}")
  _print(f"  After truncation:     {total:,}")
  _print(f"  Discarded (too long): {discarded:,}")
  _print(f"  Train: {n_train:,} | Val: {n_val:,} | Test: {n_test:,}")
  _print(f"  Avg sequence length:  {avg_len:.0f} tokens")
  _print("  Source breakdown:")

  for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    label = f"{cnt / 1000:.0f}K" if cnt >= 1000 else str(cnt)
    _print(f"    {src}: {label}")

  _print("=" * 60)


def main() -> None:
  from data.scripts.generate_synthetic import OSS_BACKENDS, PROVIDER_CHOICES

  parser = argparse.ArgumentParser(
    description="Merge, filter, tokenize, and split all training data",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""\
Steps:
  filter     Load all sources → quality checks → dedup → filtered.jsonl
  tokenizer  Train production BPE on filtered corpus → model/tokenizer.json
  process    Tokenize + loss masks + shuffle + split → {train,val,test}.npz
  all        Run filter → tokenizer → process sequentially
""",
  )

  parser.add_argument("--step", required=True, choices=["filter", "tokenizer", "process", "all"])
  parser.add_argument(
    "--cap",
    action="append",
    default=None,
    help="Per-source row cap: SOURCE=N (repeatable, e.g. --cap squad2=45000 --cap drop=40000). "
    "Applied during the process step before tokenization. Rows are sampled deterministically.",
  )
  parser.add_argument("--llm-validate", action="store_true", help="Run LLM validation on a sample (costs $)")
  parser.add_argument(
    "--llm-provider",
    default="anthropic",
    choices=PROVIDER_CHOICES,
    help="Provider for LLM validation (supports anthropic/openai/minimax/kimi/glm/round-robin)",
  )
  parser.add_argument(
    "--llm-model",
    action="append",
    default=None,
    help="Model override(s) for LLM validation. Plain value for single providers "
    "(e.g. gpt-4o-mini). For round-robin use PROVIDER=MODEL (repeatable).",
  )
  parser.add_argument(
    "--llm-oss-backend",
    default=None,
    choices=list(OSS_BACKENDS),
    help="OSS backend for minimax/kimi/glm validation providers: baseten or together (auto-detect by default)",
  )
  parser.add_argument("--sample-pct", type=float, default=0.05, help="LLM validation sample fraction (default: 0.05)")
  parser.add_argument("--concurrency", type=int, default=10)

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
  logging.getLogger("httpx").setLevel(logging.WARNING)
  logging.getLogger("httpcore").setLevel(logging.WARNING)

  steps = ["filter", "tokenizer", "process"] if args.step == "all" else [args.step]

  for step in steps:
    log.info("=== Step: %s ===", step)
    t0 = time.monotonic()
    {"filter": step_filter, "tokenizer": step_tokenizer, "process": step_process}[step](args)
    log.info("Step '%s' completed in %.1fs\n", step, time.monotonic() - t0)


if __name__ == "__main__":
  main()
