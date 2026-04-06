"""Evaluation suite for PicoRouter grounded QA mode.

Metrics:
  1. Extractive F1 / Exact Match (answerable examples)
  2. Refusal rate (unanswerable examples)
  3. Hallucination probe (adversarial context/question mismatch)
  4. Latency benchmark (tokens/sec at various context lengths)
"""

import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import torch
from tqdm import tqdm

from model.architecture import PicoConfig, PicoRouter
from model.tokenizer import encode_example, load_tokenizer

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FILTERED_PATH = DATA_DIR / "filtered.jsonl"
EVAL_REPORT_PATH = Path(__file__).parent / "eval_report.json"
CACHED_TEST_PATH = DATA_DIR / "cached_test_split.json"

REFUSAL_PHRASE = "I don't have enough information"
DEFAULT_BATCH_SIZE = 32


def get_device(override: str | None = None) -> torch.device:
  if override:
    return torch.device(override)

  if torch.cuda.is_available():
    return torch.device("cuda")

  if torch.backends.mps.is_available():
    return torch.device("mps")

  return torch.device("cpu")


def load_model(
  checkpoint_dir: Path,
  device: torch.device,
  compile_model: bool = True,
) -> tuple[PicoRouter, PicoConfig]:
  with open(checkpoint_dir / "config.json") as config_file:
    config = PicoConfig(**json.load(config_file))

  model = PicoRouter(config).to(device)

  weights_path = checkpoint_dir / "weights.pt"
  state_dict = torch.load(weights_path, map_location=device, weights_only=True)
  model.load_state_dict(state_dict)
  model.eval()

  if compile_model:
    print("  Compiling model with torch.compile...")
    model = torch.compile(model)

  return model, config


@torch.no_grad()
def generate(
  model: PicoRouter,
  token_ids: list[int],
  eos_id: int,
  device: torch.device,
  max_new_tokens: int = 256,
) -> list[int]:
  """Greedy autoregressive generation with KV-cache (single example)."""
  generated = []
  cache = None

  input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

  for _ in range(max_new_tokens):
    logits, cache = model(input_ids, cache)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)

    tid = next_token.item()

    if tid == eos_id:
      break

    generated.append(tid)

    input_ids = torch.tensor([[tid]], dtype=torch.long, device=device)

  return generated


@torch.no_grad()
def generate_batch(
  model: PicoRouter,
  token_ids_list: list[list[int]],
  eos_id: int,
  device: torch.device,
  max_new_tokens: int = 256,
  batch_size: int = DEFAULT_BATCH_SIZE,
  desc: str = "Generating",
) -> list[list[int]]:
  """Batched greedy autoregressive generation with KV-cache.

  Left-pads variable-length prompts, builds per-example position_ids for
  correct RoPE encoding, and uses explicit attention masks to ignore padding.
  """
  if not token_ids_list:
    return []

  max_seq_len = model.config.max_seq_len
  indices: list[int] = []
  filtered: list[list[int]] = []

  for i, toks in enumerate(token_ids_list):
    if len(toks) < max_seq_len:
      indices.append(i)
      filtered.append(toks)

  n_batches = (len(filtered) + batch_size - 1) // batch_size
  gen_results: list[list[int]] = []

  for batch_start in tqdm(range(0, len(filtered), batch_size), total=n_batches, desc=desc, unit="batch"):
    batch_tokens = filtered[batch_start : batch_start + batch_size]
    gen = _generate_one_batch(model, batch_tokens, eos_id, device, max_new_tokens)
    gen_results.extend(gen)

  result: list[list[int]] = [[] for _ in token_ids_list]

  for idx, gen in zip(indices, gen_results, strict=True):
    result[idx] = gen

  return result


def _generate_one_batch(
  model: PicoRouter,
  batch_tokens: list[list[int]],
  eos_id: int,
  device: torch.device,
  max_new_tokens: int,
) -> list[list[int]]:
  B = len(batch_tokens)

  if B == 0:
    return []

  max_seq_len = model.config.max_seq_len
  prompt_lengths = [len(tokens) for tokens in batch_tokens]
  max_prompt_length = max(prompt_lengths)
  max_new_tokens = min(max_new_tokens, max_seq_len - max_prompt_length)

  if max_new_tokens <= 0:
    return [[] for _ in range(B)]

  padding_lengths = [max_prompt_length - prompt_length for prompt_length in prompt_lengths]

  padded = torch.zeros(B, max_prompt_length, dtype=torch.long, device=device)

  for i, tokens in enumerate(batch_tokens):
    padded[i, padding_lengths[i] :] = torch.tensor(tokens, dtype=torch.long, device=device)

  position_ids = torch.zeros(B, max_prompt_length, dtype=torch.long, device=device)

  for i in range(B):
    position_ids[i, padding_lengths[i] :] = torch.arange(prompt_lengths[i], device=device)

  causal = torch.tril(torch.ones(max_prompt_length, max_prompt_length, dtype=torch.bool, device=device))

  key_mask = torch.zeros(B, max_prompt_length, dtype=torch.bool, device=device)

  for i in range(B):
    key_mask[i, padding_lengths[i] :] = True

  attn_mask = causal.unsqueeze(0).unsqueeze(0) & key_mask.unsqueeze(1).unsqueeze(2)

  logits, cache = model(padded, None, attn_mask=attn_mask, position_ids=position_ids)
  next_tokens = torch.argmax(logits[:, -1, :], dim=-1)

  finished = next_tokens == eos_id
  generated: list[list[int]] = [[] for _ in range(B)]

  for i in range(B):
    if not finished[i]:
      generated[i].append(next_tokens[i].item())

  max_total_kv = max_prompt_length + max_new_tokens

  decode_mask_full = torch.ones(B, 1, 1, max_total_kv, dtype=torch.bool, device=device)

  for i in range(B):
    decode_mask_full[i, 0, 0, : padding_lengths[i]] = False

  current_pos = torch.tensor(prompt_lengths, dtype=torch.long, device=device)

  for step in range(1, max_new_tokens):
    if finished.all():
      break

    input_ids = next_tokens.unsqueeze(1)
    kv_len = max_prompt_length + step
    decode_mask = decode_mask_full[:, :, :, :kv_len]
    decode_pos = current_pos.unsqueeze(1)

    logits, cache = model(input_ids, cache, attn_mask=decode_mask, position_ids=decode_pos)
    next_tokens = torch.argmax(logits[:, -1, :], dim=-1)
    current_pos += 1

    for i in range(B):
      if not finished[i]:
        if next_tokens[i].item() == eos_id:
          finished[i] = True
        else:
          generated[i].append(next_tokens[i].item())

  return generated


def load_test_examples() -> dict[str, list[dict]]:
  """Load filtered.jsonl and split into categories for evaluation.

  Mirrors the training pipeline: tokenize each example, discard those
  exceeding max_seq_len, shuffle with the same seed, then apply the
  same 90/5/5 split. Returns only the test portion grouped by type.

  Caches the result to disk so subsequent runs skip the tokenization pass.
  """
  if (
    CACHED_TEST_PATH.exists()
    and FILTERED_PATH.exists()
    and CACHED_TEST_PATH.stat().st_mtime > FILTERED_PATH.stat().st_mtime
  ):
    print("  (loading cached test split)")

    with open(CACHED_TEST_PATH, encoding="utf-8") as cached_test_file:
      return json.load(cached_test_file)

  tokenizer = load_tokenizer()

  kept = []

  with open(FILTERED_PATH, encoding="utf-8") as filtered_file:
    for line in filtered_file:
      line = line.strip()

      if not line:
        continue

      ex = json.loads(line)

      tokens = encode_example(tokenizer, ex)

      if len(tokens) <= 1024:
        kept.append(ex)

  rng = random.Random(42)
  indices = list(range(len(kept)))
  rng.shuffle(indices)
  kept = [kept[i] for i in indices]

  n = len(kept)
  n_val = int(n * 0.05)
  n_test = int(n * 0.05)
  n_train = n - n_val - n_test
  test_set = kept[n_train + n_val :]

  groups: dict[str, list[dict]] = {"rc": [], "refusal": []}

  for example in test_set:
    is_refusal = any(
      REFUSAL_PHRASE in turn["content"] for turn in example["conversation"] if turn["role"] == "assistant"
    )

    if is_refusal:
      groups["refusal"].append(example)
    else:
      groups["rc"].append(example)

  with open(CACHED_TEST_PATH, "w", encoding="utf-8") as cached_test_file:
    json.dump(groups, cached_test_file)

  print(f"  (cached test split to {CACHED_TEST_PATH})")

  return groups


def build_prompt_tokens(tokenizer, example: dict) -> list[int]:
  """Build the prompt token sequence up to (and including) the final <|assistant|> token.

  For a 2-turn example the prompt is everything before the assistant's answer.
  For multi-turn, we include all but the last assistant turn and append <|assistant|>.
  """
  from model.tokenizer import get_special_token_id

  ctx_id = get_special_token_id(tokenizer, "<|context|>")
  user_id = get_special_token_id(tokenizer, "<|user|>")
  asst_id = get_special_token_id(tokenizer, "<|assistant|>")

  tokens = [ctx_id]
  tokens.extend(tokenizer.encode(example["context"]).ids)

  conversation = example["conversation"]

  for i, turn in enumerate(conversation):
    if turn["role"] == "user":
      tokens.append(user_id)
      tokens.extend(tokenizer.encode(turn["content"]).ids)
    elif turn["role"] == "assistant":
      if i == len(conversation) - 1:
        tokens.append(asst_id)
      else:
        tokens.append(asst_id)
        tokens.extend(tokenizer.encode(turn["content"]).ids)

  return tokens


def get_gold_answer(example: dict) -> str:
  """Return the last assistant turn's content."""
  for turn in reversed(example["conversation"]):
    if turn["role"] == "assistant":
      return turn["content"]
  return ""


def _normalize(text: str) -> str:
  text = text.lower()
  text = re.sub(r"\[source:[^\]]*\]", "", text)
  text = re.sub(r"<\|[a-z_]+\|>", "", text)
  text = "".join(ch for ch in text if ch not in string.punctuation)
  text = re.sub(r"\b(a|an|the)\b", " ", text)
  return " ".join(text.split())


def token_f1(prediction: str, gold: str) -> tuple[float, float]:
  pred_tokens = _normalize(prediction).split()
  gold_tokens = _normalize(gold).split()

  if not gold_tokens:
    return (1.0, 1.0) if not pred_tokens else (0.0, 0.0)

  if not pred_tokens:
    return 0.0, 0.0

  common = set(pred_tokens) & set(gold_tokens)

  num_common = sum(min(pred_tokens.count(w), gold_tokens.count(w)) for w in common)

  if num_common == 0:
    return 0.0, 0.0

  precision = num_common / len(pred_tokens)
  recall = num_common / len(gold_tokens)
  f1 = 2 * precision * recall / (precision + recall)
  em = 1.0 if _normalize(prediction) == _normalize(gold) else 0.0

  return f1, em


def eval_extractive(
  model: PicoRouter,
  tokenizer,
  examples: list[dict],
  device: torch.device,
  max_examples: int = 2000,
  batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  subset = examples[:max_examples]

  prompts = [build_prompt_tokens(tokenizer, example) for example in subset]
  all_gen_ids = generate_batch(model, prompts, eos_id, device, batch_size=batch_size, desc="[1/4] Extractive")

  f1_scores, exact_match_scores = [], []

  for gen_ids, example in zip(all_gen_ids, subset, strict=True):
    prediction = tokenizer.decode(gen_ids)
    gold = get_gold_answer(example)
    f1_score, exact_match = token_f1(prediction, gold)
    f1_scores.append(f1_score)
    exact_match_scores.append(exact_match)

  n = len(f1_scores)

  return {
    "mean_f1": round(sum(f1_scores) / max(n, 1), 4),
    "mean_em": round(sum(exact_match_scores) / max(n, 1), 4),
    "median_f1": round(sorted(f1_scores)[n // 2] if f1_scores else 0.0, 4),
    "n": n,
  }


def eval_refusal(
  model: PicoRouter,
  tokenizer,
  refusal_examples: list[dict],
  device: torch.device,
  max_examples: int = 2000,
  batch_size: int = DEFAULT_BATCH_SIZE,
  *,
  rc_gen_ids: list[list[int]],
  rc_predictions: list[str],
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  refuse_id = tokenizer.token_to_id("<|refuse|>")

  refusal_subset = refusal_examples[:max_examples]

  prompts = [build_prompt_tokens(tokenizer, ex) for ex in refusal_subset]
  all_gen_ids = generate_batch(model, prompts, eos_id, device, batch_size=batch_size, desc="[2/4] Refusal")

  correct_refusals = 0

  for gen_ids in all_gen_ids:
    prediction = tokenizer.decode(gen_ids)

    if REFUSAL_PHRASE.lower() in prediction.lower() or refuse_id in gen_ids:
      correct_refusals += 1

  false_refusals = 0

  for gen_ids, prediction in zip(rc_gen_ids, rc_predictions, strict=True):
    if REFUSAL_PHRASE.lower() in prediction.lower() or refuse_id in gen_ids:
      false_refusals += 1

  n_ref = len(refusal_subset)
  n_rc = len(rc_gen_ids)

  return {
    "correct_refusal_rate": round(correct_refusals / max(n_ref, 1), 4),
    "false_refusal_rate": round(false_refusals / max(n_rc, 1), 4),
    "n_refusal": n_ref,
    "n_rc_for_false_rate": n_rc,
  }


def build_adversarial_examples(
  examples: list[dict],
  n: int = 500,
  seed: int = 123,
) -> list[dict]:
  """Pair context from one example with a question from a different example.

  The model should refuse (the question is unanswerable from the given context).
  """
  rng = random.Random(seed)

  pool = [
    example
    for example in examples
    if not any(REFUSAL_PHRASE in t["content"] for t in example["conversation"] if t["role"] == "assistant")
  ]

  if len(pool) < 2:
    return []

  adversarial = []
  indices = list(range(len(pool)))

  for _ in range(n):
    i, j = rng.sample(indices, 2)

    adv = {
      "context": pool[i]["context"],
      "tools": [],
      "conversation": [
        pool[j]["conversation"][0],
        {"role": "assistant", "content": ""},
      ],
    }

    adversarial.append(adv)

  return adversarial


def eval_hallucination(
  model: PicoRouter,
  tokenizer,
  all_examples: list[dict],
  device: torch.device,
  n_probes: int = 500,
  batch_size: int = DEFAULT_BATCH_SIZE,
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  refuse_id = tokenizer.token_to_id("<|refuse|>")

  adversarial = build_adversarial_examples(all_examples, n=n_probes)

  prompts = [build_prompt_tokens(tokenizer, ex) for ex in adversarial]
  all_gen_ids = generate_batch(model, prompts, eos_id, device, batch_size=batch_size, desc="[3/4] Hallucination")

  hallucinations = 0

  for gen_ids in all_gen_ids:
    prediction = tokenizer.decode(gen_ids)
    is_refusal = REFUSAL_PHRASE.lower() in prediction.lower() or refuse_id in gen_ids

    if not is_refusal:
      hallucinations += 1

  n = len(adversarial)

  return {
    "hallucination_rate": round(hallucinations / max(n, 1), 4),
    "n": n,
  }


def _sync_device(device: torch.device):
  """Ensure all pending GPU/MPS work completes for accurate timing."""
  if device.type == "cuda":
    torch.cuda.synchronize()


@torch.no_grad()
def eval_latency(
  model: PicoRouter,
  tokenizer,
  device: torch.device,
  context_lengths: list[int] | None = None,
  gen_tokens: int = 64,
  n_trials: int = 5,
) -> dict:
  if context_lengths is None:
    context_lengths = [256, 512, 1024]

  eos_id = tokenizer.token_to_id("<|eos|>")
  results = {}

  for ctx_len in context_lengths:
    prompt = list(range(2, min(ctx_len, model.config.max_seq_len - gen_tokens)))

    if len(prompt) < 10:
      prompt = list(range(2, 50))

    ttfts = []
    tok_rates = []

    for _ in range(n_trials):
      cache = None

      input_ids = torch.tensor([prompt], dtype=torch.long, device=device)

      _sync_device(device)

      t0 = time.perf_counter()

      logits, cache = model(input_ids, cache)
      first_token = torch.argmax(logits[:, -1, :], dim=-1)

      _sync_device(device)
      ttft = time.perf_counter() - t0

      ttfts.append(ttft)

      generated = 0
      t_gen_start = time.perf_counter()

      tid = first_token.item()

      for _ in range(gen_tokens - 1):
        if tid == eos_id:
          break

        input_ids = torch.tensor([[tid]], dtype=torch.long, device=device)
        logits, cache = model(input_ids, cache)
        next_token = torch.argmax(logits[:, -1, :], dim=-1)

        tid = next_token.item()
        generated += 1

      _sync_device(device)

      t_gen_end = time.perf_counter()

      gen_elapsed = t_gen_end - t_gen_start

      if generated > 0 and gen_elapsed > 0:
        tok_rates.append(generated / gen_elapsed)

    avg_ttft = sum(ttfts) / len(ttfts)
    avg_tok_s = sum(tok_rates) / len(tok_rates) if tok_rates else 0

    results[f"context_{ctx_len}"] = {
      "ttft_ms": round(avg_ttft * 1000, 1),
      "tok_per_sec": round(avg_tok_s, 1),
    }

  return results


def find_checkpoints() -> list[Path]:
  """Find all checkpoint directories with weights.pt and config.json."""
  if not CHECKPOINT_DIR.exists():
    return []

  checkpoints = []

  for checkpoint_dir in sorted(CHECKPOINT_DIR.iterdir()):
    if (
      checkpoint_dir.is_dir() and (checkpoint_dir / "weights.pt").exists() and (checkpoint_dir / "config.json").exists()
    ):
      checkpoints.append(checkpoint_dir)

  return checkpoints


def composite_score(report: dict) -> float:
  f1 = report.get("extractive", {}).get("mean_f1", 0)
  refusal = report.get("refusal", {}).get("correct_refusal_rate", 0)
  hallucination = report.get("hallucination", {}).get("hallucination_rate", 1)
  false_refusal = report.get("refusal", {}).get("false_refusal_rate", 1)
  return 0.25 * f1 + 0.25 * refusal + 0.35 * (1 - hallucination) + 0.15 * (1 - false_refusal)


def select_best_checkpoint(reports: dict[str, dict]) -> str | None:
  if not reports:
    return None

  return max(reports, key=lambda k: composite_score(reports[k]))


def run_evaluation(
  checkpoint_dir: Path,
  device: torch.device,
  max_rc: int = 2000,
  max_refusal: int = 2000,
  n_hallucination: int = 500,
  skip_latency: bool = False,
  batch_size: int = DEFAULT_BATCH_SIZE,
  compile_model: bool = True,
) -> dict:
  print(f"\nLoading checkpoint: {checkpoint_dir}")

  model, _ = load_model(checkpoint_dir, device, compile_model=compile_model)

  tokenizer = load_tokenizer()

  print("Loading test examples...")

  groups = load_test_examples()

  print(f"  RC: {len(groups['rc'])} | Refusal: {len(groups['refusal'])}")

  all_test = groups["rc"] + groups["refusal"]
  report = {"checkpoint": str(checkpoint_dir.name)}

  eos_id = tokenizer.token_to_id("<|eos|>")
  rc_fp_subset = groups["rc"][:500]

  print(f"\n[0/4] Pre-generating RC false-positive predictions (N={len(rc_fp_subset)})...")

  rc_fp_prompts = [build_prompt_tokens(tokenizer, ex) for ex in rc_fp_subset]
  rc_fp_gen_ids = generate_batch(model, rc_fp_prompts, eos_id, device, batch_size=batch_size, desc="[0/4] RC FP")
  rc_fp_predictions = [tokenizer.decode(ids) for ids in rc_fp_gen_ids]

  print("       done")

  print("\n[1/4] Extractive F1 / Exact Match...")

  report["extractive"] = eval_extractive(model, tokenizer, groups["rc"], device, max_rc, batch_size)

  print(f"       F1={report['extractive']['mean_f1']}  EM={report['extractive']['mean_em']}")

  print("\n[2/4] Refusal rate...")

  report["refusal"] = eval_refusal(
    model,
    tokenizer,
    groups["refusal"],
    device,
    max_refusal,
    batch_size,
    rc_gen_ids=rc_fp_gen_ids,
    rc_predictions=rc_fp_predictions,
  )

  ref = report["refusal"]

  print(f"       Correct={ref['correct_refusal_rate']}  FalseRefusal={ref['false_refusal_rate']}")

  print("\n[3/4] Hallucination probe...")

  report["hallucination"] = eval_hallucination(model, tokenizer, all_test, device, n_hallucination, batch_size)

  print(f"       Rate={report['hallucination']['hallucination_rate']}")

  if not skip_latency:
    print("\n[4/4] Latency benchmark...")

    report["latency"] = eval_latency(model, tokenizer, device)

    for k, v in report["latency"].items():
      print(f"       {k}: TTFT={v['ttft_ms']}ms  {v['tok_per_sec']} tok/s")
  else:
    print("\n[4/4] Latency benchmark... SKIPPED")

  score = composite_score(report)
  report["composite_score"] = round(score, 4)

  print("\n" + "=" * 50)

  print("PicoRouter Evaluation Report")

  print("=" * 50)

  print(f"Checkpoint: {checkpoint_dir.name}")

  print()

  extractive = report["extractive"]

  print("Reading Comprehension:")

  print(f"  F1: {extractive['mean_f1']} | EM: {extractive['mean_em']} | N={extractive['n']}")

  print()

  print("Refusal:")

  print(f"  Correct Refusal Rate: {ref['correct_refusal_rate']} | False Refusal Rate: {ref['false_refusal_rate']}")

  print()

  hallucination = report["hallucination"]

  print("Hallucination Probe:")

  print(f"  Hallucination Rate: {hallucination['hallucination_rate']} (N={hallucination['n']})")

  if "latency" in report:
    print()

    device_label = str(device).upper()

    print(f"Latency ({device_label}):")

    for k, v in report["latency"].items():
      ctx = k.replace("context_", "")
      print(f"  Context {ctx}: TTFT={v['ttft_ms']}ms, {v['tok_per_sec']} tok/s")

  print()

  print(f"Composite Score: {report['composite_score']}")

  print("=" * 50)

  return report


def main():
  parser = argparse.ArgumentParser(description="Evaluate PicoRouter checkpoint(s)")

  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to a specific checkpoint dir. If omitted, evaluates all checkpoints and selects the best.",
  )

  parser.add_argument("--max-rc", type=int, default=2000)
  parser.add_argument("--max-refusal", type=int, default=2000)
  parser.add_argument("--n-hallucination", type=int, default=500)
  parser.add_argument("--skip-latency", action="store_true")
  parser.add_argument("--device", type=str, default=None, help="Force device (cuda, mps, cpu)")
  parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size for generation")
  parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile")
  args = parser.parse_args()

  device = get_device(args.device)
  compile_model = not args.no_compile

  print(f"Using device: {device}")

  if args.checkpoint:
    checkpoint_dir = Path(args.checkpoint)

    report = run_evaluation(
      checkpoint_dir,
      device,
      max_rc=args.max_rc,
      max_refusal=args.max_refusal,
      n_hallucination=args.n_hallucination,
      skip_latency=args.skip_latency,
      batch_size=args.batch_size,
      compile_model=compile_model,
    )

    with open(EVAL_REPORT_PATH, "w") as eval_report_file:
      json.dump(report, eval_report_file, indent=2)

    print(f"\nReport saved to {EVAL_REPORT_PATH}")
  else:
    checkpoints = find_checkpoints()

    if not checkpoints:
      print(f"No checkpoints found in {CHECKPOINT_DIR}")
      return

    print(f"Found {len(checkpoints)} checkpoint(s): {[c.name for c in checkpoints]}")

    to_eval = checkpoints[-10:]

    all_reports: dict[str, dict] = {}

    for checkpoint_dir in to_eval:
      report = run_evaluation(
        checkpoint_dir,
        device,
        max_rc=args.max_rc,
        max_refusal=args.max_refusal,
        n_hallucination=args.n_hallucination,
        skip_latency=args.skip_latency,
        batch_size=args.batch_size,
        compile_model=compile_model,
      )

      all_reports[checkpoint_dir.name] = report

    best_name = select_best_checkpoint(all_reports)

    if best_name:
      print(f"\n*** Best checkpoint: {best_name} (score={all_reports[best_name]['composite_score']}) ***")

      import shutil

      best_src = CHECKPOINT_DIR / best_name
      best_dest = CHECKPOINT_DIR / "best"

      if best_dest.exists():
        shutil.rmtree(best_dest)

      shutil.copytree(best_src, best_dest)

      print(f"Copied to {best_dest}")

      final_report = all_reports[best_name]
      final_report["all_scores"] = {k: round(composite_score(v), 4) for k, v in all_reports.items()}
    else:
      final_report = {}

    with open(EVAL_REPORT_PATH, "w") as eval_report_file:
      json.dump(final_report, eval_report_file, indent=2)

    print(f"\nReport saved to {EVAL_REPORT_PATH}")


if __name__ == "__main__":
  main()
