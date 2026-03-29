"""Evaluation suite for PicoRouter.

Metrics:
  1. Extractive F1 / Exact Match (RC examples)
  2. Tool-call accuracy (tool examples)
  3. Refusal rate (refusal examples)
  4. Hallucination probe (adversarial context/question mismatch)
  5. Latency benchmark (tokens/sec at various context lengths)
"""

import argparse
import json
import random
import re
import string
import time
from pathlib import Path

import torch

from model.architecture import PicoConfig, PicoRouter
from model.tokenizer import encode_example, load_tokenizer

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
FILTERED_PATH = DATA_DIR / "filtered.jsonl"
EVAL_REPORT_PATH = Path(__file__).parent / "eval_report.json"

REFUSAL_PHRASE = "I don't have enough information"


def get_device(override: str | None = None) -> torch.device:
  if override:
    return torch.device(override)
  if torch.cuda.is_available():
    return torch.device("cuda")
  if torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")


def load_model(checkpoint_dir: Path, device: torch.device) -> tuple[PicoRouter, PicoConfig]:
  with open(checkpoint_dir / "config.json") as config_file:
    cfg = PicoConfig(**json.load(config_file))

  model = PicoRouter(cfg).to(device)

  weights_path = checkpoint_dir / "weights.pt"
  state_dict = torch.load(weights_path, map_location=device, weights_only=True)
  model.load_state_dict(state_dict)
  model.eval()

  return model, cfg


@torch.no_grad()
def generate(
  model: PicoRouter,
  token_ids: list[int],
  eos_id: int,
  device: torch.device,
  max_new_tokens: int = 256,
) -> list[int]:
  """Greedy autoregressive generation with KV-cache."""
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


def load_test_examples() -> dict[str, list[dict]]:
  """Load filtered.jsonl and split into categories for evaluation.

  Mirrors the training pipeline: tokenize each example, discard those
  exceeding max_seq_len, shuffle with the same seed, then apply the
  same 90/5/5 split. Returns only the test portion grouped by type.
  """
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

  groups: dict[str, list[dict]] = {"rc": [], "tool": [], "refusal": []}

  for ex in test_set:
    has_tools = bool(ex.get("tools"))
    is_refusal = any(REFUSAL_PHRASE in t["content"] for t in ex["conversation"] if t["role"] == "assistant")

    if is_refusal:
      groups["refusal"].append(ex)
    elif has_tools:
      groups["tool"].append(ex)
    else:
      groups["rc"].append(ex)

  return groups


def build_prompt_tokens(tokenizer, example: dict) -> list[int]:
  """Build the prompt token sequence up to (and including) the final <|assistant|> token.

  For a 2-turn example the prompt is everything before the assistant's answer.
  For multi-turn, we include all but the last assistant turn and append <|assistant|>.
  """
  from model.tokenizer import get_special_token_id

  ctx_id = get_special_token_id(tokenizer, "<|context|>")
  tools_id = get_special_token_id(tokenizer, "<|tools|>")
  user_id = get_special_token_id(tokenizer, "<|user|>")
  asst_id = get_special_token_id(tokenizer, "<|assistant|>")

  tokens = [ctx_id]
  tokens.extend(tokenizer.encode(example["context"]).ids)
  tokens.append(tools_id)

  if example.get("tools"):
    tokens.extend(tokenizer.encode(json.dumps(example["tools"])).ids)

  conv = example["conversation"]

  for i, turn in enumerate(conv):
    if turn["role"] == "user":
      tokens.append(user_id)
      tokens.extend(tokenizer.encode(turn["content"]).ids)
    elif turn["role"] == "assistant":
      if i == len(conv) - 1:
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
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  subset = examples[:max_examples]
  f1s, ems = [], []

  for ex in subset:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)
    prediction = tokenizer.decode(gen_ids)
    gold = get_gold_answer(ex)
    f1, em = token_f1(prediction, gold)
    f1s.append(f1)
    ems.append(em)

  n = len(f1s)

  return {
    "mean_f1": round(sum(f1s) / max(n, 1), 4),
    "mean_em": round(sum(ems) / max(n, 1), 4),
    "median_f1": round(sorted(f1s)[n // 2] if f1s else 0.0, 4),
    "n": n,
  }


def parse_tool_call(text: str) -> dict | None:
  """Try to extract a tool call from generated text.

  Expected format: <|tool_call|>{"name": ..., "arguments": ...}
  The special token may already be stripped by the tokenizer, so we also
  look for a raw JSON object with "name" and "arguments" keys.
  """
  text = text.strip()

  json_match = re.search(r'\{[^{}]*"name"\s*:', text)
  if json_match:
    candidate = text[json_match.start() :]
    brace_depth = 0
    end = 0

    for i, ch in enumerate(candidate):
      if ch == "{":
        brace_depth += 1
      elif ch == "}":
        brace_depth -= 1
        if brace_depth == 0:
          end = i + 1
          break

    if end:
      try:
        parsed = json.loads(candidate[:end])
        if "name" in parsed:
          return parsed
      except json.JSONDecodeError:
        pass

  return None


def parse_gold_tool_call(gold: str) -> dict | None:
  cleaned = re.sub(r"<\|[a-z_]+\|>", "", gold).strip()
  return parse_tool_call(cleaned)


def eval_tool_accuracy(
  model: PicoRouter,
  tokenizer,
  tool_examples: list[dict],
  rc_examples: list[dict],
  device: torch.device,
  max_examples: int = 2000,
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  tool_call_id = tokenizer.token_to_id("<|tool_call|>")
  tool_subset = tool_examples[:max_examples]

  routing_correct = 0
  name_correct = 0
  full_match = 0

  for ex in tool_subset:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)
    prediction = tokenizer.decode(gen_ids)
    gold = get_gold_answer(ex)

    pred_has_tool = tool_call_id in gen_ids or parse_tool_call(prediction) is not None

    if pred_has_tool:
      routing_correct += 1

    pred_call = parse_tool_call(prediction)
    gold_call = parse_gold_tool_call(gold)

    if pred_call and gold_call and pred_call.get("name") == gold_call.get("name"):
      name_correct += 1

      if pred_call.get("arguments") == gold_call.get("arguments"):
        full_match += 1

  n_tool = len(tool_subset)

  false_tool = 0
  rc_subset = rc_examples[:500]

  for ex in rc_subset:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)

    prediction = tokenizer.decode(gen_ids)

    if tool_call_id in gen_ids or parse_tool_call(prediction) is not None:
      false_tool += 1

  n_rc = len(rc_subset)

  return {
    "routing_accuracy": round(routing_correct / max(n_tool, 1), 4),
    "name_accuracy": round(name_correct / max(n_tool, 1), 4),
    "full_match_accuracy": round(full_match / max(n_tool, 1), 4),
    "false_tool_call_rate": round(false_tool / max(n_rc, 1), 4),
    "n_tool": n_tool,
    "n_rc_for_false_rate": n_rc,
  }


def eval_refusal(
  model: PicoRouter,
  tokenizer,
  refusal_examples: list[dict],
  rc_examples: list[dict],
  device: torch.device,
  max_examples: int = 2000,
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  refuse_id = tokenizer.token_to_id("<|refuse|>")

  correct_refusals = 0
  refusal_subset = refusal_examples[:max_examples]

  for ex in refusal_subset:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)

    prediction = tokenizer.decode(gen_ids)

    if REFUSAL_PHRASE.lower() in prediction.lower() or refuse_id in gen_ids:
      correct_refusals += 1

  false_refusals = 0
  rc_subset = rc_examples[:500]

  for ex in rc_subset:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)

    prediction = tokenizer.decode(gen_ids)

    if REFUSAL_PHRASE.lower() in prediction.lower() or refuse_id in gen_ids:
      false_refusals += 1

  n_ref = len(refusal_subset)
  n_rc = len(rc_subset)

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
    ex
    for ex in examples
    if not any(REFUSAL_PHRASE in t["content"] for t in ex["conversation"] if t["role"] == "assistant")
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
) -> dict:
  eos_id = tokenizer.token_to_id("<|eos|>")
  refuse_id = tokenizer.token_to_id("<|refuse|>")

  adversarial = build_adversarial_examples(all_examples, n=n_probes)
  hallucinations = 0

  for ex in adversarial:
    prompt = build_prompt_tokens(tokenizer, ex)
    gen_ids = generate(model, prompt, eos_id, device)
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
  tool_acc = report.get("tool_calling", {}).get("full_match_accuracy", 0)
  refusal = report.get("refusal", {}).get("correct_refusal_rate", 0)
  halluc = report.get("hallucination", {}).get("hallucination_rate", 1)
  return 0.4 * f1 + 0.3 * tool_acc + 0.2 * refusal + 0.1 * (1 - halluc)


def select_best_checkpoint(reports: dict[str, dict]) -> str | None:
  if not reports:
    return None
  return max(reports, key=lambda k: composite_score(reports[k]))


def run_evaluation(
  checkpoint_dir: Path,
  device: torch.device,
  max_rc: int = 2000,
  max_tool: int = 2000,
  max_refusal: int = 2000,
  n_hallucination: int = 500,
  skip_latency: bool = False,
) -> dict:
  print(f"\nLoading checkpoint: {checkpoint_dir}")
  model, _ = load_model(checkpoint_dir, device)

  tokenizer = load_tokenizer()

  print("Loading test examples...")

  groups = load_test_examples()

  print(f"  RC: {len(groups['rc'])} | Tool: {len(groups['tool'])} | Refusal: {len(groups['refusal'])}")

  all_test = groups["rc"] + groups["tool"] + groups["refusal"]
  report = {"checkpoint": str(checkpoint_dir.name)}

  print("\n[1/5] Extractive F1 / Exact Match...")

  report["extractive"] = eval_extractive(model, tokenizer, groups["rc"], device, max_rc)

  print(f"       F1={report['extractive']['mean_f1']}  EM={report['extractive']['mean_em']}")

  print("\n[2/5] Tool-call accuracy...")

  report["tool_calling"] = eval_tool_accuracy(model, tokenizer, groups["tool"], groups["rc"], device, max_tool)
  tc = report["tool_calling"]

  print(f"       Routing={tc['routing_accuracy']}  Name={tc['name_accuracy']}  Full={tc['full_match_accuracy']}")

  print("\n[3/5] Refusal rate...")

  report["refusal"] = eval_refusal(model, tokenizer, groups["refusal"], groups["rc"], device, max_refusal)
  ref = report["refusal"]

  print(f"       Correct={ref['correct_refusal_rate']}  FalseRefusal={ref['false_refusal_rate']}")

  print("\n[4/5] Hallucination probe...")

  report["hallucination"] = eval_hallucination(model, tokenizer, all_test, device, n_hallucination)

  print(f"       Rate={report['hallucination']['hallucination_rate']}")

  if not skip_latency:
    print("\n[5/5] Latency benchmark...")

    report["latency"] = eval_latency(model, tokenizer, device)

    for k, v in report["latency"].items():
      print(f"       {k}: TTFT={v['ttft_ms']}ms  {v['tok_per_sec']} tok/s")
  else:
    print("\n[5/5] Latency benchmark... SKIPPED")

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

  print("Tool Calling:")
  print(f"  Routing Accuracy: {tc['routing_accuracy']} | Name Accuracy: {tc['name_accuracy']}")
  print(f"  Full Match: {tc['full_match_accuracy']} | False Tool-Call Rate: {tc['false_tool_call_rate']}")

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
  parser.add_argument("--max-tool", type=int, default=2000)
  parser.add_argument("--max-refusal", type=int, default=2000)
  parser.add_argument("--n-hallucination", type=int, default=500)
  parser.add_argument("--skip-latency", action="store_true")
  parser.add_argument("--device", type=str, default=None, help="Force device (cuda, mps, cpu)")
  args = parser.parse_args()

  device = get_device(args.device)
  print(f"Using device: {device}")

  if args.checkpoint:
    checkpoint_dir = Path(args.checkpoint)

    report = run_evaluation(
      checkpoint_dir,
      device,
      max_rc=args.max_rc,
      max_tool=args.max_tool,
      max_refusal=args.max_refusal,
      n_hallucination=args.n_hallucination,
      skip_latency=args.skip_latency,
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
        max_tool=args.max_tool,
        max_refusal=args.max_refusal,
        n_hallucination=args.n_hallucination,
        skip_latency=args.skip_latency,
      )

      all_reports[checkpoint_dir.name] = report

    best_name = select_best_checkpoint(all_reports)

    if best_name:
      print(f"\n*** Best checkpoint: {best_name} (score={all_reports[best_name]['composite_score']}) ***")

      import shutil

      best_src = CHECKPOINT_DIR / best_name
      best_dst = CHECKPOINT_DIR / "best"

      if best_dst.exists():
        shutil.rmtree(best_dst)

      shutil.copytree(best_src, best_dst)

      print(f"Copied to {best_dst}")

      final_report = all_reports[best_name]
      final_report["all_scores"] = {k: round(composite_score(v), 4) for k, v in all_reports.items()}
    else:
      final_report = {}

    with open(EVAL_REPORT_PATH, "w") as f:
      json.dump(final_report, f, indent=2)

    print(f"\nReport saved to {EVAL_REPORT_PATH}")


if __name__ == "__main__":
  main()
