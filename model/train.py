"""Training loop for PicoRouter using MLX."""

import json
import random
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils

from model.architecture import PicoConfig, PicoRouter, count_params_flat
from model.tokenizer import (
  SEED_DATA_PATH,
  TOKENIZER_PATH,
  encode_example,
  find_assistant_start,
  load_tokenizer,
  save_tokenizer,
  train_tokenizer,
)

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def load_and_tokenize_data(tokenizer, data_path: Path, max_seq_len: int) -> list[dict]:
  """Load seed data and encode each example into token sequences with loss masks."""
  examples = []

  with open(data_path) as f:
    for line in f:
      raw = json.loads(line)
      token_ids = encode_example(tokenizer, raw)

      if len(token_ids) > max_seq_len:
        token_ids = token_ids[:max_seq_len]

      asst_start = find_assistant_start(tokenizer, token_ids)

      examples.append({"token_ids": token_ids, "asst_start": asst_start})

  return examples


def collate_batch(examples: list[dict], pad_id: int) -> tuple[mx.array, mx.array, mx.array]:
  """Pad examples to the same length and create input/target/mask tensors."""
  max_len = max(len(ex["token_ids"]) for ex in examples)

  input_ids = []
  target_ids = []
  loss_mask = []

  for example in examples:
    tokens = example["token_ids"]
    asst_start = example["asst_start"]
    pad_len = max_len - len(tokens)

    inp = tokens[:-1] + [pad_id] * pad_len
    tgt = tokens[1:] + [pad_id] * pad_len

    # only compute loss on tokens after <|assistant|> (shifted by 1 for targets)
    mask = [0.0] * max(0, asst_start) + [1.0] * (len(tokens) - 1 - asst_start) + [0.0] * pad_len

    input_ids.append(inp)
    target_ids.append(tgt)
    loss_mask.append(mask)

  return (
    mx.array(input_ids, dtype=mx.int32),
    mx.array(target_ids, dtype=mx.int32),
    mx.array(loss_mask, dtype=mx.float32),
  )


def loss_fn(model: PicoRouter, inputs: mx.array, targets: mx.array, mask: mx.array) -> mx.array:
  logits, _ = model(inputs)
  loss = nn.losses.cross_entropy(logits, targets, reduction="none")
  masked_loss = (loss * mask).sum() / mask.sum()
  return masked_loss


def train(
  num_steps: int = 100,
  batch_size: int = 8,
  lr: float = 3e-4,
  data_path: Path = SEED_DATA_PATH,
  vocab_size: int = 512,
):
  if not TOKENIZER_PATH.exists():
    print("Tokenizer not found, training one...")
    tokenizer = train_tokenizer(vocab_size=vocab_size, data_path=data_path)
    save_tokenizer(tokenizer)
  else:
    print(f"Loading tokenizer from {TOKENIZER_PATH}")

  tokenizer = load_tokenizer()

  pad_id = tokenizer.token_to_id("<|pad|>")

  assert pad_id is not None, "<|pad|> token not in vocabulary"

  config = PicoConfig(vocab_size=tokenizer.get_vocab_size())
  model = PicoRouter(config)

  n_params = count_params_flat(model)

  print(f"Model: {n_params:,} parameters ({n_params / 1e6:.1f}M)")

  examples = load_and_tokenize_data(tokenizer, data_path, config.max_seq_len)

  print(f"Loaded {len(examples)} training examples")

  optimizer = optim.AdamW(learning_rate=lr)
  loss_and_grad = nn.value_and_grad(model, loss_fn)

  CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

  print(f"\nTraining for {num_steps} steps (batch_size={batch_size}, lr={lr})")
  print("-" * 60)

  start_time = time.time()

  for step in range(1, num_steps + 1):
    batch = random.choices(examples, k=batch_size)
    inputs, targets, mask = collate_batch(batch, pad_id)

    loss, grads = loss_and_grad(model, inputs, targets, mask)
    optimizer.update(model, grads)

    mx.eval(model.parameters(), optimizer.state)

    if step % 10 == 0 or step == 1:
      elapsed = time.time() - start_time
      print(f"  step {step:4d}/{num_steps}  loss={loss.item():.4f}  elapsed={elapsed:.1f}s")

  elapsed = time.time() - start_time

  print("-" * 60)
  print(f"Training complete in {elapsed:.1f}s")

  checkpoint_path = CHECKPOINT_DIR / "tracer_bullet.npz"
  flat_params = dict(mlx.utils.tree_flatten(model.parameters()))

  mx.savez(str(checkpoint_path), **flat_params)

  print(f"Saved checkpoint to {checkpoint_path}")

  config_path = CHECKPOINT_DIR / "config.json"

  with open(config_path, "w") as file:
    json.dump(config.__dict__, file, indent=2)

  print(f"Saved config to {config_path}")


if __name__ == "__main__":
  train()
