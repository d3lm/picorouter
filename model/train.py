"""Production training loop for PicoRouter using MLX."""

import argparse
import json
import math
import random
import signal
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils
import numpy as np

from model.architecture import PicoConfig, PicoRouter, count_params_flat

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = Path(__file__).parent / "training_log.jsonl"

_interrupted = False


def _handle_signal(signum, _frame):
  global _interrupted
  _interrupted = True
  print(f"\n[signal {signum}] Graceful shutdown requested, saving checkpoint...")


def load_npz_split(name: str) -> tuple[np.ndarray, np.ndarray]:
  """Load a pre-tokenized .npz split (memory-mapped for large files)."""
  path = DATA_DIR / f"{name}.npz"
  data = np.load(str(path), mmap_mode="r")
  return data["input_ids"], data["loss_mask"]


def collate_batch_npz(
  input_ids: np.ndarray,
  loss_mask: np.ndarray,
  indices: list[int],
) -> tuple[mx.array, mx.array, mx.array]:
  """Build input/target/mask tensors from pre-tokenized, pre-padded data."""
  batch_ids = input_ids[indices]
  batch_mask = loss_mask[indices]

  input_tokens = mx.array(batch_ids[:, :-1], dtype=mx.int32)
  target_tokens = mx.array(batch_ids[:, 1:], dtype=mx.int32)
  token_mask = mx.array(batch_mask[:, 1:], dtype=mx.float32)

  return input_tokens, target_tokens, token_mask


def loss_fn(model: PicoRouter, inputs: mx.array, targets: mx.array, loss_mask: mx.array) -> mx.array:
  logits, _ = model(inputs)
  per_token_loss = nn.losses.cross_entropy(logits, targets, reduction="none")
  return (per_token_loss * loss_mask).sum() / loss_mask.sum()


def compute_val_loss(
  model: PicoRouter,
  val_ids: np.ndarray,
  val_mask: np.ndarray,
  batch_size: int,
  max_examples: int = 2048,
) -> float:
  """Compute average validation loss on a random subset."""
  num_examples = min(len(val_ids), max_examples)
  indices = random.sample(range(len(val_ids)), num_examples)

  total_loss = 0.0
  total_tokens = 0

  for i in range(0, num_examples, batch_size):
    batch_indices = indices[i : i + batch_size]
    input_tokens, target_tokens, token_mask = collate_batch_npz(val_ids, val_mask, batch_indices)
    logits, _ = model(input_tokens)
    per_token_loss = nn.losses.cross_entropy(logits, target_tokens, reduction="none")
    batch_loss = (per_token_loss * token_mask).sum()
    batch_tokens = token_mask.sum()
    mx.eval(batch_loss, batch_tokens)
    total_loss += batch_loss.item()
    total_tokens += batch_tokens.item()

  return total_loss / max(total_tokens, 1)


def save_checkpoint(model: PicoRouter, config: PicoConfig, step: int, tag: str | None = None):
  name = tag or f"step-{step}"
  ckpt_dir = CHECKPOINT_DIR / name
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  flat_params = dict(mlx.utils.tree_flatten(model.parameters()))
  mx.savez(str(ckpt_dir / "weights.npz"), **flat_params)

  with open(ckpt_dir / "config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

  print(f"  [checkpoint] saved {ckpt_dir}")


def train(
  epochs: int = 3,
  batch_size: int = 64,
  peak_lr: float = 3e-4,
  min_lr: float = 3e-5,
  warmup_steps: int = 2000,
  weight_decay: float = 0.1,
  max_grad_norm: float = 1.0,
  checkpoint_every: int = 1000,
  val_every: int = 500,
  log_every: int = 100,
  seed: int = 42,
):
  random.seed(seed)

  signal.signal(signal.SIGINT, _handle_signal)
  signal.signal(signal.SIGTERM, _handle_signal)

  print("Loading data...")
  train_ids, train_mask = load_npz_split("train")
  val_ids, val_mask = load_npz_split("val")
  num_train_examples = len(train_ids)
  print(f"  Train: {num_train_examples:,} examples | Val: {len(val_ids):,} examples")

  config = PicoConfig()
  model = PicoRouter(config)
  num_params = count_params_flat(model)

  print(f"  Model: {num_params:,} parameters ({num_params / 1e6:.1f}M)")

  steps_per_epoch = math.ceil(num_train_examples / batch_size)
  total_steps = steps_per_epoch * epochs

  print(f"  {steps_per_epoch:,} steps/epoch x {epochs} epochs = {total_steps:,} total steps")

  warmup = optim.linear_schedule(0.0, peak_lr, steps=warmup_steps)
  decay = optim.cosine_decay(peak_lr, total_steps - warmup_steps, min_lr)
  lr_schedule = optim.join_schedules([warmup, decay], [warmup_steps])

  optimizer = optim.AdamW(
    learning_rate=lr_schedule,
    weight_decay=weight_decay,
    betas=[0.9, 0.95],
  )

  loss_and_grad = nn.value_and_grad(model, loss_fn)

  CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

  print(f"\nTraining for {total_steps:,} steps  (bs={batch_size}, peak_lr={peak_lr})")
  print("=" * 70)

  global_step = 0
  start_time = time.time()
  epoch_indices = list(range(num_train_examples))

  with open(LOG_PATH, "w") as log_file:
    for epoch in range(1, epochs + 1):
      random.shuffle(epoch_indices)
      print(f"\n--- Epoch {epoch}/{epochs} ---")

      for batch_start in range(0, num_train_examples, batch_size):
        if _interrupted:
          save_checkpoint(model, config, global_step, tag=f"interrupted-step-{global_step}")
          print("Training interrupted. Checkpoint saved.")
          return

        global_step += 1
        batch_indices = epoch_indices[batch_start : batch_start + batch_size]

        if len(batch_indices) < 2:
          continue

        input_tokens, target_tokens, token_mask = collate_batch_npz(train_ids, train_mask, batch_indices)

        loss, gradients = loss_and_grad(model, input_tokens, target_tokens, token_mask)
        gradients, gradient_norm = optim.clip_grad_norm(gradients, max_norm=max_grad_norm)
        optimizer.update(model, gradients)

        mx.eval(model.parameters(), optimizer.state, loss, gradient_norm)

        current_lr = optimizer.learning_rate.item()
        train_loss = loss.item()

        if global_step % log_every == 0 or global_step == 1:
          elapsed = time.time() - start_time
          tokens_this_step = int(token_mask.sum().item())
          tokens_per_sec = tokens_this_step / max(elapsed / global_step, 1e-9)

          print(
            f"  step {global_step:6d}/{total_steps}  "
            f"loss={train_loss:.4f}  lr={current_lr:.2e}  "
            f"gnorm={gradient_norm.item():.2f}  "
            f"tok/s={tokens_per_sec:.0f}  "
            f"elapsed={elapsed:.0f}s"
          )

        if global_step % val_every == 0:
          val_loss = compute_val_loss(model, val_ids, val_mask, batch_size)
          print(f"  >>> val_loss={val_loss:.4f} (step {global_step})")
        else:
          val_loss = None

        log_entry = {
          "step": global_step,
          "epoch": epoch,
          "train_loss": round(train_loss, 6),
          "lr": round(current_lr, 8),
          "wall_clock": round(time.time() - start_time, 1),
        }

        if val_loss is not None:
          log_entry["val_loss"] = round(val_loss, 6)

        log_file.write(json.dumps(log_entry) + "\n")
        log_file.flush()

        if global_step % checkpoint_every == 0:
          save_checkpoint(model, config, global_step)

    save_checkpoint(model, config, global_step, tag=f"final-step-{global_step}")

  elapsed = time.time() - start_time

  print("=" * 70)
  print(f"Training complete — {global_step:,} steps in {elapsed / 3600:.1f}h")


def main():
  parser = argparse.ArgumentParser(description="Train PicoRouter")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--peak-lr", type=float, default=3e-4)
  parser.add_argument("--min-lr", type=float, default=3e-5)
  parser.add_argument("--warmup-steps", type=int, default=2000)
  parser.add_argument("--weight-decay", type=float, default=0.1)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  parser.add_argument("--checkpoint-every", type=int, default=1000)
  parser.add_argument("--val-every", type=int, default=500)
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()

  train(
    epochs=args.epochs,
    batch_size=args.batch_size,
    peak_lr=args.peak_lr,
    min_lr=args.min_lr,
    warmup_steps=args.warmup_steps,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    checkpoint_every=args.checkpoint_every,
    val_every=args.val_every,
    log_every=args.log_every,
    seed=args.seed,
  )


if __name__ == "__main__":
  main()
