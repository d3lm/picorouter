"""Production training loop for PicoRouter using PyTorch."""

import argparse
import json
import random
import signal
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model.architecture import PicoConfig, PicoRouter

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
LOG_PATH = Path(__file__).parent / "training_log.jsonl"

_interrupted = False


def _handle_signal(signum, _frame):
  global _interrupted
  _interrupted = True
  print(f"\n[signal {signum}] Graceful shutdown requested, saving checkpoint...")


def get_device(override: str | None = None) -> torch.device:
  if override:
    return torch.device(override)

  if torch.cuda.is_available():
    try:
      torch.cuda.init()
      return torch.device("cuda")
    except RuntimeError:
      pass

  if torch.backends.mps.is_available():
    return torch.device("mps")

  return torch.device("cpu")


class TokenDataset(Dataset):
  """Wraps pre-tokenized numpy arrays as a PyTorch Dataset."""

  def __init__(self, input_ids: np.ndarray, loss_mask: np.ndarray):
    self.input_ids = input_ids
    self.loss_mask = loss_mask

  def __len__(self):
    return len(self.input_ids)

  def __getitem__(self, idx):
    ids = self.input_ids[idx]
    mask = self.loss_mask[idx]
    input_tokens = torch.tensor(ids[:-1], dtype=torch.long)
    target_tokens = torch.tensor(ids[1:], dtype=torch.long)
    token_mask = torch.tensor(mask[1:], dtype=torch.float32)
    return input_tokens, target_tokens, token_mask


def load_npz_split(name: str) -> tuple[np.ndarray, np.ndarray]:
  """Load a pre-tokenized .npz split (memory-mapped for large files)."""
  path = DATA_DIR / f"{name}.npz"
  data = np.load(str(path), mmap_mode="r")
  return data["input_ids"], data["loss_mask"]


def compute_val_loss(
  model: PicoRouter,
  val_ids: np.ndarray,
  val_mask: np.ndarray,
  batch_size: int,
  device: torch.device,
  max_examples: int = 2048,
  use_amp: bool = False,
) -> float:
  """Compute average validation loss on a random subset."""
  model.eval()

  num_examples = min(len(val_ids), max_examples)

  indices = random.sample(range(len(val_ids)), num_examples)

  total_loss = 0.0
  total_tokens = 0

  with torch.no_grad():
    for i in range(0, num_examples, batch_size):
      batch_indices = indices[i : i + batch_size]
      batch_ids = val_ids[batch_indices]
      batch_mask_np = val_mask[batch_indices]

      input_tokens = torch.tensor(batch_ids[:, :-1], dtype=torch.long, device=device)
      target_tokens = torch.tensor(batch_ids[:, 1:], dtype=torch.long, device=device)
      token_mask = torch.tensor(batch_mask_np[:, 1:], dtype=torch.float32, device=device)

      with torch.amp.autocast("cuda", enabled=use_amp):
        logits, _ = model(input_tokens)
        per_token_loss = F.cross_entropy(logits.transpose(1, 2), target_tokens, reduction="none")

      batch_loss = (per_token_loss * token_mask).sum()
      batch_tokens = token_mask.sum()

      total_loss += batch_loss.item()
      total_tokens += batch_tokens.item()

  model.train()

  return total_loss / max(total_tokens, 1)


def save_checkpoint(model: PicoRouter, config: PicoConfig, step: int, tag: str | None = None):
  name = tag or f"step-{step}"
  ckpt_dir = CHECKPOINT_DIR / name
  ckpt_dir.mkdir(parents=True, exist_ok=True)

  torch.save(model.state_dict(), ckpt_dir / "weights.pt")

  with open(ckpt_dir / "config.json", "w") as f:
    json.dump(config.__dict__, f, indent=2)

  print(f"  [checkpoint] saved {ckpt_dir}")


def train(
  epochs: int = 3,
  batch_size: int = 16,
  peak_lr: float = 3e-4,
  min_lr: float = 3e-5,
  warmup_steps: int = 2000,
  weight_decay: float = 0.1,
  max_grad_norm: float = 1.0,
  checkpoint_every: int = 1000,
  val_every: int = 500,
  log_every: int = 100,
  seed: int = 42,
  device_override: str | None = None,
):
  random.seed(seed)

  torch.manual_seed(seed)

  device = get_device(device_override)

  print(f"Using device: {device}")

  signal.signal(signal.SIGINT, _handle_signal)
  signal.signal(signal.SIGTERM, _handle_signal)

  print("Loading data...")

  train_ids, train_mask = load_npz_split("train")
  val_ids, val_mask = load_npz_split("val")
  num_train_examples = len(train_ids)

  print(f"  Train: {num_train_examples:,} examples | Val: {len(val_ids):,} examples")

  config = PicoConfig()
  model = PicoRouter(config).to(device)
  num_params = sum(p.numel() for p in model.parameters())

  print(f"  Model: {num_params:,} parameters ({num_params / 1e6:.1f}M)")

  steps_per_epoch = num_train_examples // batch_size
  total_steps = steps_per_epoch * epochs

  print(f"  {steps_per_epoch:,} steps/epoch x {epochs} epochs = {total_steps:,} total steps")

  optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=peak_lr,
    betas=(0.9, 0.95),
    weight_decay=weight_decay,
  )

  warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1e-8 / peak_lr, end_factor=1.0, total_iters=warmup_steps
  )

  decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=total_steps - warmup_steps, eta_min=min_lr
  )

  scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer, schedulers=[warmup_scheduler, decay_scheduler], milestones=[warmup_steps]
  )

  use_amp = device.type == "cuda"
  scaler = torch.amp.GradScaler(enabled=use_amp)

  train_dataset = TokenDataset(train_ids, train_mask)

  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2,
    pin_memory=use_amp,
    drop_last=True,
    persistent_workers=True,
  )

  raw_model = model

  if device.type == "cuda":
    model = torch.compile(model)

  CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

  print(f"\nTraining for {total_steps:,} steps  (bs={batch_size}, peak_lr={peak_lr})")
  print("=" * 70)

  global_step = 0
  best_val_loss = None
  start_time = time.time()

  model.train()

  with open(LOG_PATH, "w") as log_file:
    for epoch in range(1, epochs + 1):
      print(f"\n--- Epoch {epoch}/{epochs} ---")

      for input_tokens, target_tokens, token_mask in train_loader:
        if _interrupted:
          save_checkpoint(raw_model, config, global_step, tag=f"interrupted-step-{global_step}")
          print("Training interrupted. Checkpoint saved.")
          return

        global_step += 1

        input_tokens = input_tokens.to(device, non_blocking=True)
        target_tokens = target_tokens.to(device, non_blocking=True)
        token_mask = token_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
          logits, _ = model(input_tokens)
          per_token_loss = F.cross_entropy(logits.transpose(1, 2), target_tokens, reduction="none")
          loss = (per_token_loss * token_mask).sum() / token_mask.sum()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        scaler.step(optimizer)

        scaler.update()

        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]
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
          val_loss = compute_val_loss(model, val_ids, val_mask, batch_size, device, use_amp=use_amp)

          model.train()

          print(f"  >>> val_loss={val_loss:.4f} (step {global_step})")

          if best_val_loss is None or val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(raw_model, config, global_step, tag="best")
            print("  >>> new best val_loss — checkpoint saved")
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
          save_checkpoint(raw_model, config, global_step)

    save_checkpoint(raw_model, config, global_step, tag=f"final-step-{global_step}")

  elapsed = time.time() - start_time

  print("=" * 70)

  print(f"Training complete — {global_step:,} steps in {elapsed / 3600:.1f}h")


def main():
  parser = argparse.ArgumentParser(description="Train PicoRouter")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--peak-lr", type=float, default=3e-4)
  parser.add_argument("--min-lr", type=float, default=3e-5)
  parser.add_argument("--warmup-steps", type=int, default=2000)
  parser.add_argument("--weight-decay", type=float, default=0.1)
  parser.add_argument("--max-grad-norm", type=float, default=1.0)
  parser.add_argument("--checkpoint-every", type=int, default=1000)
  parser.add_argument("--val-every", type=int, default=500)
  parser.add_argument("--log-every", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--device", type=str, default=None, help="Force device (cuda, mps, cpu)")

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
    device_override=args.device,
  )


if __name__ == "__main__":
  main()
