"""Export PicoRouter from PyTorch checkpoint to ONNX format."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model.architecture import PicoConfig, PicoRouter

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
EXPORT_DIR = Path(__file__).parent / "exports"


def export(checkpoint_path: Path | None = None):
  if checkpoint_path is None:
    checkpoint_path = CHECKPOINT_DIR / "best"

  config_path = checkpoint_path / "config.json"
  weights_path = checkpoint_path / "weights.pt"

  if not weights_path.exists():
    print(f"ERROR: Checkpoint not found at {weights_path}")
    print("Run training first: uv run python -m model.train")
    return

  with open(config_path) as config_file:
    config_dict = json.load(config_file)

  config = PicoConfig(**config_dict)

  print(f"Loaded config: {config}")

  print("Loading PyTorch weights...")

  model = PicoRouter(config)

  state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)

  model.load_state_dict(state_dict)

  model.eval()

  num_params = sum(p.numel() for p in model.parameters())

  print(f"  {num_params:,} parameters loaded")

  EXPORT_DIR.mkdir(parents=True, exist_ok=True)

  onnx_path = EXPORT_DIR / "picorouter.onnx"

  print(f"Exporting to ONNX at {onnx_path}...")

  dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

  class ForwardWithoutCache(torch.nn.Module):
    """Thin wrapper that drops the KV-cache output for static ONNX export."""

    def __init__(self, inner: PicoRouter):
      super().__init__()
      self.inner = inner

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
      logits, _ = self.inner(token_ids)
      return logits

  export_model = ForwardWithoutCache(model)

  torch.onnx.export(
    export_model,
    (dummy_input,),
    str(onnx_path),
    input_names=["input_ids"],
    output_names=["logits"],
    dynamic_axes={
      "input_ids": {0: "batch", 1: "seq_len"},
      "logits": {0: "batch", 1: "seq_len"},
    },
    opset_version=17,
    dynamo=False,
  )

  import onnx

  onnx_model = onnx.load(str(onnx_path), load_external_data=True)

  onnx.save_model(onnx_model, str(onnx_path), save_as_external_data=False)

  data_file = onnx_path.with_suffix(".onnx.data")

  if data_file.exists():
    data_file.unlink()

  print(f"  ONNX file saved: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")

  print("Verifying with ONNX Runtime...")

  import onnxruntime as ort

  session = ort.InferenceSession(str(onnx_path))
  test_input = np.array([[1, 2, 3, 4, 5]], dtype=np.int64)
  outputs = session.run(None, {"input_ids": test_input})

  print(f"  Input shape:  {test_input.shape}")
  print(f"  Output shape: {outputs[0].shape}")
  print(f"  Output sample (first 5 logits): {outputs[0][0, -1, :5]}")
  print("\nONNX export and verification successful!")


def main():
  parser = argparse.ArgumentParser(description="Export PicoRouter to ONNX")

  parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to checkpoint directory (default: checkpoints/best)",
  )

  args = parser.parse_args()

  checkpoint_path = Path(args.checkpoint) if args.checkpoint else None

  export(checkpoint_path)


if __name__ == "__main__":
  main()
