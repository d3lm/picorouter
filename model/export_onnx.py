"""Export PicoRouter from MLX checkpoint to ONNX format via PyTorch."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
EXPORT_DIR = Path(__file__).parent / "exports"


@dataclass
class PicoConfig:
  vocab_size: int = 8192
  hidden_dim: int = 384
  num_layers: int = 6
  num_heads: int = 6
  ffn_dim: int = 1024
  max_seq_len: int = 1024
  rope_theta: float = 10000.0


def load_mlx_weights(checkpoint_path: Path) -> dict[str, np.ndarray]:
  """Load MLX .npz checkpoint as numpy arrays."""
  import mlx.core as mx

  raw = mx.load(str(checkpoint_path))

  return {k: np.array(v) for k, v in raw.items()}


def build_pytorch_model(config: PicoConfig):
  """Build a PyTorch model matching the MLX architecture (no KV-cache, for export)."""

  import torch
  import torch.nn as torch_nn
  import torch.nn.functional as F

  class RMSNorm(torch_nn.Module):
    def __init__(self, dim, eps=1e-6):
      super().__init__()
      self.weight = torch_nn.Parameter(torch.ones(dim))
      self.eps = eps

    def forward(self, x):
      norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
      return x * norm * self.weight

  class RotaryEmbedding(torch_nn.Module):
    def __init__(self, dim, max_seq_len, theta=10000.0):
      super().__init__()
      freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
      t = torch.arange(max_seq_len, dtype=torch.float32)
      angles = t.unsqueeze(1) * freqs.unsqueeze(0)
      self.register_buffer("cos_cached", torch.cos(angles))
      self.register_buffer("sin_cached", torch.sin(angles))

    def forward(self, x, seq_len):
      return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

  def apply_rotary(x, cos, sin):
    # x: [B, heads, seq_len, head_dim], cos/sin: [seq_len, head_dim/2]
    cos = cos.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(1)
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

  class Attention(torch_nn.Module):
    def __init__(self, cfg):
      super().__init__()
      self.num_heads = cfg.num_heads
      self.head_dim = cfg.hidden_dim // cfg.num_heads
      self.q_proj = torch_nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
      self.k_proj = torch_nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
      self.v_proj = torch_nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
      self.o_proj = torch_nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
      self.rope = RotaryEmbedding(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x, mask=None):
      B, L, _ = x.shape
      q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
      k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
      v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

      cos, sin = self.rope(q, L)
      q = apply_rotary(q, cos, sin)
      k = apply_rotary(k, cos, sin)

      scale = self.head_dim**-0.5
      attn = (q @ k.transpose(-2, -1)) * scale
      if mask is not None:
        attn = attn + mask
      attn = F.softmax(attn, dim=-1)
      out = (attn @ v).transpose(1, 2).reshape(B, L, -1)
      return self.o_proj(out)

  class SwiGLUFFN(torch_nn.Module):
    def __init__(self, cfg):
      super().__init__()
      self.gate_proj = torch_nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
      self.up_proj = torch_nn.Linear(cfg.hidden_dim, cfg.ffn_dim, bias=False)
      self.down_proj = torch_nn.Linear(cfg.ffn_dim, cfg.hidden_dim, bias=False)

    def forward(self, x):
      return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

  class TransformerBlock(torch_nn.Module):
    def __init__(self, cfg):
      super().__init__()
      self.attention = Attention(cfg)
      self.ffn = SwiGLUFFN(cfg)
      self.attn_norm = RMSNorm(cfg.hidden_dim)
      self.ffn_norm = RMSNorm(cfg.hidden_dim)

    def forward(self, x, mask=None):
      x = x + self.attention(self.attn_norm(x), mask)
      x = x + self.ffn(self.ffn_norm(x))
      return x

  class PicoRouterTorch(torch_nn.Module):
    def __init__(self, cfg):
      super().__init__()
      self.embed = torch_nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
      self.layers = torch_nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
      self.norm = RMSNorm(cfg.hidden_dim)

    def forward(self, token_ids):
      seq_len = token_ids.shape[1]
      mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
      mask = mask.unsqueeze(0).unsqueeze(0)

      x = self.embed(token_ids)
      for layer in self.layers:
        x = layer(x, mask)
      x = self.norm(x)
      logits = x @ self.embed.weight.T
      return logits

  return PicoRouterTorch(config)


def map_mlx_to_pytorch(mlx_weights: dict[str, np.ndarray], config: PicoConfig) -> dict:
  """Map MLX flat weight keys to PyTorch state_dict keys."""
  import torch

  mapping = {}

  mapping["embed.weight"] = torch.from_numpy(mlx_weights["embed.weight"])
  mapping["norm.weight"] = torch.from_numpy(mlx_weights["norm.weight"])

  for i in range(config.num_layers):
    prefix_mlx = f"layers.{i}"
    prefix_pt = f"layers.{i}"

    for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
      key_mlx = f"{prefix_mlx}.attention.{proj}.weight"
      key_pt = f"{prefix_pt}.attention.{proj}.weight"
      mapping[key_pt] = torch.from_numpy(mlx_weights[key_mlx])

    for proj in ["gate_proj", "up_proj", "down_proj"]:
      key_mlx = f"{prefix_mlx}.ffn.{proj}.weight"
      key_pt = f"{prefix_pt}.ffn.{proj}.weight"
      mapping[key_pt] = torch.from_numpy(mlx_weights[key_mlx])

    mapping[f"{prefix_pt}.attn_norm.weight"] = torch.from_numpy(mlx_weights[f"{prefix_mlx}.attn_norm.weight"])
    mapping[f"{prefix_pt}.ffn_norm.weight"] = torch.from_numpy(mlx_weights[f"{prefix_mlx}.ffn_norm.weight"])

  return mapping


def export():
  import torch

  config_path = CHECKPOINT_DIR / "config.json"
  checkpoint_path = CHECKPOINT_DIR / "tracer_bullet.npz"

  if not checkpoint_path.exists():
    print(f"ERROR: Checkpoint not found at {checkpoint_path}")
    print("Run training first: uv run python -m model.train")
    return

  with open(config_path) as f:
    config_dict = json.load(f)
  config = PicoConfig(**config_dict)

  print(f"Loaded config: {config}")

  print("Loading MLX weights...")

  mlx_weights = load_mlx_weights(checkpoint_path)

  print(f"  Loaded {len(mlx_weights)} weight tensors")

  print("Building PyTorch model...")

  pt_model = build_pytorch_model(config)

  print("Mapping weights...")

  state_dict = map_mlx_to_pytorch(mlx_weights, config)
  pt_model.load_state_dict(state_dict, strict=False)
  pt_model.eval()

  EXPORT_DIR.mkdir(parents=True, exist_ok=True)

  onnx_path = EXPORT_DIR / "picorouter.onnx"

  print(f"Exporting to ONNX at {onnx_path}...")

  dummy_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)

  torch.onnx.export(
    pt_model,
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


if __name__ == "__main__":
  export()
