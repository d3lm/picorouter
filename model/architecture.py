"""LLaMA-style ~14M-parameter decoder-only transformer in PyTorch."""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PicoConfig:
  vocab_size: int = 8192
  hidden_dim: int = 384
  num_layers: int = 6
  num_heads: int = 6
  ffn_dim: int = 1024
  max_seq_len: int = 1024
  rope_theta: float = 10000.0


class RMSNorm(nn.Module):
  def __init__(self, dim: int, eps: float = 1e-6):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(dim))
    self.eps = eps

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return x * norm * self.weight


def apply_rope(
  x: torch.Tensor,
  cos_freq: torch.Tensor,
  sin_freq: torch.Tensor,
  offset: int = 0,
  position_ids: torch.Tensor | None = None,
) -> torch.Tensor:
  if position_ids is not None:
    cos_f = cos_freq[position_ids].unsqueeze(1)
    sin_f = sin_freq[position_ids].unsqueeze(1)
  else:
    seq_len = x.shape[2]
    cos_f = cos_freq[offset : offset + seq_len].unsqueeze(0).unsqueeze(1)
    sin_f = sin_freq[offset : offset + seq_len].unsqueeze(0).unsqueeze(1)

  x1 = x[..., ::2]
  x2 = x[..., 1::2]

  return torch.cat([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], dim=-1)


class Attention(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()

    self.num_heads = config.num_heads
    self.head_dim = config.hidden_dim // config.num_heads

    self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

  def forward(
    self,
    x: torch.Tensor,
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    attn_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    B, L, _ = x.shape

    q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

    if position_ids is not None:
      q = apply_rope(q, cos_freq, sin_freq, position_ids=position_ids)
      k = apply_rope(k, cos_freq, sin_freq, position_ids=position_ids)
    else:
      offset = 0

      if cache is not None:
        offset = cache[0].shape[2]

      q = apply_rope(q, cos_freq, sin_freq, offset)
      k = apply_rope(k, cos_freq, sin_freq, offset)

    if cache is not None:
      k = torch.cat([cache[0], k], dim=2)
      v = torch.cat([cache[1], v], dim=2)

    new_cache = (k, v)

    if attn_mask is not None:
      out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
    else:
      # keep this as a Python bool for ONNX tracing compatibility
      is_causal = cache is None
      out = F.scaled_dot_product_attention(q, k, v, is_causal=is_causal)

    out = out.transpose(1, 2).reshape(B, L, -1)

    return self.o_proj(out), new_cache


class SwiGLUFFN(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.attention = Attention(config)
    self.ffn = SwiGLUFFN(config)
    self.attn_norm = RMSNorm(config.hidden_dim)
    self.ffn_norm = RMSNorm(config.hidden_dim)

  def forward(
    self,
    x: torch.Tensor,
    cos_freq: torch.Tensor,
    sin_freq: torch.Tensor,
    cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    attn_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    h, new_cache = self.attention(self.attn_norm(x), cos_freq, sin_freq, cache, attn_mask, position_ids)
    x = x + h
    x = x + self.ffn(self.ffn_norm(x))
    return x, new_cache


class PicoRouter(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.config = config
    self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
    self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
    self.norm = RMSNorm(config.hidden_dim)

    head_dim = config.hidden_dim // config.num_heads
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(config.max_seq_len, dtype=torch.float32)
    angles = t.unsqueeze(1) * freqs.unsqueeze(0)
    self.register_buffer("_cos_freq", torch.cos(angles))
    self.register_buffer("_sin_freq", torch.sin(angles))

  def forward(
    self,
    token_ids: torch.Tensor,
    cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
    attn_mask: torch.Tensor | None = None,
    position_ids: torch.Tensor | None = None,
  ) -> tuple[torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
    x = self.embed(token_ids)

    if cache is None:
      cache = [None] * self.config.num_layers

    new_cache = []

    for i, layer in enumerate(self.layers):
      x, layer_cache = layer(x, self._cos_freq, self._sin_freq, cache[i], attn_mask, position_ids)
      new_cache.append(layer_cache)

    x = self.norm(x)

    logits = x @ self.embed.weight.T

    return logits, new_cache


if __name__ == "__main__":
  config = PicoConfig()
  model = PicoRouter(config)

  num_params = sum(p.numel() for p in model.parameters())

  print("PicoRouter model instantiated")
  print(f"  Config: {config}")
  print(f"  Parameters: {num_params:,} ({num_params / 1e6:.1f}M)")

  dummy_input = torch.tensor([[1, 2, 3, 4, 5]])
  logits, _ = model(dummy_input)

  print(f"  Input shape:  {dummy_input.shape}")
  print(f"  Output shape: {logits.shape}")
  print(f"  Vocab size:   {config.vocab_size}")
