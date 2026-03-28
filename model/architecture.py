"""LLaMA-style 12M-parameter decoder-only transformer in MLX."""

from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn
from mlx.core import array


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
    self.weight = mx.ones((dim,))
    self.eps = eps

  def __call__(self, x: mx.array) -> mx.array:
    norm = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
    return x * norm * self.weight


def precompute_rope_frequencies(dim: int, max_seq_len: int, theta: float = 10000.0) -> tuple[mx.array, mx.array]:
  freqs = 1.0 / (theta ** (mx.arange(0, dim, 2, dtype=mx.float32) / dim))
  t = mx.arange(max_seq_len, dtype=mx.float32)
  angles = mx.expand_dims(t, 1) * mx.expand_dims(freqs, 0)
  return mx.cos(angles), mx.sin(angles)


def apply_rope(x: mx.array, cos_freq: mx.array, sin_freq: mx.array, offset: int = 0) -> mx.array:
  seq_len = x.shape[1]
  cos_f = cos_freq[offset : offset + seq_len]
  sin_f = sin_freq[offset : offset + seq_len]

  cos_f = mx.expand_dims(mx.expand_dims(cos_f, 0), 2)
  sin_f = mx.expand_dims(mx.expand_dims(sin_f, 0), 2)

  x1 = x[..., ::2]
  x2 = x[..., 1::2]

  rotated = mx.concatenate([x1 * cos_f - x2 * sin_f, x1 * sin_f + x2 * cos_f], axis=-1)

  return rotated


class Attention(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.num_heads = config.num_heads
    self.head_dim = config.hidden_dim // config.num_heads

    self.q_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.k_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.v_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)
    self.o_proj = nn.Linear(config.hidden_dim, config.hidden_dim, bias=False)

  def __call__(
    self,
    x: mx.array,
    cos_freq: mx.array,
    sin_freq: mx.array,
    mask: mx.array | None = None,
    cache: tuple[mx.array, mx.array] | None = None,
  ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    B, L, _ = x.shape

    q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    k = self.k_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
    v = self.v_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    offset = 0

    if cache is not None:
      offset = cache[0].shape[2]

    q = apply_rope(q, cos_freq, sin_freq, offset)
    k = apply_rope(k, cos_freq, sin_freq, offset)

    if cache is not None:
      k = mx.concatenate([cache[0], k], axis=2)
      v = mx.concatenate([cache[1], v], axis=2)

    new_cache = (k, v)

    scale = self.head_dim**-0.5
    attn = (q @ k.transpose(0, 1, 3, 2)) * scale

    if mask is not None:
      attn = attn + mask

    attn = mx.softmax(attn, axis=-1)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, -1)

    return self.o_proj(out), new_cache


class SwiGLUFFN(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.up_proj = nn.Linear(config.hidden_dim, config.ffn_dim, bias=False)
    self.down_proj = nn.Linear(config.ffn_dim, config.hidden_dim, bias=False)

  def __call__(self, x: mx.array) -> mx.array:
    return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.attention = Attention(config)
    self.ffn = SwiGLUFFN(config)
    self.attn_norm = RMSNorm(config.hidden_dim)
    self.ffn_norm = RMSNorm(config.hidden_dim)

  def __call__(
    self,
    x: mx.array,
    cos_freq: mx.array,
    sin_freq: mx.array,
    mask: mx.array | None = None,
    cache: tuple[mx.array, mx.array] | None = None,
  ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
    h, new_cache = self.attention(self.attn_norm(x), cos_freq, sin_freq, mask, cache)
    x = x + h
    x = x + self.ffn(self.ffn_norm(x))
    return x, new_cache


class PicoRouter(nn.Module):
  def __init__(self, config: PicoConfig):
    super().__init__()
    self.config = config
    self.embed = nn.Embedding(config.vocab_size, config.hidden_dim)
    self.layers = [TransformerBlock(config) for _ in range(config.num_layers)]
    self.norm = RMSNorm(config.hidden_dim)

    cos_freq, sin_freq = precompute_rope_frequencies(
      config.hidden_dim // config.num_heads,
      config.max_seq_len,
      config.rope_theta,
    )

    self._cos_freq = cos_freq
    self._sin_freq = sin_freq

  def __call__(
    self,
    token_ids: mx.array,
    cache: list[tuple[mx.array, mx.array]] | None = None,
  ) -> tuple[mx.array, list[tuple[mx.array, mx.array]]]:
    _B, L = token_ids.shape
    x = self.embed(token_ids)

    if cache is None:
      mask = nn.MultiHeadAttention.create_additive_causal_mask(L).astype(x.dtype)
      cache = [None] * self.config.num_layers
    else:
      mask = None

    new_cache = []

    for i, layer in enumerate(self.layers):
      x, layer_cache = layer(x, self._cos_freq, self._sin_freq, mask, cache[i])
      new_cache.append(layer_cache)

    x = self.norm(x)

    logits = x @ self.embed.weight.T

    return logits, new_cache


def count_parameters(model: PicoRouter) -> int:
  total = 0
  for _name, param in model.parameters().items():
    if isinstance(param, mx.array):
      total += param.size
    elif isinstance(param, dict):
      for v in param.values():
        if isinstance(v, mx.array):
          total += v.size
    elif isinstance(param, list):
      for item in param:
        if isinstance(item, mx.array):
          total += item.size
        elif isinstance(item, dict):
          for v in item.values():
            if isinstance(v, mx.array):
              total += v.size
  return total


def count_params_flat(model: PicoRouter) -> int:
  """Count parameters using the flat leaf structure from tree_flatten."""
  import mlx.utils

  leaves = mlx.utils.tree_flatten(model.parameters())
  return sum(p.size for _, p in leaves)


if __name__ == "__main__":
  config = PicoConfig()
  model = PicoRouter(config)

  n_params = count_params_flat(model)

  print("PicoRouter model instantiated")
  print(f"  Config: {config}")
  print(f"  Parameters: {n_params:,} ({n_params / 1e6:.1f}M)")

  dummy_input = mx.array([[1, 2, 3, 4, 5]])
  logits, _ = model[array | list[tuple[array, array]], ...](dummy_input)

  print(f"  Input shape:  {dummy_input.shape}")
  print(f"  Output shape: {logits.shape}")
  print(f"  Vocab size:   {config.vocab_size}")
