"""BPE tokenizer training and loading for PicoRouter."""

import json
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

SPECIAL_TOKENS = [
  "<|pad|>",
  "<|eos|>",
  "<|context|>",
  "<|tools|>",
  "<|user|>",
  "<|assistant|>",
  "<|tool_call|>",
  "<|source|>",
  "<|refuse|>",
]

DEFAULT_VOCAB_SIZE = 512
MODEL_DIR = Path(__file__).parent
TOKENIZER_PATH = MODEL_DIR / "tokenizer.json"
SEED_DATA_PATH = MODEL_DIR.parent / "data" / "processed" / "seed.jsonl"


def collect_texts(data_path: Path) -> list[str]:
  """Extract all text from seed data for tokenizer training."""
  texts = []

  with open(data_path) as file:
    for line in file:
      example = json.loads(line)
      texts.append(example["context"])
      for turn in example["conversation"]:
        texts.append(turn["content"])
      if example.get("tools"):
        texts.append(json.dumps(example["tools"]))

  return texts


def train_tokenizer(vocab_size: int = DEFAULT_VOCAB_SIZE, data_path: Path = SEED_DATA_PATH) -> Tokenizer:
  """Train a BPE tokenizer on seed data."""
  tokenizer = Tokenizer(models.BPE())
  tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

  trainer = trainers.BpeTrainer(
    vocab_size=vocab_size,
    special_tokens=SPECIAL_TOKENS,
    show_progress=True,
    min_frequency=2,
  )

  texts = collect_texts(data_path)
  tokenizer.train_from_iterator(texts, trainer=trainer)

  return tokenizer


def save_tokenizer(tokenizer: Tokenizer, path: Path = TOKENIZER_PATH):
  tokenizer.save(str(path))
  print(f"Saved tokenizer to {path} (vocab size: {tokenizer.get_vocab_size()})")


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
  return Tokenizer.from_file(str(path))


def get_special_token_id(tokenizer: Tokenizer, token: str) -> int:
  token_id = tokenizer.token_to_id(token)

  if token_id is None:
    raise ValueError(f"Special token {token!r} not found in tokenizer vocabulary")

  return token_id


def encode_example(tokenizer: Tokenizer, example: dict) -> list[int]:
  """Encode a training example into the packed token sequence.

  Format: <|context|> {context} <|tools|> {tools} <|user|> {question} <|assistant|> {answer} <|eos|>
  """
  ctx_id = get_special_token_id(tokenizer, "<|context|>")
  tools_id = get_special_token_id(tokenizer, "<|tools|>")
  user_id = get_special_token_id(tokenizer, "<|user|>")
  asst_id = get_special_token_id(tokenizer, "<|assistant|>")
  eos_id = get_special_token_id(tokenizer, "<|eos|>")

  tokens = [ctx_id]
  tokens.extend(tokenizer.encode(example["context"]).ids)
  tokens.append(tools_id)

  if example.get("tools"):
    tokens.extend(tokenizer.encode(json.dumps(example["tools"])).ids)

  for turn in example["conversation"]:
    if turn["role"] == "user":
      tokens.append(user_id)
      tokens.extend(tokenizer.encode(turn["content"]).ids)
    elif turn["role"] == "assistant":
      tokens.append(asst_id)
      tokens.extend(tokenizer.encode(turn["content"]).ids)

  tokens.append(eos_id)

  return tokens


def find_assistant_start(tokenizer: Tokenizer, token_ids: list[int]) -> int:
  """Find the index of the first <|assistant|> token (loss is computed from this point)."""
  asst_id = get_special_token_id(tokenizer, "<|assistant|>")

  for i, tid in enumerate(token_ids):
    if tid == asst_id:
      return i

  raise ValueError("No <|assistant|> token found in sequence")


if __name__ == "__main__":
  print("Training BPE tokenizer...")

  tokenizer = train_tokenizer()

  save_tokenizer(tokenizer)

  test_text = "When was the Eiffel Tower completed?"

  encoded = tokenizer.encode(test_text)
  decoded = tokenizer.decode(encoded.ids)

  print("\nRound-trip test:")
  print(f"  Input:   {test_text!r}")
  print(f"  Tokens:  {encoded.ids}")
  print(f"  Decoded: {decoded!r}")

  with open(SEED_DATA_PATH) as f:
    example = json.loads(f.readline())

  packed = encode_example(tokenizer, example)
  asst_start = find_assistant_start(tokenizer, packed)

  print(f"\nPacked example: {len(packed)} tokens, assistant starts at position {asst_start}")
  print(f"Special tokens: {', '.join(f'{t}={tokenizer.token_to_id(t)}' for t in SPECIAL_TOKENS)}")
