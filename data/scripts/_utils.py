"""Shared utilities for data pipeline scripts."""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def ensure_nltk_data() -> None:
  """Download punkt_tab tokenizer data if not already present."""
  import nltk

  try:
    nltk.data.find("tokenizers/punkt_tab")
  except LookupError:
    nltk.download("punkt_tab", quiet=True)


def download_file(url: str, dest: Path, *, resume: bool = True) -> Path:
  """Stream-download a file with progress logging.

  When *resume* is True (the default), skips the download if *dest* already
  exists and is non-empty.
  """
  dest.parent.mkdir(parents=True, exist_ok=True)

  if resume and dest.exists() and dest.stat().st_size > 0:
    log.info("Already downloaded, skipping: %s", dest.name)
    return dest

  log.info("Downloading %s", url)

  timeout = httpx.Timeout(connect=30.0, read=120.0, write=30.0, pool=30.0)

  with httpx.stream("GET", url, follow_redirects=True, timeout=timeout) as response:
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(dest, "wb") as file:
      for chunk in response.iter_bytes(chunk_size=64 * 1024):
        file.write(chunk)
        downloaded += len(chunk)

        if total and downloaded % (10 * 1024 * 1024) < 64 * 1024:
          log.info("  %d / %d MB (%.0f%%)", downloaded >> 20, total >> 20, downloaded / total * 100)

  log.info("Saved %s (%d MB)", dest.name, dest.stat().st_size >> 20)

  return dest
