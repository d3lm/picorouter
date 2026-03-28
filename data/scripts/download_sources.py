"""Download source corpora (Wikipedia dumps) and chunk into passages.

Usage:
  uv run python -m data.scripts.download_sources --source simplewiki
  uv run python -m data.scripts.download_sources --source enwiki_curated
  uv run python -m data.scripts.download_sources --source all --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import mwparserfromhell
from nltk.tokenize import sent_tokenize

from data.scripts._utils import download_file, ensure_nltk_data

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
SOURCES_DIR = ROOT / "data" / "sources"
DUMPS_DIR = SOURCES_DIR / ".dumps"

SIMPLEWIKI_URL = "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
ENWIKI_API = "https://en.wikipedia.org/w/api.php"

MIN_WORDS = 150
MAX_WORDS = 400


def wiki_to_plaintext(wikitext: str) -> str:
  """Strip wikitext markup and return clean prose (used for API-fetched articles)."""
  parsed = mwparserfromhell.parse(wikitext)

  text = parsed.strip_code(normalize=True, collapse=True)
  text = re.sub(r"\[\[(?:Category|File|Image):[^\]]*\]\]", "", text, flags=re.IGNORECASE)
  text = re.sub(r"\{\{[^}]*\}\}", "", text)
  text = re.sub(r"<[^>]+>", "", text)
  text = re.sub(r"[ \t]+", " ", text)
  text = re.sub(r"\n{3,}", "\n\n", text)

  return text.strip()


def should_skip(title: str, wikitext: str) -> str | None:
  """Return a reason string if the article should be skipped, else None.
  Operates on raw wikitext; used for the enwiki API path.
  """
  if ":" in title:
    return "namespace-prefix"

  lower = wikitext.lower()

  if "disambiguation" in title.lower():
    return "disambiguation"

  if "{{disambig" in lower or "{{disambiguation" in lower:
    return "disambiguation"

  if re.search(r"\{\{[^}]*-stub\s*\}\}", lower) or "{{stub}}" in lower:
    return "stub"

  lines = [line.strip() for line in wikitext.split("\n") if line.strip()]

  if lines:
    list_frac = sum(1 for ln in lines if ln.startswith(("*", "#"))) / len(lines)

    if list_frac > 0.6:
      return "list-heavy"

  return None


def chunk_article(text: str, title: str, page_id: str, source: str) -> list[dict]:
  """Split article text into 150-400 word passages, preserving sentence boundaries."""
  paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
  chunks: list[dict] = []
  buf: list[str] = []
  buf_wc = 0
  idx = 0

  def emit():
    nonlocal buf, buf_wc, idx

    if buf and buf_wc >= MIN_WORDS:
      chunks.append(
        {
          "id": f"{source}-{page_id}-{idx}",
          "title": title,
          "text": " ".join(buf),
          "word_count": buf_wc,
        }
      )

      idx += 1

    buf, buf_wc = [], 0

  for paragraph in paragraphs:
    for sentence in sent_tokenize(paragraph):
      swc = len(sentence.split())

      if buf_wc + swc > MAX_WORDS and buf_wc >= MIN_WORDS:
        emit()

      buf.append(sentence)
      buf_wc += swc

  emit()

  return chunks


def _iter_extracted_articles(extracted_dir: Path):
  """Yield (id, title, text) from wikiextractor JSON output files."""
  for subdir in sorted(extracted_dir.iterdir()):
    if not subdir.is_dir():
      continue

    for file_path in sorted(subdir.iterdir()):
      with open(file_path, encoding="utf-8") as file:
        for line in file:
          line = line.strip()

          if not line:
            continue

          article = json.loads(line)

          yield article["id"], article["title"], article["text"]


def process_simplewiki(*, resume: bool = False) -> Path:
  """Download Simple English Wikipedia, extract with wikiextractor, and chunk into passage JSONL."""
  dump_path = DUMPS_DIR / "simplewiki-latest-pages-articles.xml.bz2"
  output_path = SOURCES_DIR / "simplewiki_passages.jsonl"
  extracted_dir = DUMPS_DIR / "simplewiki_extracted"

  download_file(SIMPLEWIKI_URL, dump_path, resume=resume)

  if not (resume and extracted_dir.exists() and any(extracted_dir.iterdir())):
    log.info("Extracting with wikiextractor...")

    subprocess.run(
      [sys.executable, "-m", "wikiextractor", str(dump_path), "-o", str(extracted_dir), "--json"],
      check=True,
    )

  log.info("Processing Simple English Wikipedia...")

  articles = 0
  passages = 0

  with open(output_path, "w", encoding="utf-8") as out:
    for page_id, title, text in _iter_extracted_articles(extracted_dir):
      if len(text.split()) < MIN_WORDS:
        continue

      articles += 1

      for chunk in chunk_article(text, title, page_id, "simplewiki"):
        out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
        passages += 1

      if articles % 2000 == 0:
        log.info("  %d articles -> %d passages ...", articles, passages)

  log.info("simplewiki complete: %d articles, %d passages", articles, passages)

  return output_path


def _api_get(client: httpx.Client, **params) -> dict:
  params.setdefault("format", "json")
  response = client.get(ENWIKI_API, params=params)
  response.raise_for_status()
  return response.json()


def _fetch_category_titles(client: httpx.Client, category: str, *, limit: int = 10_000) -> list[str]:
  """Paginate through a Wikipedia category and collect article titles."""
  titles: list[str] = []

  params = {
    "action": "query",
    "list": "categorymembers",
    "cmtitle": category,
    "cmtype": "page",
    "cmlimit": "500",
  }

  while len(titles) < limit:
    data = _api_get(client, **params)
    titles.extend(member["title"] for member in data["query"]["categorymembers"])
    continue_token = data.get("continue", {}).get("cmcontinue")

    if not continue_token:
      break

    params["cmcontinue"] = continue_token

    log.info("  %d titles collected ...", len(titles))

  return titles[:limit]


def _fetch_articles_batch(client: httpx.Client, titles: list[str]) -> dict[str, tuple[str, int]]:
  """Fetch wikitext for up to 50 articles. Returns {title: (wikitext, pageid)}."""
  data = _api_get(
    client,
    action="query",
    titles="|".join(titles),
    prop="revisions",
    rvprop="content",
    rvslots="main",
  )

  results: dict[str, tuple[str, int]] = {}

  for page in data.get("query", {}).get("pages", {}).values():
    title = page.get("title", "")
    page_id = page.get("pageid", 0)
    revisions = page.get("revisions", [])

    if not revisions:
      continue

    slot = revisions[0].get("slots", {}).get("main", {})
    wikitext = slot.get("content", slot.get("*", ""))
    results[title] = (wikitext, page_id)

  return results


def process_enwiki_curated(*, resume: bool = False) -> Path:
  """Fetch Featured Articles from English Wikipedia and produce passage JSONL."""
  output_path = SOURCES_DIR / "enwiki_curated_passages.jsonl"
  titles_cache = DUMPS_DIR / "enwiki_featured_titles.json"

  if resume and output_path.exists() and output_path.stat().st_size > 0:
    log.info("Already exists, skipping: %s", output_path.name)
    return output_path

  DUMPS_DIR.mkdir(parents=True, exist_ok=True)

  with httpx.Client(timeout=60) as client:
    if resume and titles_cache.exists():
      titles = json.loads(titles_cache.read_text())
      log.info("Loaded %d cached article titles", len(titles))
    else:
      log.info("Fetching Featured Articles list from English Wikipedia...")
      titles = _fetch_category_titles(client, "Category:Featured articles")
      titles_cache.write_text(json.dumps(titles))
      log.info("Found %d Featured Articles", len(titles))

    passages = 0
    batch_size = 50

    with open(output_path, "w", encoding="utf-8") as out:
      for i in range(0, len(titles), batch_size):
        batch = titles[i : i + batch_size]

        try:
          articles = _fetch_articles_batch(client, batch)
        except Exception:
          log.warning("Batch %d failed, skipping", i // batch_size)
          continue

        for title, (wikitext, page_id) in articles.items():
          if should_skip(title, wikitext):
            continue

          text = wiki_to_plaintext(wikitext)

          if len(text.split()) < MIN_WORDS:
            continue

          for chunk in chunk_article(text, title, str(page_id), "enwiki"):
            out.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            passages += 1

        batch_num = i // batch_size

        if batch_num % 20 == 0:
          log.info("  batch %d/%d, %d passages ...", batch_num, len(titles) // batch_size + 1, passages)

        time.sleep(0.5)

  log.info("enwiki_curated complete: %d passages", passages)

  return output_path


def validate_output(path: Path) -> int:
  """Validate passage JSONL. Returns count of valid records."""
  ok, bad = 0, 0
  required_keys = {"id", "title", "text", "word_count"}

  with open(path, encoding="utf-8") as file:
    for lineno, line in enumerate(file, 1):
      try:
        obj = json.loads(line)
        missing = required_keys - obj.keys()

        if missing:
          raise ValueError(f"missing keys: {missing}")

        ok += 1
      except (json.JSONDecodeError, ValueError) as exc:
        bad += 1

        if bad <= 3:
          log.warning("  line %d: %s", lineno, exc)

  log.info("Validated %s: %d valid, %d errors", path.name, ok, bad)

  return ok


def main():
  parser = argparse.ArgumentParser(description="Download and chunk source corpora into passages")
  parser.add_argument("--source", choices=["simplewiki", "enwiki_curated", "all"], default="all")
  parser.add_argument("--resume", action="store_true", help="Skip already-downloaded files")

  args = parser.parse_args()

  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

  ensure_nltk_data()

  SOURCES_DIR.mkdir(parents=True, exist_ok=True)
  DUMPS_DIR.mkdir(parents=True, exist_ok=True)

  sources = ["simplewiki", "enwiki_curated"] if args.source == "all" else [args.source]

  processors = {
    "simplewiki": process_simplewiki,
    "enwiki_curated": process_enwiki_curated,
  }

  for src in sources:
    path = processors[src](resume=args.resume)
    validate_output(path)


if __name__ == "__main__":
  main()
