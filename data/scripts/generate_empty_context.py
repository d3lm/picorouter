"""Generate empty-context training examples that bridge refusal and tool calling.

This does not require any LLM calls, it's just pure cross-pairing, so it's free and fast.

Produces three flavors:
  - Tool-call: empty context, question matches a tool in the subset → <|tool_call|>
  - Refusal (wrong tools): empty context, matching tool NOT in subset → <|refuse|>
  - Refusal (no tools): empty context, tools=[] → <|refuse|>

Usage:
  uv run python -m data.scripts.generate_empty_context \
    --output data/synthetic/empty_context.jsonl \
    --limit 5000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
TOOL_SCHEMAS_PATH = DATA_DIR / "tool_schemas.json"

REFUSAL_PHRASE = "I don't have enough information in the provided context"

REFUSAL_RESPONSES = [
  f"<|refuse|>{REFUSAL_PHRASE} to answer that question.",
  f"<|refuse|>{REFUSAL_PHRASE} to answer that.",
  f"<|refuse|>{REFUSAL_PHRASE} to answer this question accurately.",
  f"<|refuse|>{REFUSAL_PHRASE}. The question is about a topic not covered here.",
  f"<|refuse|>{REFUSAL_PHRASE} to provide an answer to that question.",
]

TOOL_QUESTION_TEMPLATES: list[tuple[str, str, dict]] = [
  # calculator
  ("What is 15 times 23?", "calculator", {"expression": "15 * 23"}),
  ("Calculate 1024 divided by 16", "calculator", {"expression": "1024 / 16"}),
  ("What's the square root of 256?", "calculator", {"expression": "sqrt(256)"}),
  ("How much is 19.99 plus 7.5% tax?", "calculator", {"expression": "19.99 * 1.075"}),
  ("What is 2 to the power of 10?", "calculator", {"expression": "2 ** 10"}),
  ("Compute 3.14 times 5 squared", "calculator", {"expression": "3.14 * 5 ** 2"}),
  ("What's 45% of 200?", "calculator", {"expression": "0.45 * 200"}),
  ("How much is 500 minus 137?", "calculator", {"expression": "500 - 137"}),
  ("What's 72 divided by 9?", "calculator", {"expression": "72 / 9"}),
  ("Calculate 1000 times 0.08", "calculator", {"expression": "1000 * 0.08"}),
  # web_search
  ("What's the latest news about SpaceX?", "web_search", {"query": "latest SpaceX news"}),
  ("Who won the Oscar for best picture this year?", "web_search", {"query": "Oscar best picture winner this year"}),
  ("What is the current population of Tokyo?", "web_search", {"query": "current population Tokyo"}),
  (
    "Find information about the James Webb telescope discoveries",
    "web_search",
    {"query": "James Webb telescope recent discoveries"},
  ),
  ("What are the top movies right now?", "web_search", {"query": "top movies currently playing"}),
  ("Search for the latest AI research papers", "web_search", {"query": "latest AI research papers"}),
  ("Who is the CEO of OpenAI?", "web_search", {"query": "CEO of OpenAI"}),
  ("What happened in the news today?", "web_search", {"query": "today's news headlines"}),
  ("Find reviews for the new iPhone", "web_search", {"query": "new iPhone reviews"}),
  ("What are the best restaurants in Paris?", "web_search", {"query": "best restaurants Paris"}),
  # get_current_date
  ("What day is it today?", "get_current_date", {}),
  ("What's today's date?", "get_current_date", {}),
  ("What day of the week is it?", "get_current_date", {}),
  ("Can you tell me the current date?", "get_current_date", {}),
  ("What is the date right now?", "get_current_date", {}),
  ("Is it a weekday or weekend today?", "get_current_date", {}),
  ("What month are we in?", "get_current_date", {}),
  ("Tell me the current date and time", "get_current_date", {}),
  # get_weather
  ("What's the weather like in New York?", "get_weather", {"location": "New York"}),
  ("Is it going to rain in London today?", "get_weather", {"location": "London"}),
  ("What's the temperature in Tokyo right now?", "get_weather", {"location": "Tokyo"}),
  ("How's the weather in San Francisco?", "get_weather", {"location": "San Francisco"}),
  ("Do I need an umbrella in Seattle?", "get_weather", {"location": "Seattle"}),
  ("What's the forecast for Chicago?", "get_weather", {"location": "Chicago"}),
  ("Is it cold outside in Moscow?", "get_weather", {"location": "Moscow"}),
  ("What's the weather in Sydney, Australia?", "get_weather", {"location": "Sydney, Australia"}),
  ("How hot is it in Dubai?", "get_weather", {"location": "Dubai"}),
  ("What are the current conditions in Denver?", "get_weather", {"location": "Denver"}),
  # unit_converter
  ("Convert 5 miles to kilometers", "unit_converter", {"value": 5, "from_unit": "miles", "to_unit": "kilometers"}),
  (
    "How many kilograms is 150 pounds?",
    "unit_converter",
    {"value": 150, "from_unit": "pounds", "to_unit": "kilograms"},
  ),
  (
    "What is 100 degrees Fahrenheit in Celsius?",
    "unit_converter",
    {"value": 100, "from_unit": "Fahrenheit", "to_unit": "Celsius"},
  ),
  ("Convert 2 liters to gallons", "unit_converter", {"value": 2, "from_unit": "liters", "to_unit": "gallons"}),
  ("How many feet are in 10 meters?", "unit_converter", {"value": 10, "from_unit": "meters", "to_unit": "feet"}),
  ("Convert 500 grams to ounces", "unit_converter", {"value": 500, "from_unit": "grams", "to_unit": "ounces"}),
  (
    "What's 30 centimeters in inches?",
    "unit_converter",
    {"value": 30, "from_unit": "centimeters", "to_unit": "inches"},
  ),
  (
    "How many cups is 500 milliliters?",
    "unit_converter",
    {"value": 500, "from_unit": "milliliters", "to_unit": "cups"},
  ),
  # define_word
  ("What does 'ephemeral' mean?", "define_word", {"word": "ephemeral"}),
  ("Define the word 'ubiquitous'", "define_word", {"word": "ubiquitous"}),
  ("What is the meaning of 'pragmatic'?", "define_word", {"word": "pragmatic"}),
  ("Can you define 'serendipity'?", "define_word", {"word": "serendipity"}),
  ("What does 'juxtaposition' mean?", "define_word", {"word": "juxtaposition"}),
  ("Define 'ambiguous'", "define_word", {"word": "ambiguous"}),
  ("What's the definition of 'resilient'?", "define_word", {"word": "resilient"}),
  ("What does 'perfunctory' mean?", "define_word", {"word": "perfunctory"}),
  # translate
  (
    "Translate 'hello, how are you?' to Spanish",
    "translate",
    {"text": "hello, how are you?", "from_lang": "English", "to_lang": "Spanish"},
  ),
  (
    "How do you say 'thank you' in French?",
    "translate",
    {"text": "thank you", "from_lang": "English", "to_lang": "French"},
  ),
  (
    "Translate 'good morning' to Japanese",
    "translate",
    {"text": "good morning", "from_lang": "English", "to_lang": "Japanese"},
  ),
  (
    "What is 'I love you' in Italian?",
    "translate",
    {"text": "I love you", "from_lang": "English", "to_lang": "Italian"},
  ),
  (
    "Translate 'where is the train station?' to German",
    "translate",
    {"text": "where is the train station?", "from_lang": "English", "to_lang": "German"},
  ),
  (
    "How do you say 'goodbye' in Portuguese?",
    "translate",
    {"text": "goodbye", "from_lang": "English", "to_lang": "Portuguese"},
  ),
  (
    "Translate 'the weather is nice today' to Mandarin",
    "translate",
    {"text": "the weather is nice today", "from_lang": "English", "to_lang": "Mandarin"},
  ),
  (
    "What's 'help me please' in Korean?",
    "translate",
    {"text": "help me please", "from_lang": "English", "to_lang": "Korean"},
  ),
  # set_timer
  ("Set a timer for 5 minutes", "set_timer", {"duration_seconds": 300, "label": "5 minute timer"}),
  ("Set a 30-second timer", "set_timer", {"duration_seconds": 30, "label": "30 second timer"}),
  ("Start a 10 minute timer for my pasta", "set_timer", {"duration_seconds": 600, "label": "pasta"}),
  ("Set a timer for 2 minutes", "set_timer", {"duration_seconds": 120, "label": "2 minute timer"}),
  ("Timer for 1 hour please", "set_timer", {"duration_seconds": 3600, "label": "1 hour timer"}),
  ("Set a 15 minute break timer", "set_timer", {"duration_seconds": 900, "label": "break"}),
  ("Start a 45-second timer", "set_timer", {"duration_seconds": 45, "label": "45 second timer"}),
  ("Set a pomodoro timer for 25 minutes", "set_timer", {"duration_seconds": 1500, "label": "pomodoro"}),
  # run_code
  ("Run this Python code: print('hello world')", "run_code", {"code": "print('hello world')"}),
  ("Execute: [x**2 for x in range(10)]", "run_code", {"code": "print([x**2 for x in range(10)])"}),
  ("Can you run: import sys; print(sys.version)", "run_code", {"code": "import sys; print(sys.version)"}),
  ("Run this code to check if 17 is prime", "run_code", {"code": "print(all(17 % i != 0 for i in range(2, 17)))"}),
  ("Execute Python: sorted([3, 1, 4, 1, 5, 9])", "run_code", {"code": "print(sorted([3, 1, 4, 1, 5, 9]))"}),
  (
    "Run: len('supercalifragilisticexpialidocious')",
    "run_code",
    {"code": "print(len('supercalifragilisticexpialidocious'))"},
  ),
  ("Execute this snippet: sum(range(1, 101))", "run_code", {"code": "print(sum(range(1, 101)))"}),
  # summarize
  (
    "Summarize the following article for me in 3 sentences",
    "summarize",
    {"text": "Please provide the article text.", "max_sentences": 3},
  ),
  (
    "Give me a brief summary of this text",
    "summarize",
    {"text": "Please provide the text to summarize.", "max_sentences": 2},
  ),
  (
    "Can you summarize this document in 5 sentences?",
    "summarize",
    {"text": "Please provide the document text.", "max_sentences": 5},
  ),
  (
    "I need a quick summary of this report",
    "summarize",
    {"text": "Please provide the report text.", "max_sentences": 3},
  ),
  ("Summarize this in one sentence", "summarize", {"text": "Please provide the text.", "max_sentences": 1}),
  # get_stock_price
  ("What's Apple's stock price?", "get_stock_price", {"symbol": "AAPL"}),
  ("How much is Tesla stock right now?", "get_stock_price", {"symbol": "TSLA"}),
  ("What's MSFT trading at?", "get_stock_price", {"symbol": "MSFT"}),
  ("Check the stock price for Amazon", "get_stock_price", {"symbol": "AMZN"}),
  ("What's Google's current stock price?", "get_stock_price", {"symbol": "GOOGL"}),
  ("How much is NVIDIA stock?", "get_stock_price", {"symbol": "NVDA"}),
  ("What's the price of META stock?", "get_stock_price", {"symbol": "META"}),
  ("Check Netflix stock price", "get_stock_price", {"symbol": "NFLX"}),
  # send_email
  (
    "Send an email to john@example.com about the meeting",
    "send_email",
    {"to": "john@example.com", "subject": "Meeting", "body": "Hi John, I wanted to discuss the meeting details."},
  ),
  (
    "Email sarah@company.com that I'll be late",
    "send_email",
    {
      "to": "sarah@company.com",
      "subject": "Running late",
      "body": "Hi Sarah, I'll be running a few minutes late today.",
    },
  ),
  (
    "Send a message to boss@work.com about the project update",
    "send_email",
    {"to": "boss@work.com", "subject": "Project Update", "body": "Here's the latest update on the project."},
  ),
  (
    "Email team@company.com about tomorrow's deadline",
    "send_email",
    {"to": "team@company.com", "subject": "Tomorrow's Deadline", "body": "Reminder: the project deadline is tomorrow."},
  ),
  (
    "Send an email to support@service.com asking about my order",
    "send_email",
    {
      "to": "support@service.com",
      "subject": "Order Status",
      "body": "I'd like to check the status of my recent order.",
    },
  ),
  # create_reminder
  (
    "Remind me to call the dentist tomorrow at 9am",
    "create_reminder",
    {"message": "Call the dentist", "datetime": "2025-03-15 09:00"},
  ),
  (
    "Set a reminder to pick up groceries at 5pm",
    "create_reminder",
    {"message": "Pick up groceries", "datetime": "2025-03-14 17:00"},
  ),
  (
    "Remind me about the team meeting at 2pm",
    "create_reminder",
    {"message": "Team meeting", "datetime": "2025-03-14 14:00"},
  ),
  (
    "Create a reminder to take medicine at 8am",
    "create_reminder",
    {"message": "Take medicine", "datetime": "2025-03-15 08:00"},
  ),
  (
    "Remind me to submit the report by Friday",
    "create_reminder",
    {"message": "Submit report", "datetime": "2025-03-21 09:00"},
  ),
  (
    "Set a reminder to water the plants at noon",
    "create_reminder",
    {"message": "Water plants", "datetime": "2025-03-14 12:00"},
  ),
  # get_directions
  (
    "How do I get from Times Square to Central Park?",
    "get_directions",
    {"origin": "Times Square, New York", "destination": "Central Park, New York", "mode": "walking"},
  ),
  (
    "Get driving directions from LA to San Francisco",
    "get_directions",
    {"origin": "Los Angeles, CA", "destination": "San Francisco, CA", "mode": "driving"},
  ),
  (
    "How do I get to the airport from downtown Chicago?",
    "get_directions",
    {"origin": "Downtown Chicago", "destination": "O'Hare Airport, Chicago", "mode": "driving"},
  ),
  (
    "Directions from Boston to New York by train",
    "get_directions",
    {"origin": "Boston, MA", "destination": "New York, NY", "mode": "transit"},
  ),
  (
    "Walking directions from the Louvre to the Eiffel Tower",
    "get_directions",
    {"origin": "Louvre Museum, Paris", "destination": "Eiffel Tower, Paris", "mode": "walking"},
  ),
  (
    "How do I drive from Seattle to Portland?",
    "get_directions",
    {"origin": "Seattle, WA", "destination": "Portland, OR", "mode": "driving"},
  ),
  # lookup_contact
  ("Find John Smith's contact info", "lookup_contact", {"name": "John Smith"}),
  ("Look up Sarah Johnson's phone number", "lookup_contact", {"name": "Sarah Johnson"}),
  ("Do you have contact info for Mike Chen?", "lookup_contact", {"name": "Mike Chen"}),
  ("Find the email for Dr. Emily Brown", "lookup_contact", {"name": "Dr. Emily Brown"}),
  ("Look up contact details for Alex Williams", "lookup_contact", {"name": "Alex Williams"}),
  ("Can you find Lisa Martinez in my contacts?", "lookup_contact", {"name": "Lisa Martinez"}),
  ("What's David Kim's phone number?", "lookup_contact", {"name": "David Kim"}),
]

GENERIC_REFUSAL_QUESTIONS = [
  "What's the meaning of life?",
  "Tell me a joke",
  "What should I have for dinner?",
  "How do I get better at chess?",
  "What's the best programming language?",
  "Can you write me a poem?",
  "What's your favorite color?",
  "How do I learn to play guitar?",
  "What's a good book to read?",
  "How do I make friends as an adult?",
  "What career should I pursue?",
  "How do I fix a leaky faucet?",
  "What's the best way to lose weight?",
  "How do I start a business?",
  "What should I name my cat?",
  "How do I improve my memory?",
  "What's the fastest animal on Earth?",
  "How tall is Mount Everest?",
  "Who invented the telephone?",
  "What year did World War II end?",
  "How many planets are in the solar system?",
  "What's the speed of light?",
  "How do magnets work?",
  "What causes rainbows?",
  "Why is the sky blue?",
]


def load_tool_schemas() -> list[dict]:
  with open(TOOL_SCHEMAS_PATH, encoding="utf-8") as f:
    return json.load(f)


def _build_schema_lookup(schemas: list[dict]) -> dict[str, dict]:
  return {s["name"]: s for s in schemas}


def generate_examples(
  tool_schemas: list[dict],
  target_count: int,
  seed: int = 42,
  min_tools: int = 2,
  max_tools: int = 5,
) -> list[dict]:
  rng = random.Random(seed)
  schema_lookup = _build_schema_lookup(tool_schemas)
  schema_names = list(schema_lookup.keys())

  templates_by_tool: dict[str, list[tuple[str, dict]]] = {}

  for question, tool_name, args in TOOL_QUESTION_TEMPLATES:
    if tool_name in schema_lookup:
      templates_by_tool.setdefault(tool_name, []).append((question, args))

  n_tool_call = int(target_count * 0.40)
  n_refusal_wrong_tools = int(target_count * 0.30)
  n_refusal_no_tools = target_count - n_tool_call - n_refusal_wrong_tools

  examples: list[dict] = []
  refusal_idx = 0

  for _ in range(n_tool_call):
    tool_name = rng.choice(list(templates_by_tool.keys()))
    question, args = rng.choice(templates_by_tool[tool_name])

    other_tools = [n for n in schema_names if n != tool_name]
    n_extra = rng.randint(min_tools - 1, min(max_tools - 1, len(other_tools)))
    extra = rng.sample(other_tools, n_extra)

    subset_names = [tool_name, *extra]

    rng.shuffle(subset_names)

    tool_subset = [schema_lookup[n] for n in subset_names]

    examples.append(
      {
        "context": "",
        "tools": tool_subset,
        "conversation": [
          {"role": "user", "content": question},
          {"role": "assistant", "content": f"<|tool_call|>{json.dumps({'name': tool_name, 'arguments': args})}"},
        ],
        "source": "synthetic-empty-context",
      }
    )

  for _ in range(n_refusal_wrong_tools):
    tool_name = rng.choice(list(templates_by_tool.keys()))
    question, _args = rng.choice(templates_by_tool[tool_name])

    other_tools = [n for n in schema_names if n != tool_name]
    n_shown = rng.randint(min_tools, min(max_tools, len(other_tools)))
    shown = rng.sample(other_tools, n_shown)
    tool_subset = [schema_lookup[n] for n in shown]

    examples.append(
      {
        "context": "",
        "tools": tool_subset,
        "conversation": [
          {"role": "user", "content": question},
          {"role": "assistant", "content": REFUSAL_RESPONSES[refusal_idx % len(REFUSAL_RESPONSES)]},
        ],
        "source": "synthetic-empty-context",
      }
    )

    refusal_idx += 1

  all_questions = list(GENERIC_REFUSAL_QUESTIONS)

  for question, _tool, _args in TOOL_QUESTION_TEMPLATES:
    all_questions.append(question)

  for _ in range(n_refusal_no_tools):
    question = rng.choice(all_questions)

    examples.append(
      {
        "context": "",
        "tools": [],
        "conversation": [
          {"role": "user", "content": question},
          {"role": "assistant", "content": REFUSAL_RESPONSES[refusal_idx % len(REFUSAL_RESPONSES)]},
        ],
        "source": "synthetic-empty-context",
      }
    )

    refusal_idx += 1

  rng.shuffle(examples)

  return examples


def main():
  parser = argparse.ArgumentParser(
    description="Generate empty-context training examples (no LLM calls)",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )

  parser.add_argument("--output", required=True, type=Path, help="Output JSONL file")
  parser.add_argument("--limit", type=int, default=5000, help="Number of examples to generate (default: 5000)")
  parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
  parser.add_argument("--min-tools", type=int, default=2, help="Min tools in subset (default: 2)")
  parser.add_argument("--max-tools", type=int, default=5, help="Max tools in subset (default: 5)")

  args = parser.parse_args()

  logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
  )

  t0 = time.monotonic()

  log.info("Loading tool schemas from %s...", TOOL_SCHEMAS_PATH)

  tool_schemas = load_tool_schemas()

  log.info("Loaded %d tool schemas", len(tool_schemas))

  templates_with_match = sum(
    1 for _, tool_name, _ in TOOL_QUESTION_TEMPLATES if tool_name in {s["name"] for s in tool_schemas}
  )

  log.info(
    "%d question templates (%d matching current schemas), %d generic refusal questions",
    len(TOOL_QUESTION_TEMPLATES),
    templates_with_match,
    len(GENERIC_REFUSAL_QUESTIONS),
  )

  examples = generate_examples(
    tool_schemas,
    target_count=args.limit,
    seed=args.seed,
    min_tools=args.min_tools,
    max_tools=args.max_tools,
  )

  n_tool_call = sum(1 for ex in examples if "<|tool_call|>" in ex["conversation"][-1]["content"])
  n_refusal = len(examples) - n_tool_call
  n_with_tools = sum(1 for ex in examples if ex["tools"])
  n_no_tools = len(examples) - n_with_tools

  args.output.parent.mkdir(parents=True, exist_ok=True)

  with open(args.output, "w", encoding="utf-8") as file:
    for ex in examples:
      file.write(json.dumps(ex, ensure_ascii=False) + "\n")

  elapsed = time.monotonic() - t0

  log.info("Done in %.1fs — wrote %d examples to %s", elapsed, len(examples), args.output)
  log.info("  Tool-call: %d (%.0f%%)", n_tool_call, n_tool_call / len(examples) * 100)
  log.info("  Refusal:   %d (%.0f%%)", n_refusal, n_refusal / len(examples) * 100)
  log.info("    with tools (wrong ones): %d", n_with_tools - n_tool_call)
  log.info("    no tools:                %d", n_no_tools)


if __name__ == "__main__":
  main()
