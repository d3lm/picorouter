# PicoRouter improvement plan

## 1. Off-topic refusal — the model never says "I don't know"

**Problem:** The model hallucinates answers from context even when the question is completely unrelated. SQuAD 2.0 unanswerables are same-topic ("What year was the tower demolished?" about an Eiffel Tower passage). The model has never seen a cross-domain mismatch paired with a refusal.

**Fix:** Add a `--mode offtopic-refusal` to `generate_synthetic.py` that pairs each passage with questions from entirely different domains.

- For each passage, randomly sample 3-5 questions from _other_ passages (different source, different topic).
- Target response: `<|refuse|>I don't have enough information in the provided context to answer that question.`
- No LLM call needed — this is pure cross-pairing, so it's free and fast.
- Also mix in some generic out-of-domain questions that would never appear in any passage ("What's the capital of Mars?", "How do I install Python?") to teach the model the general pattern.
- Generate ~15-20K examples to match the scale of other sources.

**Where to change:** New function in `data/scripts/generate_synthetic.py` or a standalone script. Output format is the same unified JSONL:

```json
{
  "context": "The Eiffel Tower was completed in 1889...",
  "tools": [],
  "conversation": [
    { "role": "user", "content": "When was ChatGPT released?" },
    {
      "role": "assistant",
      "content": "<|refuse|>I don't have enough information in the provided context to answer that question."
    }
  ],
  "source": "synthetic-offtopic-refusal"
}
```

**Also:** Upsample existing SQuAD 2.0 refusals 2-3x during tokenization to rebalance the dataset.

## 2. Tool-calling is underrepresented

**Problem:** The synthetic tools mode generates 4 tool-call + 3 context-answer per passage, but the tool examples are a small fraction of the total dataset. SQuAD 2.0, CoQA, and DROP all have `"tools": []`, so the model overwhelmingly learns to answer from context without tools.

**Fixes (do all three):**

### 2a. Generate more tool-calling data

- Run `generate_synthetic --mode tools` with `--limit` increased to cover more passages.
- Add more tool schemas to `data/tool_schemas.json` — 8 tools is low. Consider adding: `summarize`, `create_reminder`, `send_email`, `lookup_contact`, `run_code`, `get_stock_price`. More tool variety forces the model to learn tool _selection_, not just tool _existence_.

### 2b. Vary the tool schema subsets per example

Right now every tools-mode example sees all 8 tools. In production the model will see different subsets. During generation, randomly sample 3-6 tools from the full set per passage so the model learns to route based on what's available, not memorize a fixed tool list.

### 2c. Upsample tool examples during training

In `merge_and_tokenize.py` `step_process`, before shuffling, repeat tool-source examples 2-3x so the model sees them proportionally to RC examples. Quick implementation: after loading from `filtered.jsonl`, duplicate rows where `source == "synthetic-tools"`.

## 3. Retrieval sends context even when nothing matches

**Problem:** In `playground/src/retrieval.ts`, when MiniSearch finds zero matches, it falls back to returning all documents with `score: 0`:

```typescript
// retrieval.ts line 106-107
if (ranked.length === 0) {
  return normalizedDocs.slice(0, topK).map((document, index) => ({ document, index, score: 0 }));
}
```

Then in `App.tsx`, the model always gets context regardless:

```typescript
// App.tsx line 55
const finalContext = retrievedContext || context;
```

**Fix — two changes:**

### 3a. `retrieval.ts` — return empty when nothing matches

Change the fallback to return an empty array instead of all documents:

```typescript
if (ranked.length === 0) {
  return [];
}
```

The `minScore` option already exists in `SearchOptions` but defaults to `0`. Set a meaningful default like `1.0` so only genuinely relevant passages survive.

### 3b. `App.tsx` — handle empty retrieval gracefully

When retrieval returns nothing, either:

- **(Option A — quick fix):** Send empty context to the model. The prompt becomes `<|context|><|tools|><|user|>...<|assistant|>`, and the model should refuse. This requires the model to have seen empty-context refusals during training (see item 4 below).
- **(Option B — better UX):** Show a message like "No relevant passages found" and still send the question to the model with minimal/empty context, so the model can attempt a tool-call or refuse.

Change in `handleAsk`:

```typescript
const retrievedContext = retrievalResults.map((result, i) => `[${i + 1}] ${result.document}`).join('\n\n');

// If retrieval found nothing relevant, send empty context
const finalContext = retrievedContext || '';
```

## 4. Train on empty/minimal context scenarios

**Problem:** The model has never seen `<|context|>` followed immediately by `<|tools|>` (empty context). It doesn't know how to behave in that situation.

**Fix:** Add ~2-5K training examples with empty or minimal context where the correct response is a refusal or tool-call:

```json
{
  "context": "",
  "tools": [],
  "conversation": [
    { "role": "user", "content": "What's the weather in Tokyo?" },
    {
      "role": "assistant",
      "content": "<|refuse|>I don't have enough information in the provided context to answer that question."
    }
  ],
  "source": "synthetic-empty-context"
}
```

And with tools available:

```json
{
  "context": "",
  "tools": [{"name": "get_weather", ...}],
  "conversation": [
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {"role": "assistant", "content": "<|tool_call|>{\"name\": \"get_weather\", \"arguments\": {\"location\": \"Tokyo\"}}"}
  ],
  "source": "synthetic-empty-context"
}
```

## 5. Evaluation gaps

**Problem:** The eval suite (`evaluate.py`) already has a hallucination probe (`build_adversarial_examples`) that does cross-pairing, but this signal isn't fed back into training. Also, the composite score weighs extractive F1 at 0.4 but hallucination at only 0.1:

```python
# evaluate.py line 702
return 0.4 * f1 + 0.3 * tool_acc + 0.2 * refusal + 0.1 * (1 - halluc)
```

**Fixes:**

- **Reweight composite score:** Bump hallucination to 0.2 and reduce F1 to 0.3. Hallucination is the user-facing failure mode you care about most.
- **Run eval before the next training round** to get baseline numbers, then track improvement.
- **Add an off-topic refusal eval category** in `load_test_examples` — right now, refusal examples only come from same-topic unanswerables. After adding off-topic refusal training data, the test split will naturally include them.

## 6. Retrieval-score signal in the model input

**Problem:** Even with the retrieval fallback fix (item 3), the model has no way to know _how confident_ retrieval was. It either gets context or doesn't — there's no gradient. A borderline retrieval hit (score 1.1 vs. threshold 1.0) looks identical to a perfect match (score 15.0).

**Fix:** Inject a retrieval confidence signal into the prompt so the model can learn to weigh context accordingly:

- **Option A — score token:** Prepend a discretized score bucket to the context, e.g. `<|retrieval_high|>`, `<|retrieval_low|>`, `<|retrieval_none|>`. Train with these tokens so the model learns that `<|retrieval_low|>` context is unreliable.

- **Option B — score threshold in tokenization:** During `merge_and_tokenize.py`, add a synthetic retrieval score field to training examples. For off-topic/refusal examples, set score to 0; for normal RC examples, set score high. The model implicitly learns the correlation.

This requires adding the signal to training data generation and to the playground inference path.

## 7. Inference improvements (playground)

### 7a. Include tool schemas in the prompt

The playground currently sends `<|tools|>` with nothing after it. The model was trained with tool schemas encoded after `<|tools|>`. Either:

- Include the tool schemas from `data/tool_schemas.json` in the playground so the model can exercise tool-call routing.
- Or explicitly train on the empty-tools pattern (item 4).

### 7b. Show retrieval scores in the UI

Display the MiniSearch scores next to retrieved passages so the user can see when retrieval is low-confidence. This helps debug whether failures are retrieval problems vs. model problems.

## Priority order

| #   | Item                                       | Effort                                      | Impact                             |
| --- | ------------------------------------------ | ------------------------------------------- | ---------------------------------- |
| 1   | Off-topic refusal data (item 1)            | Low — no API cost, cross-pair existing data | Fixes the main hallucination issue |
| 2   | Retrieval fallback fix (item 3)            | Low — ~10 lines of code                     | Stops feeding irrelevant context   |
| 3   | Upsample tool + refusal (item 2c)          | Low — change `step_process`                 | Rebalances training signal         |
| 4   | Empty context training (item 4)            | Low — synthetic, no API cost                | Model handles edge case            |
| 5   | More tool schemas + variety (items 2a, 2b) | Medium — needs API calls                    | Better tool routing                |
| 6   | Include tools in playground (item 7a)      | Low — load JSON, encode in prompt           | Model uses its training            |
| 7   | Eval reweighting + baseline (item 5)       | Low — tweak constants, run eval             | Measures improvement               |
| 8   | Retrieval-score signal (item 6)            | Medium — new tokens, training + inference   | Model knows retrieval confidence   |
