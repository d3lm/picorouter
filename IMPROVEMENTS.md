# PicoRouter Improvement Ideas

> Current direction: grounded QA only (answer from context or refuse).
> Tool-calling has been removed from the active training/eval/inference pipeline.

## 1. Prioritize hallucination suppression in training mix

**Problem:** Answerable RC examples still dominate, so the model can over-answer.

**Fixes:**

### 1a. Increase refusal share

Upsample `synthetic-offtopic-refusal` and SQuAD2 unanswerable examples so refusal is a first-class behavior rather than a tail behavior.

### 1b. Keep answerable coverage broad

Preserve diversity from SQuAD2, DROP, and CoQA so refusal gains do not collapse extractive quality.

## 2. Retrieval should return empty on no match

**Problem:** In `playground/src/retrieval.ts`, when no result is found, fallback returns top documents with score `0`, which still feeds unrelated context.

**Fix:** Return `[]` when ranking is empty and enforce a positive `minScore` threshold.

## 3. Empty retrieval should send empty context

**Problem:** In `App.tsx`, fallback currently sends raw context when retrieval is empty.

**Fix:** Send empty context (`''`) on no retrieval match to force answer-or-refuse behavior based on real evidence.

## 4. Eval should track grounded QA only

**Problem:** Composite scoring can hide refusal failures when hallucination is the top objective.

**Fix:** Keep headline metrics as:

- extractive F1/EM
- correct refusal rate
- false refusal rate
- hallucination rate (primary)

Weight composite score toward low hallucination first, then low false refusals.

## 5. Add near-topic adversarial refusals

**Problem:** Random off-topic negatives are easier than realistic near-miss cases.

**Fix:** Generate additional adversarial examples where context and question share domain/entities but do not contain the answer span. This improves refusal calibration on hard cases.
