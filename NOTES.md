- Train as-is and evaluate. See if the model actually struggles with multiturn before solving a problem you might not have.

- Upsample multiturn during training. Repeat the 1,205 multiturn examples ~5-10x in the training data so the model sees them roughly as often as tool examples. This is cheap and common practice — no extra generation cost.

- Generate more multiturn. Resume the multiturn generation with --limit 5000 to get closer to a 2:1 or 3:1 ratio. But at ~1 example per passage, you'd need ~4,000 more passages to get to ~5K multiturn examples, which costs more API calls.
