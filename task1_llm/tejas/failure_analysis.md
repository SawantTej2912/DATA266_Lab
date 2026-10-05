# Task 1 — Sequence Model Failure Analysis (Tejas)

**Model:** character-level GPT built from scratch — 6 layers, 6 heads, 384-dim embeddings, 256-character context, pre-LN, tied embeddings (10.78M parameters). Best checkpoint: epoch 10 (val CE 0.5586, perplexity 1.75, 0.81 bits/char, 82.1% next-character accuracy).

**Generation setup:** 300 new characters per prompt, using greedy decoding (argmax) and temperature sampling (T = 0.8). Candidate failures were taken from `outputs/failure_candidates.json`, which ranks generations by repeated-4-gram rate.

**Diversity metrics (word level, continuation only):**

| Decoding | Repeated-4-gram rate | Distinct-2 |
|---|---|---|
| Greedy | 0.080 | 0.628 |
| Sampled (T = 0.8) | 0.004 | 0.693 |

Greedy decoding repeats itself about 20× more often than sampling, which is the pattern behind Cases 1 and 2.

---

## Case 1 — Repetition / looping
**Decoding:** greedy · **Prompt:** "One day, Tom" · **Repeated-4-gram rate:** 0.213 (highest of all samples)

> "One day, Tom wanted to play with his toys and run around outside. He saw a big tree with many leaves. Tom wanted to climb the tree, but he was too small. He tried to climb the tree, but he could not. Tom was sad. Tom saw a big tree…"

**Failure type:** repetition loop.

**Observation:** The story makes almost no progress. "climb the tree" appears twice in consecutive sentences, the plot stalls at "he could not… Tom was sad", and the model then restarts the same scene with "Tom saw a big tree", re-describing what it had already said instead of continuing the narrative.

**Likely cause:** Greedy decoding always selects the single most probable next character. Phrases like "climb the tree" and "a big tree" are very frequent in TinyStories, so once they appear the most probable continuation is to produce them again, and the loop reinforces itself. Nothing in greedy decoding penalizes repetition, which is why its repeated-4-gram rate (0.080) is far higher than sampling's (0.004).

---

## Case 2 — Story reset to a memorized template
**Decoding:** greedy · **Prompt:** "The little dog"

> "The little dog was so happy to have a new friend. They played together all day long. The end.
>
> Once upon a time, there was a little girl named Lily. She loved to play outside in the sunshine. One day, she saw a big box in the garden."

**Failure type:** loss of coherence / mode collapse to the most frequent training story.

**Observation:** The model ends the story about the dog after two sentences ("The end.") and starts an unrelated new story. That new story is word-for-word identical to the greedy output for the separate prompt "Once upon a time" ("a little girl named Lily… play outside in the sunshine… a big box in the garden"). Across prompts, greedy generations keep converging on the same Lily / box / mommy template.

**Likely cause:** Two effects combine. (1) After "The end.", the most probable continuation in TinyStories is a new story's opening, and greedy decoding always takes the single most likely path, which is the dataset's most common opening ("Once upon a time, there was a little girl named Lily"). (2) Generating 300 characters exceeds the 256-character context window, so the original prompt eventually falls out of view and can no longer steer the generation. The model falls back to its strongest prior, a form of mode collapse.

---

## Case 3 — Semantic hallucination (fluent but meaningless)
**Decoding:** temperature sampling, T = 0.8 · **Prompt:** "In the forest"

> "In the forest was a bright, sunny barn. The barn was full of lots of animals, flowers and flowers. At night, the barn went for many long walks in the meadow. But then, something bad happened. A man came to say he had a spell."

**Failure type:** hallucination / semantic incoherence. The grammar is correct, but the meaning is impossible.

**Observation:** The setting contradicts the prompt (a barn "in the forest"), an inanimate barn performs actions ("went for many long walks"), and the list is degenerate ("flowers and flowers"). The same pattern appears with the prompt "Once upon a time" at T = 0.8: "The truck had many lots of firemen… The truck driver was very naughty. He said he did not take medicine… the truck got too hurt". Each sentence is locally well formed, but together they violate world knowledge (animacy, quantities, causality).

**Likely cause:** The model has learned spelling, syntax, and short-range word co-occurrence very well (82% next-character accuracy, 0.81 bits/char), but it has no explicit representation of meaning. Temperature sampling adds diversity, which lets the model combine fragments that are each plausible locally ("the barn was full of…", "went for many long walks") into combinations it never saw together and that make no sense globally. A 10.8M-parameter character-level model has limited capacity to learn longer-range semantic consistency.

---

## Additional observation
Greedy, prompt "Lily was sad because" → "she loved her mommy and daddy. She wanted to help her mommy and daddy. She asked her mommy if she could help her mommy and daddy…". This is a causal non-sequitur (being sad *because* she loved her parents) combined with phrase repetition ("mommy and daddy" four times). It combines the failure modes of Cases 1 and 3.

## Summary

| Case | Decoding | Failure type | Main cause |
|---|---|---|---|
| 1 | Greedy | Repetition loop | Argmax reinforces frequent phrases; no repetition penalty |
| 2 | Greedy | Story reset / mode collapse | Prompt exceeds the 256-char context; "The end." leads to the most frequent opening |
| 3 | Sampled (T = 0.8) | Semantic hallucination | Strong local syntax, weak world knowledge; sampling recombines fragments |

## What would help
- **Against repetition:** a repetition penalty, or top-k / nucleus (top-p) sampling instead of pure greedy decoding.
- **Against story resets:** a longer context window (512+) or keeping the prompt in view while generating; stopping generation at "The end.".
- **Against hallucination:** more capacity (a larger model, more training), or subword/word tokenization so each context position carries more meaning; a lower temperature (e.g. 0.6) trades some diversity for coherence.

*Note:* the "possible_nonwords" flags in `failure_candidates.json` (e.g. "cl", "ang", "op") are words cut off by the 220-character snippet length, not spelling errors in the generations.