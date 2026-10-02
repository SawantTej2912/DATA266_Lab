# Task 1 — sequence model failure analysis

Model: `gpt_char_tinystories_v1`, checkpoint
`task1_llm/shriram_dundigalla/checkpoints/gpt_char_tinystories_v1_20260928T204206Z.pt`.
All snippets are unedited from
`outputs/generations_gpt_char_tinystories_v1_20260928T204206Z.txt`, produced with the
prompt `"Once upon a time"` and 400 new characters. Per-temperature diversity numbers
are in `outputs/generation_metrics_*.csv`.

I generated across five temperatures (0.0, 0.5, 0.8, 1.0, 1.2) rather than sampling
once, because the failure mode is not a fixed property of the model — it changes with
the decoding setting, and two of the three cases below only appear at one end of the
sweep.

---

## Case 1 — Verbatim repetition under greedy decoding

**Temperature 0.0 (greedy).**

> Once upon a time, there was a little girl named Lily. She loved to play outside in
> the sunshine. One day, she went to the park with her mommy. She saw a big box of
> stars and a big box. She wanted to see what was inside. **She wanted to see what was
> inside.** She took a big bite and started to cry.

**Failure type:** repetition (degenerate looping).

**Observation.** The sentence "She wanted to see what was inside." is emitted twice in
a row, character for character. The same snippet also contains a near-repetition one
sentence earlier — "a big box of stars and a big box" — so the loop is not a one-off.

This is a decoding failure rather than a model failure, and the metrics separate the
two. The repeated word-level 4-gram rate is **0.060 at temperature 0.0 but 0.000 at
0.5, 0.8 and 1.2**, and word-level distinct-2 rises from 0.882 to 0.988 over the same
range. Greedy decoding always takes the argmax, so once the model enters a state whose
most likely continuation reproduces a sentence it has already produced, nothing breaks
the cycle — the model's distribution has no say in it. Any temperature above zero
injects enough entropy to escape, which is exactly what the numbers show.

---

## Case 2 — Broken grammar and agreement mid-sentence

**Temperature 0.8.**

> One day, Timmy wanted to go to the park, so he asked his mommy if he could have him
> playing. His mommy bought the nuts and said hello. Timmy remembered his mommy
> **tost** his toys and she asked him **what he was wrong**.

**Failure type:** broken grammar — a non-word, plus a malformed complement clause.

**Observation.** Three distinct breakages in one passage. "if he could have him
playing" is a syntactically invalid complement; "tost" is not an English word at all;
and "asked him what he was wrong" collapses two constructions ("asked him what was
wrong" and "asked him what he did wrong").

"tost" is the informative one, and it is a direct consequence of the character-level
vocabulary. The model composes words letter by letter from 98 symbols with no notion of
a word as a unit, so nothing structurally prevents it from emitting a plausible letter
sequence that happens not to be a word. The measured **training-vocabulary word rate is 0.988 at
temperature 0.8** — about one word in eighty-three is an invented one, and "tost" is
that one in eighty-three. This is also why I added training-vocabulary word rate as a metric: the
required distinct-1/2/3 and repeated-4-gram figures are all perfect here (1.000, 0.988,
1.000 and 0.000), so the standard diversity metrics score this passage as flawless
while a reader can see it is not.

---

## Case 3 — Premature document boundary and loss of coherence

**Temperature 1.0.**

> Once upon a time, there was a little boy named Benny. Benny loved his car and
> **rolled his doll's teeth**. He was happy because he got to hold on the swings and
> sang with his mom and dad. They were talking to the park to play them with it.
>
> **Once upon a time, there was a little boy named Timmy.** Timmy loved to play
> outside in the sun.

**Failure type:** loss of coherence, ending in a premature document boundary
(context abandonment).

**Observation.** The story degrades progressively rather than all at once. "rolled his
doll's teeth" is semantically empty but grammatical; "They were talking to the park to
play them with it" has correct local syntax with no recoverable meaning; and then the
model abandons Benny entirely and starts a fresh story about a new character.

The restart is traceable to a preprocessing decision of mine. I joined stories with a
blank line (`"\n\n"`) so the model would have an explicit document boundary to learn
instead of one story running into the next mid-sentence. It learned that boundary, and
it also learned that `"\n\n"` is followed by `"Once upon a time"` — which is how most
TinyStories entries open. At temperature 1.0 the separator gets sampled while the
current story is still unfinished, and the model does the locally correct thing.

The underlying cause is the 128-character context window. Benny is introduced at
character 40 and is out of the attention window by character 170, so by the time the
model reaches the end of the passage there is nothing in context to be coherent *with*.
Starting a new story is the highest-probability continuation available to a model that
can no longer see what it was writing about.

---

## What these three share

None of the three is a training-stability problem. The run recorded **0 NaN/inf values
and 0 loss spikes**, gradient norms averaged 0.70 against a clip of 1.0, and validation
loss fell monotonically for all 12 epochs. The generalization gap is **0.033 nats**,
which means the model is not overfitting either.

That combination points somewhere specific: the failures are capacity and context
limits, not optimisation limits. Case 1 is fixed at the decoder (sample rather than
greedy, or add a repetition penalty). Cases 2 and 3 need either a longer context window
or a token-level vocabulary — and since validation loss was **still falling at epoch
12**, a wider or deeper model trained longer is the change I would test first.
