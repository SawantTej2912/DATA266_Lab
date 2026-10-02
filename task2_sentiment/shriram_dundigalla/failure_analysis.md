# Task 2 — manual error review (20 errors)

**Model reviewed:** `exp2_textcnn` — my highest macro-F1 model (0.9344).
**Errors:** 2,492 of 38,000 test reviews (**6.56%**).
**Worst slice:** `exclamations = no_exclaim`, error rate 0.075 over 20,364 reviews.
**Selection record with diagnostics:** `outputs/error_review_exp2_textcnn.csv`

The four buckets required by section 2.2.4 — 5 confident false positives, 5 confident
false negatives, 5 near-threshold errors, 5 from the worst slice — were selected
programmatically (`src/error_review.py`) so the choice is reproducible and not
cherry-picked. The **error types below are my own reading of the review text**, and in
most cases they disagree with the automatic heuristic the script attaches: the script
labelled 10 of 20 "negation scope" because 17 of the 20 contain a negation token, but
reading them shows negation is usually incidental to what actually went wrong.

Every fix below names an experiment that would confirm or refute the diagnosis, not
just a direction to move in.

---

## A. Confident false positives (true negative, predicted positive)

### A1 — Sarcasm · `[11429]` p(positive) = 0.9994, 4 tokens

> Yay. Cafeteria food. Yay.

**Error type:** sarcasm / ironic inversion.
Every surviving token after cleaning is positive. There is no lexical, syntactic, or
positional cue that inverts it — the inversion lives entirely in world knowledge about
cafeteria food and in the flatness of the repetition. None of my three architectures
can represent this, and neither could a much larger one without pretraining.

**Testable fix:** hand-label a 100-review sarcasm probe set drawn from short reviews
with high positive-word density and negative labels, then measure accuracy on it
separately. If it lands near 50%, sarcasm is an irreducible error floor for this model
class and should be reported as such rather than chased.

### A2 — Late verdict reversal · `[7661]` p = 0.9991, 24 tokens

> We had a great time and I thoroughly enjoyed the food, **but I need to give this a 1
> start given that if we had eaten a week later, this salmonella outbreak could have
> caused immeasurable harm to our child.**

**Error type:** late verdict reversal under a concessive clause.
Roughly 80% of the tokens are positive; the verdict arrives in the final clause and
inverts everything before it. Global max-pooling is structurally the wrong operator
here — it takes the single strongest activation anywhere in the review, so a strong
"thoroughly enjoyed" detector fires and nothing downstream can suppress it.

**Testable fix:** add a position-weighted pooling branch (or simply concatenate a
max-pool over the final 20% of tokens) and re-train. If accuracy on
`but`/`however`-containing reviews improves while overall accuracy holds, the
pooling operator was the binding constraint, not the encoder.

### A3 and A4 — Probable label noise · `[29330]` p = 0.9997, `[12480]` p = 0.9990

> Wow love the place and everything is very clean and new! Great place to come and
> relax worth a try! Cheers, Eric Van Nguyen. Visited April 2012

> my husband had an omelette that was good. i had a blt, a little on the small side for
> $10, but bacon was great. Our server was awesome!

**Error type:** probable label error, not a model error.
Both are labelled negative (1-2 stars) and both read as unambiguously positive. I
cannot verify the true star rating — Yelp Polarity ships no star column — so I am
calling these *probable* rather than certain. But they are the two most confident
errors the model makes, and a model being maximally confident and wrong on text that a
human reads the same way the model does is the signature of label noise rather than
model failure.

If this rate holds, it bounds the achievable accuracy: 2 of my 5 most confident false
positives being label noise suggests a non-trivial share of the 6.56% error rate is
irreducible.

**Testable fix:** hand-label the 200 highest-confidence errors (both directions) as
correct-label or wrong-label. The share that are wrong-label is a direct estimate of
the dataset's label-noise ceiling, and every accuracy number in my report should then
be read against that ceiling rather than against 100%.

### A5 — Rhetorical question · `[37451]` p = 0.9990, 39 tokens

> Decor and atmosphere really good... Food was good, but not inexpensive... **Was it
> worth $20. No, more in the area of $12-14.** Service was avg.

**Error type:** verdict carried by a rhetorical question and a bare "No".
The negative judgement is expressed as a question answered by an isolated "No" —
syntactically detached from the thing it negates. My negation-retention step keeps the
"No" token, which is necessary but plainly not sufficient: the model keeps the word and
still cannot bind it to a referent four tokens earlier.

**Testable fix:** re-score with negation-scope marking (prefix the next three tokens
after a negation with `NOT_`) and check specifically whether this review flips. A bare
sentence-initial "No" has nothing within three tokens to mark, so I predict it will
*not* flip — which would establish that scope marking fixes intra-clause negation but
not detached negation, and narrow where the remaining error lives.

---

## B. Confident false negatives (true positive, predicted negative)

### B1 and B2 — Temporal contrast · `[30793]` p = 0.0005, `[22807]` p = 0.0003

> This place is **so much better since they changed owners**. My wife and I went when it
> was the old owners, **it was terrible**. We waited forever and the food never came
> before we walked out... **It was horrible.** Now its much better. The staff are very
> friendly... I have nothing but positive things to now say about this place.

> **EDIT: They really did change the service up since I last posted this.**
> Horrible service. Used to be my favorite pizza in the city...

**Error type:** temporal contrast — past-negative, present-positive.
This is the most interesting failure in the set and it is a genuine architectural
limitation rather than a data problem. Both reviews spend most of their length
describing a *past* bad experience in order to praise the *present*. The negative
vocabulary ("terrible", "horrible", "waited forever") is dense and literal; the
positive reframing is carried by tense and by two discourse markers, "since they
changed owners" and "EDIT:".

Global max-pooling has no way to discount a strong negative activation on the grounds
that it sits in a past-tense clause. Notably the BiLSTM, which *does* carry sequential
state, gets `[30793]` wrong too — so this is not something a recurrent encoder fixes
for free at this scale.

**Testable fix:** build a 150-review temporal-contrast probe set by filtering for
`used to|since they|has changed|EDIT:|under new` plus a positive label, and compare all
three models on it. If all three sit far below their overall accuracy, discourse
structure is a shared blind spot of the whole model family and belongs in the report as
a limitation, not as a tuning problem.

### B3 — Target confusion · `[2285]` p = 0.0006, 28 tokens

> **Someone deleted my review.. intentionally. That is very rude and disrespectful.**
> Don't be silly guys.. I like this place and new owner is very nice and friendly. I
> really liked the chicken yakisoba... Good price and service :)

**Error type:** target confusion — the complaint is aimed at Yelp, not the restaurant.
The negative sentiment is real, strongly worded, and directed at an entity that is not
the subject of the review. The model has no notion of aspect or target.

**Testable fix:** check whether the error survives when the first sentence is removed.
If the prediction flips to positive on the truncated text, the failure is target
attribution rather than sentiment detection, and the fix direction is aspect-based
sentiment rather than more capacity.

### B4 — Concessive with negation · `[26683]` p = 0.0005, 13 tokens

> I've just been forced to concede that, **despite still not digging their ordering
> process**, their food is just **too good to disrespect** with a 2 star review.

**Error type:** stacked concessive with a double inversion.
"despite still not digging" is a negation inside a concessive, and "too good to
disrespect" is a positive expressed through two negative-polarity words. After
stemming and stopword removal the model sees roughly `forc conced despit still not dig
order process food good disrespect 2 star review` — the surviving tokens skew negative.

**Testable fix:** this is the clearest candidate for negation-scope marking. Re-train
with `NOT_` prefixing and re-score this specific review; unlike A5, the negation here
does have its target within three tokens, so it *should* flip if scope marking works.

### B5 — Truncation · `[21780]` p = 0.0002, 434 tokens, **truncated**

A 434-token review cut at `max_len = 250`. The visible portion is a neutral,
slightly grudging setup ("you get spoiled with exceptional service by your host");
the verdict is in the discarded 184 tokens.

**Error type:** truncation — the evidence is outside the model's input.
This is the only one of the 20 that is purely a preprocessing decision and not a
modelling limitation. Only 1.4% of test reviews truncate, so the aggregate cost is
small, but the cost is concentrated in exactly the reviews the model is most confident
and wrong about.

**Testable fix:** re-score only the 1.4% of truncated test reviews with `max_len`
raised to 512. If accuracy on that subset rises materially, raise `max_len`; if it does
not, the length was never the problem and I can stop paying attention to truncation.

---

## C. Near-threshold errors (the model is correctly uncertain)

All five sit within 0.0015 of the 0.5 threshold. The representative case:

### C1 — `[29444]` p = 0.5003, 12 tokens

> **Not sure** how we spent $70 on 5 slices of pizza, 3 strombolis and 2 sodas. Pizza
> was **good**, but the strombolis were **pouring grease**.

**Error type:** genuinely mixed sentiment — arguably not a model failure at all.
Positive on the pizza, negative on the price and the grease. The model outputs 0.5003
and is marked wrong because the threshold falls on the other side of a coin flip.
Four of the five near-threshold errors read this way.

This is worth separating from the confident errors, because it says something different
about the model. The confident errors in sections A and B are cases where the model was
sure and wrong. These are cases where the model correctly recognised that the review is
mixed and the binary label is the thing that cannot represent it. My ECE of 0.0061
supports that reading — the probabilities are well calibrated, so a 0.50 output
genuinely means "coin flip".

**Testable fix:** measure accuracy as a function of a confidence band. If accuracy
inside `0.45 < p < 0.55` is near 50% and accuracy outside it is far higher, an
abstention option would convert most of these from errors into referrals, and the
right product answer is to abstain rather than to add capacity.

---

## D. Slice-specific failures — `exclamations = no_exclaim`

The worst slice for all three models: **7.50% error rate** for the TextCNN against
5.47% on reviews that contain an exclamation mark.

### D1 — `[0]` p = 0.0194, 54 tokens

> **Contrary to other reviews**, I have **zero complaints** about the service or the
> prices... this is one place that I **do not** feel like I am being taken advantage
> of... Other auto mechanics have been notorious for capitalizing on my ignorance of
> cars, and have **sucked my bank account dry**. But here, my service and road coverage
> has all been well explained.

**Error type:** negation scope plus third-party contrast.
Praise expressed entirely through negated negatives — "zero complaints", "do not feel
taken advantage of" — alongside a vivid negative description of *other* businesses.
Every strongly-polarised token in the review is negative; the positive meaning exists
only in their negation and attribution.

**Why this slice is the worst slice** is the useful finding here, and it is not about
exclamation marks as such. Reviews without exclamation marks are measured, hedged,
comparative prose, and that is where sentiment is carried by structure rather than by
vocabulary. Reviews *with* exclamation marks say "AMAZING!!!" and are lexically trivial.
The exclamation slice is a proxy for "is this review lexically obvious", which is
precisely the axis my baseline and my experimental models differ on.

The slice table supports this directly: the TextCNN's advantage over the baseline is
largest on `shouty_caps` (+2.55 points) and `short` reviews (+1.58), and smallest on
`long` reviews (+0.43). When there are fewer words, *how* they are arranged matters
more.

**Testable fix:** build a "lexically obvious" indicator — count of strongly-polarised
unigrams by training-set pointwise mutual information — and slice by it directly
instead of by exclamation marks. If the error-rate gap across that slice is wider than
the exclamation gap, it is the better robustness axis to report, and it would confirm
that exclamation marks were only ever a proxy.

---

## Summary of error types across the 20

| Error type | Count |
|---|---|
| Negation scope / detached negation | 3 |
| Temporal contrast (past-negative, present-positive) | 2 |
| Genuinely mixed sentiment (near-threshold) | 5 |
| Probable label noise | 2 |
| Late verdict reversal / concessive | 2 |
| Sarcasm | 1 |
| Target confusion | 1 |
| Truncation | 1 |
| Rhetorical question | 1 |
| Insufficient evidence (very short) | 2 |

Three conclusions I draw from the read-through:

1. **Retaining negations was necessary but not sufficient.** 17 of the 20 errors
   contain a negation token, which vindicates keeping them — but only 3 are actually
   *caused* by negation handling. The rest fail on scope, discourse structure, or
   target, all of which survive having the word present.

2. **A meaningful share of the residual error is not addressable by modelling.**
   Sarcasm (1), probable label noise (2) and genuinely mixed reviews (5) together are 8
   of 20. Chasing the remaining 6.56% as though it were all model error would waste
   effort, and the honest ceiling is lower than 100%.

3. **The errors that *are* addressable cluster on discourse structure**, not on
   vocabulary. Late verdict reversal, temporal contrast and concessive constructions
   are 4 of 20, and all four are cases where the verdict is positioned or framed rather
   than stated. That is the one place where a targeted change — position-aware pooling,
   or negation-scope marking — has a clear mechanism to help, and it is what I would
   try first.
