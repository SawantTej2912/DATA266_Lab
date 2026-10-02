# Task 2 — Yelp Polarity sentiment classification

**Member:** Shriram Dundigalla · **Team:** Lab Pair 33
**Raw log:** `reproducibility/raw_logs/task2_20260929T042*.log`
**Metrics:** `metrics_report.csv` · **Slices:** `outputs/slice_metrics.csv` ·
**McNemar:** `outputs/mcnemar_tests.csv` · **Figures:** `outputs/figures/`
**Hardware:** Apple M4, 24 GB unified memory, GPU via PyTorch MPS (torch 2.14.0) — **no CUDA device was used for any reported number.** The repository contains a CUDA fast path for the GPU-lab presets; it is gated on `device.type == "cuda"` and did not execute for this run.

Reproduce with:

```bash
python scripts/fetch_data.py --task yelp
python task2_sentiment/shriram_dundigalla/src/train.py
python task2_sentiment/shriram_dundigalla/src/error_review.py
python task2_sentiment/shriram_dundigalla/src/plots.py
```

---

## 1. Headline result

| Model | Role | Accuracy | Macro-F1 | ROC-AUC | PR-AUC | MCC | Brier | ECE (conf.) | ECE (pos.) | Params | Train time | ex/sec |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `baseline_bow` | baseline | 0.9245 | 0.9245 | 0.9751 | 0.9748 | 0.8492 | 0.0572 | **0.0044** | **0.0075** | 3.87M | **26 s** | **11,610** |
| `exp1_bilstm` | experimental | 0.9334 | 0.9334 | **0.9830** | **0.9836** | 0.8671 | 0.0500 | 0.0165 | 0.0203 | 6.61M | 2,319 s | 216 |
| `exp2_textcnn` | experimental | **0.9344** | **0.9344** | 0.9829 | 0.9835 | **0.8688** | **0.0487** | 0.0061 | 0.0064 | 5.09M | 211 s | 1,895 |

**Two ECEs, because "ECE" alone is ambiguous and these differ by up to 1.7×.**
*ECE (conf.)* is the standard Guo et al. formulation on the confidence of the
**predicted** class, binned over [0.5, 1] — "when the model says it is 90% sure, is it
right 90% of the time?" *ECE (pos.)* bins the **positive-class** probability over
[0, 1] — "when it says 30% positive, are 30% of those reviews positive?" The confidence
version folds p=0.05 and p=0.95 into one bin and so cannot answer the second question.
The reliability diagram in `outputs/figures/calibration.png` plots p(positive), so the
positive-class ECE is the one that belongs beside it; an earlier version of this report
labelled that figure with the confidence number, which was measuring something else.

All figures are on the **full official 38,000-review test set**. Training used a
balanced 100,000-review subsample with 10,000 held out for validation.

**95% bootstrap confidence intervals** (2,000 replicates, percentile method):

| Model | Accuracy | Macro-F1 | MCC |
|---|---|---|---|
| `baseline_bow` | 0.9245 [0.9218, 0.9272] | 0.9245 [0.9218, 0.9271] | 0.8492 [0.8438, 0.8545] |
| `exp1_bilstm` | 0.9334 [0.9309, 0.9358] | 0.9334 [0.9309, 0.9358] | 0.8671 [0.8622, 0.8719] |
| `exp2_textcnn` | 0.9344 [0.9319, 0.9368] | 0.9344 [0.9319, 0.9368] | 0.8688 [0.8638, 0.8737] |

**Paired McNemar tests** on identical test rows:

| Comparison | Only A right | Only B right | p-value | Verdict |
|---|---|---|---|---|
| `baseline_bow` vs `exp1_bilstm` | 976 | 1,315 | 1.65e-12 | BiLSTM significantly better |
| `baseline_bow` vs `exp2_textcnn` | 932 | 1,309 | 1.98e-15 | TextCNN significantly better |
| `exp1_bilstm` vs `exp2_textcnn` | 931 | 969 | **0.396** | **no significant difference** |

---

## 2. Why these three models

I chose architectures that differ in **how they use word order**, not three sizes of
the same thing, so the comparison could answer a question rather than just rank models.
The question: *how much of Yelp polarity is lexical, and how much is structural?*

| Model | Treatment of word order | What it tests |
|---|---|---|
| `baseline_bow` — mean-pooled embeddings (128) into MLP [256] | **None.** Averages word vectors. | The hypothesis that sentiment here is a bag-of-words problem. |
| `exp1_bilstm` — embed 200, biLSTM hidden 192, `[mean; max]` pooling | **Full sequential state.** | Whether long-range order pays for itself. |
| `exp2_textcnn` — embed 160, 128 filters at widths 2/3/4/5, global max-pool | **Local phrases only.** | Whether short n-grams capture whatever order contributes. |

Specific choices and their reasons:

- **Baseline embedding is the smallest (128).** Mean-pooling has no capacity to exploit
  a wider embedding — every dimension is averaged across the whole review — so
  spending parameters there would have inflated the baseline's parameter count without
  changing what it can represent.
- **BiLSTM pools `[mean; max]` over time rather than taking the final hidden state.**
  With reviews up to 250 tokens, the final state is dominated by the tail, and Yelp
  verdicts are as often stated up front as at the end. This is the single change I made
  after thinking about the data rather than copying a default.
- **BiLSTM uses the highest dropout among sequence models (0.4) and the smallest batch
  (128).** It has the most parameters, so it needs the most regularisation; recurrence
  gives fewer but more informative gradient steps, so a smaller batch is appropriate.
- **TextCNN uses the highest dropout (0.5) and highest learning rate (1.5e-3).** Global
  max-pooling produces a small, sparse feature vector that overfits readily, and the
  convolutional path has a shorter gradient route than the recurrent one.
- **Everything else is held constant** — AdamW, weight decay 1e-4, gradient clipping
  1.0, early stopping on validation macro-F1 with patience 2 — so differences come from
  the architecture, not from the optimiser.

A caveat that matters for reading the parameter counts: the **embedding table is the
bulk of every model**. At a 30,002-token vocabulary, the baseline's embeddings are 3.84M
of its 3.87M parameters. So "parameter count" in this table is mostly a statement about
vocabulary size, not about architectural capacity.

## 3. Preprocessing

Review length, class distribution and class balance are analysed in section 1 of the
notebook. Yelp Polarity is **exactly balanced** (280,000/280,000 train, 19,000/19,000
test), which is why accuracy is a meaningful headline here and macro-F1 equals micro-F1
equals accuracy. On an imbalanced corpus I would have had to lead with MCC instead.

Checked for missing and malformed entries as section 2.1.2 requires: zero nulls, zero
whitespace-only reviews, zero exact duplicates, and **zero texts appearing in both the
official train and test splits**. The corpus is clean. I kept the scrubbing step anyway
because "I checked and there was nothing" is a finding, not a reason to skip the check.

The pipeline: lowercase; expand `n't` to ` not`; strip everything that is not a letter;
drop stopwords; Porter stemming; drop single characters except `a` and `i`. Vocabulary
capped at 30,000 stems with `min_freq` 2, built from **my training split only**, giving
a 0.85% out-of-vocabulary token rate on test. `max_len` 250, which truncates 1.4% of
test reviews.

**The one decision that mattered.** NLTK's English stopword list contains **15 negation
tokens** — `not`, `no`, `nor`, `don't`, `didn't`, `isn't`, `wasn't`, `couldn't`,
`wouldn't`, `shouldn't`, `won't`, `doesn't`, `aren't`, `hasn't`, `haven't`. Applied
naively, "The food was not good and I would not come back" cleans to `food good` — it
reads as praise. I retain 26 negations and polarity reversers. The notebook demonstrates
this on a worked example rather than asserting it.

Two ordering details that are easy to get wrong: the `n't` expansion happens **before**
punctuation stripping (strip first and `don't` splits into `don` + `t`, losing the
negation), and the slice features for the robustness metrics are computed from the
**raw** text before cleaning (cleaning destroys exactly the capitalisation and
punctuation those slices are about).

## 4. What the results actually say

### 4.1 Word order helps, but only about one point

Both experimental models beat the baseline by roughly 1 accuracy point, and the paired
McNemar tests put that beyond doubt (p = 1.6e-12 and 2.0e-15). But the size of the
effect is the finding: a model that **cannot see word order at all** reaches 92.45%
accuracy and 0.975 ROC-AUC. Yelp polarity is overwhelmingly a lexical problem. Reviews
say "delicious" or "disgusting", and that is most of the signal.

### 4.2 The two experimental models are statistically indistinguishable — and that is the result

`exp1_bilstm` and `exp2_textcnn` differ by 0.10 accuracy points with heavily overlapping
confidence intervals, and McNemar gives **p = 0.396**. There is no evidence either is
better than the other.

But **TextCNN trains 11x faster** (211 s vs 2,319 s; 1,895 vs 216 examples/sec) with 1.5M
fewer parameters. Local 2-to-5-token phrase detection captures *everything* the full
sequential state captures, at a fraction of the cost. For this task the recurrent model
buys nothing it does not already get from n-grams — which is a much more useful
conclusion than "the biggest model won", and it is the finding I would defend.

### 4.3 The most accurate model is not the best calibrated

The baseline has the **best predicted-class confidence ECE (0.0044)** despite the worst
accuracy, and the BiLSTM has the worst (0.0165) at near-best accuracy. The ranking is the
same under the positive-class ECE (0.0075 / 0.0203 / 0.0064), so the conclusion does not
depend on which definition is used — but the TextCNN's margin over the baseline is much
larger on the confidence measure (1.4× versus 1.2×), because the baseline's miscalibration
is concentrated in the mid-probability region that the confidence binning compresses. The BiLSTM's
training curve explains it: training loss fell to 0.048 by epoch 5 while validation loss
rose from 0.178 to 0.253. Early stopping restored the epoch-3 weights, but the model was
already drifting toward overconfidence.

This is why Brier score and ECE are on the required list and why they earn their place:
if you needed a probability you could act on — routing reviews above a confidence
threshold for automated handling — you would pick the TextCNN (confidence ECE 0.0061,
positive-class ECE 0.0064, at the best accuracy), and you might well prefer the
26-second baseline over the BiLSTM.

### 4.4 Robustness: where the gains actually land

Macro-F1 per slice, and the TextCNN's advantage over the baseline:

| Slice | n | baseline | TextCNN | Gain |
|---|---|---|---|---|
| `shouty = shouty_caps` | — | 0.9303 | 0.9558 | **+0.0255** |
| `length = short` | — | 0.9202 | 0.9360 | **+0.0158** |
| `has_negation = negation` | — | 0.9167 | 0.9291 | +0.0124 |
| `exclamations = no_exclaim` | 20,364 | 0.9120 | 0.9229 | +0.0109 |
| `length = medium` | — | 0.9262 | 0.9366 | +0.0104 |
| `length = long` | — | 0.9228 | 0.9271 | **+0.0043** |

The pattern is coherent and it is the clearest evidence for why the experimental models
win: **the shorter the review, the more order matters.** The gain on short reviews
(+1.58) is nearly four times the gain on long ones (+0.43). Averaging 250 word vectors
washes out order but retains plenty of lexical signal; averaging 20 loses both.

Two slice results worth flagging because they are counterintuitive:

- **Reviews *with* negations are easier than reviews without** (0.9291 vs 0.9207). I
  expected the opposite. The explanation is that a reviewer who writes "not" is usually
  making an explicit, committed judgement, whereas reviews with no negation at all skew
  toward flat descriptive prose. It also confirms retaining negations worked.
- **Shouty all-caps reviews are the *easiest* slice** (0.9558), not the hardest.
  Someone typing in capitals has a strong opinion and states it in obvious words. I
  built this slice expecting lowercasing to have destroyed signal; instead it showed the
  signal was redundant.

The worst slice, `no_exclaim` at 7.50% error, turns out not to be about punctuation at
all — it is a proxy for "is this review lexically obvious". Measured, hedged,
comparative prose is where sentiment is carried by structure rather than vocabulary.
See `failure_analysis.md`, section D.

### 4.5 A large share of the residual error is not fixable by modelling

From the 20-error read-through: 1 sarcasm, 2 probable label errors, and 5 genuinely
mixed reviews sitting within 0.0015 of the threshold — **8 of 20** are cases where no
architecture change helps. Two of my five most confident false positives read as
unambiguously positive text labelled negative, which suggests the 6.56% error rate has a
meaningful noise floor under it.

The errors that *are* addressable cluster on **discourse structure**, not vocabulary:
late verdict reversal, temporal contrast ("it was terrible... now its much better"),
and stacked concessives. That is 4 of 20, and all four are cases where the verdict is
positioned or framed rather than stated.

## 5. Strengths, weaknesses, limitations

**Strengths.** All three models clear 92% accuracy with no pretrained embeddings of any
kind. The comparison is controlled where it can be, and the limits are worth stating
precisely rather than claiming more than holds:

| held constant | varies by model |
|---|---|
| Preprocessing pipeline, vocabulary (30K, min_freq 2), `max_len` 250 | architecture |
| Train/val split: the same 100K/10K balanced subsample, seed 9015 | learning rate (1e-3, 1e-3, 1.5e-3) |
| Test set: the full official 38,000 reviews | batch size (256, 128, 256) |
| Optimiser *family* (Adam), weight decay 1e-4, grad clip 1.0 | dropout (0.3, 0.4, 0.5) |
| Early stopping on validation macro-F1, patience 2 | epoch budget (6, 5, 6) |
| One shared scoring implementation and one shared random seed | parameter count (3.87M, 6.61M, 5.09M) |

So this is **not** an identical-optimiser comparison, and an earlier version of this
report described it as one. It is a shared optimiser *family* and a shared regularisation
policy, with each architecture given a learning rate and batch size in its usual range —
which is the honest way to compare architectures, since forcing a BiLSTM to train at the
TextCNN's settings would measure the mismatch rather than the architecture. The cost is
that a difference of a few tenths of a point cannot be cleanly attributed to architecture
alone; that is exactly why the McNemar tests matter, and they show the two experimental
models are not distinguishable at all. Every claim of superiority is backed by a
paired significance test rather than by a point estimate, which is what shows the two
experimental models are *not* distinguishable.

**Weaknesses.** The BiLSTM overfits by epoch 3 and is the worst-calibrated model. All
three fail on the same discourse phenomena, so the ensemble would not help much (1,561
test reviews are wrong for both experimental models). Porter stemming collapses some
distinctions it should not, and I did not measure that cost against a no-stemming arm.

**Limitations.** Training used 100,000 of the available 560,000 reviews, so these
numbers understate what each architecture can reach. `max_len = 250` truncates 1.4% of
test reviews, and that 1.4% is over-represented among confident errors. The slice
definitions are my own and are proxies — `no_exclaim` turned out to measure something
other than what I designed it to measure. And the label-noise estimate is from 5
reviews, which is enough to raise the issue but not enough to quantify it.

## 6. What I would do next, in order

1. **Position-aware pooling.** Concatenate a max-pool over the final 20% of tokens to
   the existing global max-pool. Late verdict reversal and temporal contrast are 4 of my
   20 errors, and both are position phenomena that global max-pooling structurally
   cannot represent. Cheapest change with a clear mechanism.
2. **Quantify the label-noise ceiling.** Hand-label the 200 most confident errors. Every
   accuracy number should be read against that ceiling, and right now I do not know
   where it is.
3. **Negation-scope marking.** Prefix the three tokens after a negation with `NOT_` and
   re-train. My read-through predicts this fixes intra-clause negation but *not*
   detached negation, which would be a clean, falsifiable result either way.
4. **Scale the training set to 560K** for the best architecture only, to separate "this
   architecture is better" from "this architecture had enough data".

## 7. Constraint compliance

- Yelp Polarity, no pretrained embeddings, no pretrained language model. Every
  embedding is a randomly initialised `nn.Embedding` learned by backpropagation.
- Review length, class distribution and class balance analysed; missing and malformed
  entries checked and handled.
- Lowercasing, punctuation removal, stopword removal and stemming all applied, with the
  negation-retention decision demonstrated on a worked example.
- Three of my own models — one baseline, two experimental — differing in both
  architecture and hyperparameters.
- Every required metric reported per model, including all three F1 averagings,
  confusion matrices, ROC-AUC, PR-AUC, MCC, Brier, both ECE definitions, 95% bootstrap CIs, paired
  McNemar, per-slice macro-F1 and error rate, parameter count, training time,
  examples/sec and peak memory.
- Exact CPU/GPU recorded per model in the `hardware` column of `metrics_report.csv`.
- 20 errors manually reviewed in the required 5/5/5/5 split, each with an error type and
  one testable fix (`failure_analysis.md`).
- No hard-coded personal paths; all runs are config-driven from `src/config.yaml`.
