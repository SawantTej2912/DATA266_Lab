# Lab Pair 33 — shared evaluation protocol

**Members:** Shriram Dundigalla · Tejas Nandkishor Sawant
**Kaggle team name:** `PairProgramming_Team_33` (exact format mandated by the TA)

> **Updated 28 Sep** after Savitha's announcement. What changed:
>
> - **Part 2 dataset is confirmed Yelp.** The IMDB line in the folder-structure
>   document is an error and is to be disregarded. No change for us — we were already
>   on Yelp.
> - **Demo policy relaxed.** We each still train our own models for all three parts,
>   but at the demo the team may present its **best** model rather than every
>   member's, and we may split task leadership between us.
> - **Kaggle is the exception and is not relaxed.** "Individual contribution to the
>   Kaggle leaderboard for Part 3 is still required from both team members — this
>   cannot be skipped or covered by one teammate alone." **Tejas must train his own
>   CycleGAN and make his own submission.** This is the one thing in the lab neither
>   of us can do for the other.
> - **Datasets do not go in the repo.** They are zipped, uploaded to Google Drive, and
>   linked from `README.md` with read access enabled.
> - **Smoke-test before the GPU session.** Explicitly recommended by the TA. Every
>   task in this repo has a `--smoke` flag for exactly this.
> - **The deadline will not be moved, for any reason.**

Section 8 of the brief asks us to "agree as a team on one evaluation script per task
**before anyone starts training**, so every member's metrics are directly comparable."
This file is that agreement. It exists so the per-task comparison tables in the report
compare *models*, not measurement choices.

**The rule:** we each build our own architectures and pick our own hyperparameters —
that is required, and Section 2 forbids near-identical models. But we hold the
**measurement** identical. Anything in this file is fixed for both of us; anything not
in this file is ours to choose independently.

If either of us needs to deviate, note it here with the reason *before* the run, so the
report can state it rather than the comparison quietly breaking.

---

## Shared for all three tasks

- **Seeds.** Set `random`, `numpy` and `torch` from one seed per run and record it.
  Shriram uses 9015 (his SID4). Tejas: use your own, and record it.
- **Device disclosure.** Every run records the exact CPU/GPU string. Helper already
  written: `task2_sentiment/shriram_dundigalla/src/train.py::hardware_string`.
- **Raw logs.** Every training run appends an unedited log to
  `reproducibility/raw_logs/`. Do not tidy these afterwards — Section 5 grades them as
  the evidence trail, "untouched since the run".
- **No personal paths.** All paths resolve from the repo root and live in a YAML config.
- **Manifest.** Each of us writes `reproducibility/manifests/<name>_manifest.md` listing
  package versions, the environment file, and which checkpoint maps to which reported
  number.

---

## Task 1 — GPT from scratch

### Fixed (must match)

- **Scoring code:** `task1_llm/shriram_dundigalla/src/metrics.py`. Import it; don't
  reimplement it. Cross-entropy is averaged **over characters, not over batches**, so a
  short final batch cannot skew the mean.
- **Split unit.** The brief's "100K / 10K" is read as **sequences**, and windows are
  **non-overlapping**. 100,000 training and 10,000 validation windows.
- **Split boundary.** The train/validation split is over **whole stories**, never by
  cutting the concatenated character stream at an offset. Cutting one stream leaks the
  text either side of the cut into both halves' sliding windows.
- **Shared raw data:** `task1_llm/data/tinystories_train_first20000.jsonl`
  (regenerate with `python scripts/fetch_data.py --task tinystories --n-stories 20000`).
- **Metric definitions.** Perplexity is `exp(CE)` per character. Bits-per-character is
  `CE / ln 2`. Generalization gap is `val_CE - train_CE`. Distinct-1/2/3 and the
  repeated-4-gram rate are reported at **word** level (character level saturates near
  1.0 and cannot detect repetition); both are computed and saved.
- **Generation probe.** Prompt `"Once upon a time"`, 400 new tokens, temperatures
  `[0.0, 0.5, 0.8, 1.0, 1.2]`. Temperature 0.0 is greedy. The headline row in
  `metrics_report.csv` uses T = 0.8.
- **Constraint.** No `nn.Transformer`, `nn.TransformerEncoderLayer`,
  `nn.MultiheadAttention`, or `F.scaled_dot_product_attention`.
- **Minimum 10 epochs**, LR warm-up plus a schedule, cross-entropy loss.

### Ours to choose independently

Block count, head count, `d_model`, FFN multiplier, dropout, context length, Pre-LN vs
Post-LN, optimiser and learning rate, batch size, epoch count above 10, tokenizer
details beyond character level.

### Already claimed — do not duplicate

| | Shriram |
|---|---|
| Blocks / heads / `d_model` | 6 / 8 / 256 |
| Context length | 128 |
| Norm placement | Pre-LN |
| Optimiser | AdamW, lr 3e-4, betas (0.9, 0.95), wd 0.1 on matmuls only |
| Schedule | 3% linear warm-up, cosine to 0.1x |
| Batch / epochs | 128 / 12 |
| Parameters | 4,822,016 |

Tejas: pick a different depth/width **and** a different schedule or optimiser setting,
so the pair differs in architecture *and* hyperparameters. A genuinely informative
contrast would be a longer context (256) at similar parameter count — Shriram's
failure case 3 is a context-window failure, so that comparison would actually test
something.

---

## Task 2 — Yelp Polarity

### Fixed (must match)

- **Dataset.** `fancyzhx/yelp_polarity`. The brief's repo diagram says "IMDB" but the
  task text and the resource link both say Yelp; we are going with Yelp. *(Confirm with
  the ISAs if there is any doubt.)*
- **Scoring code:** `task2_sentiment/shriram_dundigalla/src/metrics.py::full_report`.
- **Test set.** The **full official 38,000-row test split**, always. Never a subsample.
  This is the single most important item in this file: it is what makes our numbers
  comparable even if we train on different amounts of data.
- **Training subsample.** A **balanced 100,000** review sample for training plus 10,000
  for validation, drawn from the official train split. If you want to train on all
  560,000, do it as an *additional* run and report both, because otherwise the
  comparison table conflates architecture with training-set size.
- **Decision threshold** 0.5. **Bootstrap** 2,000 replicates, percentile method, 95%.
  **Calibration** 15 equal-width bins.
- **Positive class** is label 1.
- **Slices** for the robustness metrics come from
  `metrics.py::SLICE_FAMILIES` — length bucket (tertiles), negation present, shouty
  caps, exclamation present — computed from the **raw** text before cleaning, because
  cleaning destroys the capitalisation and punctuation the slices are about. Minimum
  200 rows per slice.
- **McNemar** is run baseline vs each experimental model, exact binomial when the
  discordant cells total under 25, otherwise chi-square with continuity correction.
- **Constraint.** No pretrained embeddings (no GloVe, word2vec, fastText vectors) and
  no pretrained LM. Every embedding is a randomly initialised `nn.Embedding`.
- **Error review.** 20 errors in the required 5/5/5/5 split — confident false
  positives, confident false negatives, near-threshold, slice-specific — each with an
  error type and one testable fix.

### Ours to choose independently

The three architectures, embedding dimensions, hidden sizes, dropout, learning rates,
batch sizes, epoch counts, early-stopping patience, vocabulary cap, `max_len`, and the
preprocessing details below.

### Preprocessing — one decision we should both make the same way

NLTK's English stopword list contains **15 negation tokens** (`not`, `no`, `nor`,
`don't`, `didn't`, `isn't`, `wasn't`, `couldn't`, `wouldn't`, `shouldn't`, `won't`,
`doesn't`, `aren't`, `hasn't`, `haven't`). Removing them deletes the strongest cue in
the corpus — "The food was not good" cleans to `food good`. Shriram retains 26
negations and polarity reversers.

This one is worth matching, because if one of us strips negations and the other keeps
them, every downstream difference is contaminated by it. If you deliberately want to
*test* the effect, make it one of your two experimental arms and say so.

### Already claimed — do not duplicate

| | Shriram |
|---|---|
| Baseline | `bag_of_embeddings` — mean-pooled embeddings (128) -> MLP [256] |
| Experimental 1 | `bilstm` — embed 200, hidden 192, `[mean; max]` pooling |
| Experimental 2 | `textcnn` — embed 160, 128 filters at widths 2/3/4/5, global max pool |
| Common | AdamW, wd 1e-4, clip 1.0, early stop on val macro-F1 patience 2 |
| Vocabulary | 30,000 stems, min_freq 2, `max_len` 250, Porter stemming |

Tejas: three different architectures. Options that would contrast usefully rather than
just differ — a GRU instead of an LSTM, a self-attention pooling layer over learned
embeddings, an FFN over TF-IDF-weighted averages, or a character-level CNN. Shriram's
three were chosen to vary *how word order is used* (none / global / local), so an
interesting fourth axis is **subword or character** granularity.

---

## Task 3 — CycleGAN

### Fixed (must match)

- **Kaggle.** Competition `data-266-fall-2026-gan-image-style-transfer`, team named
  exactly **`PairProgramming_Team_33`**. **Both of us must submit individually** — the
  TA's announcement states this cannot be covered by one teammate. Multiple
  submissions are allowed and encouraged; submit early and improve.
- **Kaggle submission format** (specified by the TA, 28 Sep). A file named
  `submission.csv` with exactly 3 columns and 2 rows:

  ```
  ID, FID, MiFID
  1, 43.456, 0.389
  ```

  `ID` must be numeric. Note what this means: the leaderboard is **self-reported**, and
  the FID and MiFID values "must match those computed using the provided evaluation
  script".
- Submitted images must be the direct output of our own trained CycleGANs. The
  integrity note makes any shortcut here a zero for the whole task.
- **Shared raw data:** `task3_gan/data/monet_jpg/` (300) and `task3_gan/data/photo_jpg/`
  (7,038). All images are already 256x256 RGB. Download with
  `kaggle competitions download -c data-266-fall-2026-gan-image-style-transfer`.

#### The evaluation script — corrected 2 Oct, please read this one

**A course evaluation script does exist**, and this protocol previously stated that it
did not. It is kept verbatim at `task3_gan/Part3_Evaluation_Script.ipynb`, and
`task3_gan/shriram_dundigalla/evaluate_local.py` is the same logic with its paths as
arguments instead of constants. **It is the authority for every reported number.** If
you scored your run against `real_stats.npz`, that figure is on a different scale and is
not comparable to the leaderboard or to mine, so it needs re-scoring before the report.

It differs from the `real_stats.npz` approach in five ways, each of which moves the
number:

| | `real_stats.npz` approach | the course script |
|---|---|---|
| Preprocessing | scale to [-1,1] ("tf") | **ImageNet mean/std** |
| Resize | bilinear interpolate to 299x299 | **`Resize(299)` + `CenterCrop(299)`** |
| Reference statistics | read from `real_stats.npz` | **recomputed from the real image folders** |
| Sample size | 7,038 | **300**, both sides truncated to the shorter |
| Reported value | one direction | **mean of both directions** |

The earlier `real_stats.npz` work was not wrong, and it is worth keeping straight why.
Re-deriving `mu_real` from the 300 real Monets under each plausible convention showed
the shipped statistics were built with the tf convention: self-FID 0.069 against
`real_stats.npz` and 0.0106 against `photo_stats.npz`, versus 20.68 and 27.66 for the
alternatives. That conclusion still holds. It simply answers a question that does not
determine our grade, because the course script never opens those files.

- **MiFID** is whatever the course script computes, which is the mean cosine distance
  between the i-th sorted real feature and the i-th sorted generated feature. The
  pairing is positional and there is no memorization threshold, so **despite the name
  this is not the memorization-penalized MiFID** from the Kaggle Monet competition.
  Report it as the script defines it, and say in the report what it actually measures --
  that observation earns marks under "quality of analysis" rather than being a complaint.
- **Sample size: 300 per direction.** The script caps each set at the first 300 sorted
  filenames, so generating more does not change the score. This replaces the earlier
  "use all 7,038" instruction, which was right for the `real_stats.npz` protocol and is
  simply not what the script does. FID is still upward-biased at small n -- the same
  checkpoint scored 93.47 at n=300 and 77.63 at n=7,038 under the old protocol -- but
  since both of us and the grader are capped at 300, that bias is common to everyone and
  cancels in any comparison.
- **Other reported metrics**, which the course script does not compute and we therefore
  own: KID (unbiased MMD, cubic polynomial kernel, 100 subsets of 100), LPIPS over
  random generated pairs (AlexNet backbone), and density/coverage at k=5. Reference
  point for diversity: the **real** Monet set scores LPIPS 0.785, density 1.105 and
  coverage 1.000 against itself. Keep these identical between us, or the comparison
  table compares implementations rather than models.
- **Which `submission.csv`.** If you score more than one run, note that a scoring cell
  writing to a fixed path overwrites the previous run's file. Mine did exactly that, and
  the Drive copy briefly held run 2's numbers under run 1's label.
  `scripts/verify_submission.py` now recomputes the mean of the two per-direction FIDs
  and rejects a `submission.csv` that disagrees with it, which is the cheap way to catch
  this.
- **Human audit.** 30 **fixed** sample indices, chosen once and used by both raters.
  Both of us rate all 30 blinded on style, content and artifacts; we report Cohen's
  kappa. *This requires both of us and cannot be done alone — it is the one item in the
  lab with a hard two-person dependency.*

### Ours to choose independently

Generator architecture (ResNet vs U-Net, block count), discriminator receptive field,
loss weights for cycle and identity terms, learning rate and schedule, buffer size,
augmentation, training resolution, epoch count.

### Already claimed — do not duplicate

Updated 2 Oct, and it swapped relative to the original plan. This protocol used to
suggest a U-Net for Tejas; Tejas built the ResNet first and Shriram rebuilt on a U-Net,
so the contrast the rubric asks for is intact with the two sides exchanged.

| | Shriram | Tejas |
|---|---|---|
| Generator | **U-Net, depth 8, ngf 64**, transposed-conv upsampling, dropout 0.5 in the first three decoder blocks | ResNet, 9 residual blocks, ngf 64 |
| Discriminator | 70x70 PatchGAN, ndf 64 | 70x70 PatchGAN, ndf 64 |
| Norm | InstanceNorm (`affine=False`) | InstanceNorm, reflection padding |
| Loss | LSGAN, cycle L1 lambda 10.0, identity 5.0 | his own |
| Optimiser | Adam, lr 2e-4, betas (0.5, 0.999), linear decay from epoch 50 | his own |
| Resolution / batch | 256 (resize 286 -> random crop 256) / 1 | 256 / 1 |
| Epochs | 100, **1000 steps each** = 100,000 updates | 60, 300 steps each = 18,000 updates |
| Parameters | 114.34M total (G 54,404,099 each, D 2,763,841 each) | ~28.3M total |
| Reported FID | 101.44 averaged, Kaggle rank 20 | **re-score with the course script** |

Two notes on reading that table. The mechanisms genuinely differ -- a ResNet preserves
content through residual identity paths, a U-Net through skip connections that hand the
decoder the encoder's spatial features -- so this is a real ablation rather than an
arbitrary difference. And our epochs are **not the same unit**: 1000 steps against 300.
Compare total updates or wall clock in the report, or the table will imply the U-Net was
undertrained when it actually saw over five times as many updates.

### One thing that is easy to get wrong — please read before you train

An "epoch" is ambiguous here because the domains are wildly unbalanced: 7,038 photos
against 300 Monets. Iterating the photo set makes one epoch 7,038 steps and shows each
Monet ~23 times, which overfits the discriminator fast. Both of us cap an epoch at
**300 steps**, the size of the smaller domain (`data.steps_per_epoch`), so "epoch"
means the same thing in both our logs.

**This cap has a trap in it that cost me a 22-epoch run, so please check your loader
before you start.** If you cap the epoch by returning a smaller `__len__`, PyTorch's
sampler can then only ever emit indices `0..length-1`. With the obvious
`files_A[i % len(files_A)]` that resolves to **the same first 300 photographs every
epoch for the entire run — 4.3% of the photo domain**, and nothing warns you. Losses
fall, samples look plausible, and the model quietly never sees 96% of its input domain.

Two consequences worth knowing about:

1. It is a **memorization regime**: 300 fixed photos against 300 fixed Monets, each seen
   once per epoch. My first run developed severe dark-blob artifacts under it, and the
   memorization is a strong candidate cause, so this also confounds any artifact
   analysis you do.
2. It quietly caps quality, because at inference you translate the *whole* photo set —
   images from a distribution the generator has largely never seen.

The fix is to hand each epoch a different slice of a shuffled permutation of domain A,
and to log `domainA_seen=<n>/7038` so the coverage is visible rather than assumed. With
that in place coverage reaches **exactly 100%** of domain A by epoch 24
(= ceil(7038/300)). The loader this was diagnosed in belonged to the ResNet attempt and
is no longer in the tree; the diagnosis is what matters, not the code.

**A second, nastier version of the same bug.** Moving the epoch slice onto the *dataset* and calling `ds.set_epoch()` from the training loop is still wrong the moment you set `persistent_workers=True`: worker processes are forked once and keep their own dataset copy, so the parent's mutation never reaches them and every epoch silently reloads epoch 0's slice. Measured with real workers: 300 distinct images across 4 epochs instead of 1,200 — and the `domainA_seen` log line still reported full coverage, because it was summing the parent's *intentions*. If you use persistent workers, put the epoch logic in a `Sampler` (the DataLoader re-iterates it in the parent each epoch) or leave persistent workers off.

**If you write your own loader, log the number of distinct domain-A images seen and
check it grows.** That one line would have saved me four hours.

### Two more traps I hit, so you do not have to

**Restore the LR scheduler when you resume, not just the optimiser.** A `LambdaLR`
constructed fresh starts at `last_epoch=0`, so a resumed run silently restarts its
schedule from the beginning. Mine ran 32 of 45 epochs at full learning rate and would
have finished at 0.73x instead of annealing to ~0 — invisible in the loss curves, and
the only symptom was the `lr=` field in the log not moving when it should have. Either
save `sched.state_dict()` in the checkpoint or step the scheduler forward
`start_epoch - 1` times after loading. CycleGAN's anneal-to-zero is not decoration:
without it the pair keeps oscillating instead of settling.

**Do not let a smoke test overwrite real weights.** A `--smoke` run that writes to the
same stable filename as a real run (any fixed `*_latest.pt` / `*_best.pt` name) will
replace trained weights with 16 steps of noise, and everything downstream scores
garbage without erroring. Guard those writes on `if not smoke`.

### DiffAugment — I recommended this and it was wrong for my run

The reasoning was: domain B is 300 paintings, `D_B` memorises them, and that is the
binding constraint. DiffAugment (Zhao et al. 2020) applies the *same differentiable*
transform to both the real and the fake branch everywhere D is evaluated, so D gains
nothing from learning it and G is not pushed to reproduce it. All of that is sound and
the implementation worked. It still made things worse.

| | baseline | DiffAugment |
|---|---|---|
| FID (averaged) | **101.44** | **112.97** |
| Generator gradient norm, mean | 27.0 | 137.0 |
| Generator gradient norm, max | 112.5 | 640.5 |
| Non-finite losses | 0 | 0 |

11.4 FID worse, with the gradient norm mean up 5x and the maximum up 5.7x. No NaN, so
this is not a numerical blow-up -- it is a harder optimization problem. The generator is
chasing a target that jitters under colour, translation and cutout on every step, and at
batch size 1 with a 54M-parameter U-Net there is no averaging to damp that noise.

The lesson is narrower than "DiffAugment is bad": it treats discriminator
**overfitting**, and my baseline was not overfitting. Its discriminator loss sat near
0.26 and never approached 0, so D was never winning. Check that your D is actually
winning before you reach for this -- if `D_loss` is heading toward zero you have the
problem it solves, and if it is not, you are paying the variance for nothing.

### Weight averaging — still worth trying, but the evidence here is thin

A GAN generator never converges to a point, it orbits one, so any single iterate is an
arbitrary draw from that orbit. Keeping an exponential moving average of the generator
weights (decay 0.999, warmed up) should land nearer the centre. It costs one extra copy
of the weights and a few multiply-adds per step and cannot destabilise training, because
nothing is ever trained against it.

Be clear about how weak my evidence is: the averaged generator beat the raw one on FID
in a **16-step sanity run** (272.8 vs 290.3). At FID in the hundreds, after sixteen
steps, that comparison means very little. Score both and keep whichever wins, but do not
budget time on the assumption it will help.

The adjacent point *is* well supported, though: **do not assume the last epoch is your
best one.** My 100-epoch run scored best at epoch 60, and the remaining 40 epochs --
about 40% of a 2.77-hour run -- did not improve the graded metric even though cycle loss
was still falling. Training loss was not a usable model-selection signal. Measure the
real metric periodically on checkpoints, or you are submitting an arbitrary point on the
orbit.

### Metrics — the rubric asks for more than the leaderboard does

The course script gives you FID and MiFID in both directions and nothing else. The
rubric additionally wants KID both directions, cycle-reconstruction L1,
content-preservation cosine, LPIPS, gradient norms and NaN count, the loss curves, the
parameter/time/throughput/memory block, and the human audit. Those are ours to compute,
so keep the implementations identical between us or the comparison table compares code
rather than models.

The reverse direction no longer needs hand-built photo statistics. That was necessary
when the reference came from `real_stats.npz`, which only covers Monets; the course
script recomputes its reference from whichever real folder you point it at, so both
directions are on the same scale by construction.

One check worth doing: your standalone cycle-L1 should match `loss_cycle / lambda_cycle`
from your training log. Mine agree to 0.35%, which is good evidence that both are right.

### Human audit — this one needs both of us

Section 3.2 requires a blinded audit of 30 fixed samples on style, content and
artifacts, with 2 raters and inter-rater agreement. The tooling is at
`scripts/human_audit.py`. It hides provenance behind shuffled sample IDs and mixes in
unlabelled real Monets as controls, so a rater who scores everything the same is visible
in the output rather than invisible.

The pool is **already built** and waiting in `task3_gan/human_audit/`: 30 scored samples
plus 5 controls, all traced byte-for-byte to `pred_B2A`. Photo→Monet is the right
direction for this, because the rubric's style question is "how Monet-like is it".

```bash
# already done -- rebuild only if the pool needs regenerating
python scripts/human_audit.py build \
    --source shriram=task3_gan/shriram_dundigalla/outputs/pred_B2A
# each of us fills in our own ratings_rater*.csv, independently, without discussing
python scripts/human_audit.py score
```

**Rate independently and do not confer until both sheets are done** -- agreement between
two people who talked it through measures the conversation, not the images. Do not open
`KEY_do_not_open_until_scored.json`.

If you want your own translations pooled in alongside mine so neither of us can favour
our own work, send me your `pred_B2A` directory and I will rebuild with both sources;
the script takes `--source` more than once. That is the stronger design, but it needs
your images and it needs doing before either of us starts rating.

### File encoding is a measured variable — but check it under the course script

FID reads the Inception features of the bytes on disk, so the encoder is part of the
measurement rather than a packaging detail. Under the `real_stats.npz` protocol the
effect was large and monotonic -- identical weights and images, only the encoder changed:

| encoding | FID (old protocol, n=7,038) |
|---|---|
| PNG | 103.387 |
| JPEG q100 | 98.734 |
| JPEG q95 | 98.112 |
| JPEG q90 | 95.478 |
| JPEG q75 | 91.874 |

**Those numbers are on the retired scale and should not be quoted in the report.** The
course script preprocesses differently -- `Resize(299)` then `CenterCrop(299)` under
ImageNet normalization, against a reference recomputed from the real folders -- so the
size of the effect has not been re-measured. The principle survives the protocol change
even though the numbers do not: write JPEG rather than PNG, since the reference folders
are JPEG, and keep the quality setting fixed and stated so our two results stay
comparable.

**Use q95 and say so.** The old curve kept falling to q75, but only the first step is a
real format choice; everything after it buys score with visible degradation and a
worsening exchange rate. At q95 the difference from the model's exact output was 0.013
LPIPS, about 1.7% of the 0.786 LPIPS between different generated images, so it is
invisible. At q75 it is not, and a blinded audit would show it. Reporting that reasoning
is worth more than the extra FID.

## Report split (Section 6)

| Item | Owner |
|---|---|
| Team ownership statement | jointly written |
| Per-task comparison tables (both members side by side) | jointly built, from each member's `metrics_report.csv` |
| Per-task joint analysis — strengths, weaknesses, limitations, next steps | jointly written |
| Each member's `results.md` and `failure_analysis.md` | individual |
| Evidence links (curves, logs, checkpoint IDs) | each member supplies their own |
| Paper citations (Attention Is All You Need, TinyStories, CycleGAN) | either |
| Final PDF at `report/DATA266_Lab1_Report_Team_33.pdf` | jointly, pushed before the deadline |

## Open questions

**Resolved by the TA announcement of 28 Sep:**

- ~~Task 2 dataset, Yelp or IMDB?~~ **Yelp.** The IMDB reference is an error.
- ~~Task 3 submission format?~~ **`submission.csv`**, columns `ID, FID, MiFID`, one
  result row, values from the competition's provided evaluation script.
- ~~Which Inception does Kaggle's `real_stats.npz` use?~~ **Settled empirically** — see
  the verified-backbone table above. Self-FID 0.069 against the competition's own
  `mu_real`.
- ~~Is the deadline Oct 6 or Oct 12?~~ **It will not be moved.** We work to Oct 6, 6pm.

**Still open — worth one email to the ISAs:**

1. **Task 1 split unit.** Does "training (100K) and validation (10K)" count characters,
   sequences, or stories? We have both read it as **sequences** and documented that
   reading; if the intent was characters, both of our runs need re-scoping. Low risk,
   but cheap to confirm.
2. **Kaggle `ID` column.** The spec says `ID` must be numeric and there is one result
   row. Is `ID` a per-member identifier (so Shriram = 1, Tejas = 2), a submission
   counter, or arbitrary? This matters because both of us must submit individually
   under one team.
3. **Where is the "provided evaluation script"?** The announcement says our reported
   FID/MiFID must match it, but nothing is attached to the competition. We have
   reproduced the competition's own reference statistics to a self-FID of 0.069, so our
   scorer is almost certainly the right one — but if a script is posted on Canvas we
   should re-score against it before the final submission, since the leaderboard is
   self-reported and a mismatch would look like a misreport rather than a bug.

## Run `scripts/verify_submission.py` before you push anything

```bash
python scripts/verify_submission.py
```

107 checks across all three parts, about two minutes. Run it before every push and
certainly before the zip goes to Canvas. It is not a formality — on the run that
introduced it, it found a cross-entropy in the reproducibility manifest that was still
the old 100-batch figure, and a Part 2 ECE quoted as 0.0204 where the CSV says 0.020347.
Both are exactly the defect the brief's "all artifacts must agree" line is about, and
neither is visible by reading.

The design rule, if you add checks: **compare prose against the machine-written CSV, not
against a number you typed into the test.** A check that hardcodes the expected value
has to be maintained alongside the result and will eventually assert something stale.

Two traps worth knowing about, because they cost time here:

- **Do not grep for a banned module name.** `task1_llm/.../src/model.py` names
  `nn.MultiheadAttention` in its own docstring in order to state that it is not used, so
  a text search flags a clean file. The checks walk the AST and look at attribute
  accesses and imports, which is what actually determines whether something is used.
- **Do not glob for "the" training summary.** Smoke runs used to write beside the
  reported artifacts, so `next(glob("train_summary_*.json"))` could compare the results
  against a two-epoch smoke run and pass. `train.py --smoke` now writes to
  `outputs/smoke/` and `checkpoints/smoke/`, both gitignored, and the checks resolve the
  run tag from the `checkpoint` column of `metrics_report.csv` instead.

## Part 1 has GPU presets too — use them if Part 3 finishes early

The reported Part 1 run underfits: validation loss was still falling at epoch 12 and the
generalization gap is 0.035 nats. `task1_llm/shriram_dundigalla/src/config_gpu.yaml`
trains the same architecture for 30 epochs at a 256-character context
(9-15 h on MPS, timed; 20-35 min on a lab GPU), and `config_gpu_large.yaml` runs an
identical recipe at 5.2x the parameters.

Run the cheap arm first. A finished cheap arm beats two half-finished runs, and the pair
only means something together: arm 2 differs from arm 1 in the `model:` block and a
memory-forced batch-size change and nothing else, so the gap between them isolates
capacity. If you change anything else in one of them, the comparison stops being an
ablation and becomes two unrelated runs.
