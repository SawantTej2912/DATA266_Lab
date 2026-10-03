# Task 1 — GPT-style LLM from scratch (TinyStories, character level)

**Member:** Shriram Dundigalla · **Team:** Lab Pair 33
**Run tag:** `gpt_char_tinystories_v1_20260928T204206Z`
**Checkpoint:** `checkpoints/gpt_char_tinystories_v1_20260928T204206Z.pt`
**Raw log:** `reproducibility/raw_logs/task1_gpt_char_tinystories_v1_20260928T204206Z.log`
**Hardware:** Apple M4, 24 GB unified memory, GPU via PyTorch MPS backend (torch 2.14.0) — **no CUDA device was used for any reported number.** The repository contains a CUDA fast path for the GPU-lab presets; it is gated on `device.type == "cuda"` and did not execute for this run.

Reproduce with:

```bash
python scripts/fetch_data.py --task tinystories --n-stories 20000
python task1_llm/shriram_dundigalla/src/train.py --config task1_llm/shriram_dundigalla/src/config.yaml
python task1_llm/shriram_dundigalla/src/evaluate.py --checkpoint <path printed by train.py>
```

---

## 1. Data preprocessing

The shared raw data is the first 20,000 TinyStories training rows
(`task1_llm/data/tinystories_train_first20000.jsonl`, 17.6M characters), streamed rather
than downloading the full ~2 GB corpus.

**Character-level tokenisation.** Vocabulary is built from the **training split only**
and comes to **98 symbols**, with index 0 reserved for `<unk>`. Building it from the
training text alone is deliberate: if a character appears only in validation, the model
genuinely has never seen it, and folding it into the vocabulary would hide that rather
than measure it. `char_to_idx` and `idx_to_char` are my own dictionaries built by
sorting the character set, so IDs are stable across runs.

**Two decisions that needed making, because the brief is ambiguous.**

*What "100K / 10K" counts.* Section 1.1.4 says to split into training (100K) and
validation (10K) without naming a unit. I read it as **sequences**, since step 1.1.2 is
what creates the sequence dataset that step 1.1.4 then splits. I use
**non-overlapping** windows of 128 characters, giving 100,000 training and 10,000
validation windows over 12.8M and 1.28M characters respectively. The alternative — a
stride-1 sliding window, which is what I used for HW4 — would have made "100K
sequences" mean 100,000 near-duplicate views of about 100K characters of text, so
twelve epochs would really have been ~1,200 passes over a very small corpus. Disjoint
windows make one epoch one honest pass over the data.

*How the split is drawn.* The split is over **whole stories**, not over the
concatenated character stream. Cutting one stream at a single offset leaks the text on
either side of the cut into both halves' windows, and with a sliding window that
leakage is systematic rather than negligible. 18,000 stories go to training and 2,000
to validation, shuffled with seed 9015, so no validation character appears in any
training window. Stories are joined with a blank line so the model has an explicit
document boundary to learn instead of one story running into the next mid-sentence.
(That separator turns out to cause one of my three failure cases — see
`failure_analysis.md`, case 3.)

Each item is `(tokens[t : t+128], tokens[t+1 : t+129])`, so the target is the input
shifted left by one and each row carries 128 next-character prediction problems.

## 2. Architecture

Decoder-only GPT, written from scratch. Nothing comes from `nn.Transformer`,
`nn.TransformerEncoderLayer`, `nn.MultiheadAttention`, or
`F.scaled_dot_product_attention`; the only `torch.nn` pieces used are `Linear`,
`Embedding`, `LayerNorm` and `Dropout`. The attention arithmetic is written out in
`src/model.py` so the head split and the causal mask are visible.

| Component | Choice |
|---|---|
| Blocks | 6 |
| Attention heads | 8 (head dim 32) |
| `d_model` | 256 |
| FFN inner dim | 1024 (4x), GELU |
| Context length | 128 characters |
| Embeddings | learned token (98 x 256) + learned absolute positional (128 x 256) |
| LM head | `Linear(256, 98)`, no bias, untied from the token embedding |
| Dropout | 0.1 on embeddings, attention weights, and both residual projections |
| **Parameters** | **4,822,016** |

**Why these choices.**

*Pre-LayerNorm, not the Post-LN of "Attention Is All You Need".* Post-LN renormalises
the residual stream at every layer, which is what makes deep Post-LN stacks need a long
warmup to avoid diverging early. Pre-LN puts the norm inside the residual branch,
leaving a clean identity path from input to output so gradients reach the early blocks
directly. The run bears this out — zero loss spikes and zero NaNs across 9,372 steps.

*Causal masking before the softmax.* A lower-triangular boolean buffer, applied with
`masked_fill(..., -inf)` **before** `softmax`, so masked positions get exactly zero
probability. Masking after the softmax would leave non-zero weight on future tokens and
then renormalise, which leaks information.

*Learned absolute positional embeddings.* Section 1.2.3 requires learnable positional
embeddings, which rules out sinusoidal and rotary encodings. This is also the reason
generation crops context to the last 128 tokens: positional embeddings exist for 128
positions and no more.

*8 heads at 32 dims each.* At `d_model=256`, 8 heads is the point where head dimension
(32) is still large enough for dot-product attention to be discriminative. 16 heads
would give 16 dims per head, where the `1/sqrt(d_head)` scaling leaves the scores too
noisy to sharpen.

*GPT-2's residual init scaling.* The two projections writing into the residual stream
are initialised at `std = 0.02 / sqrt(2 * n_layer)`, so residual-stream variance does
not grow with depth.

## 3. Training

| Hyperparameter | Value | Reason |
|---|---|---|
| Optimiser | AdamW, betas (0.9, 0.95) | 0.95 rather than 0.999 is the standard GPT setting; shorter second-moment memory suits a short run |
| Peak LR | 3e-4 | |
| Schedule | linear warmup over first 3% of steps (281 of 9,372), then cosine decay to 0.1x peak | |
| Weight decay | 0.1 on matmul weights only (25 tensors); 0 on LayerNorm gains, biases and embeddings (52 tensors) | decay on normalisation scales and unused embedding rows pulls them to zero for no benefit |
| Batch size | 128 sequences = 16,384 characters | |
| Epochs | 12 (brief requires >= 10) | |
| Gradient clipping | 1.0, norm measured **pre-clip** so the logged number is the raw gradient | |
| Loss | cross-entropy over the 98-symbol vocabulary | |
| Seed | 9015 | |

Warmup is required by section 1.3.2, but it earns its place even with Pre-LN blocks:
Adam's second-moment estimate is near zero for the first few steps, so a full-size
learning rate there produces an enormous effective step before the optimiser state has
settled. The logged first step confirms the ramp — `lr=1.07e-06` at step 0.

Batch size 128 was chosen empirically. I benchmarked 128/256/512 with and without bf16
autocast on MPS and throughput was flat at 31K–43K tokens/sec across all six
combinations, because this model is matmul-bound at roughly 2 TFLOP/s fp32 — the
measured ceiling of the M4 GPU. Since larger batches bought nothing, I kept the
smallest, which gives the most optimiser steps per epoch.

## 4. Results

Loss curves: `outputs/loss_curves_gpt_char_tinystories_v1_20260928T204206Z.png`
Per-step loss and gradient norm: `outputs/stability_gpt_char_tinystories_v1_20260928T204206Z.png`
All metrics in one row: `metrics_report.csv`

| Metric | Train | Validation |
|---|---|---|
| Cross-entropy (nats/char) | 0.7089 | **0.7441** |
| Perplexity | 2.0317 | **2.1045** |
| Bits per character | 1.0227 | **1.0735** |
| Top-1 next-char accuracy | 0.7728 | **0.7639** |
| Generalization gap | — | **0.0352** |

**Sample sizes.** The training column is the **whole training split** — 781 batches ×
128 = 99,968 of 100,000 windows — and the validation column is all 10,000 validation
windows. An earlier version of this table computed the training column from the first
100 batches, 12.8% of the split, and reported it as the training loss; the full pass
moves cross-entropy from 0.7110 to 0.7089 and the gap from 0.0331 to 0.0352. Small, but
the number now means what its label says. `metrics_report.csv` carries
`train_eval_windows` so the sample size travels with the metric.

The 32 missing windows are `drop_last=True` on the training loader (100,000 = 781×128 +
32), kept rather than switched off for two reasons: a ragged final batch of 32 would
contribute a roughly 4× noisier gradient at full learning rate at the end of every
epoch, and because the loader shuffles every epoch the dropped 32 are a *different* 32
each time, so nothing is permanently excluded — the chance a given window is never seen
across 12 epochs is (32/100000)¹². The post-hoc training metrics use the same
`drop_last=True` loader, so the figure describes exactly the windows the model trained on.

| Generation metric (T = 0.8) | Single draw | Mean ± SD over 15 draws |
|---|---|---|
| Distinct-1 (word level) | 0.707 | 0.714 ± 0.046 |
| Distinct-2 (word level) | 0.988 | 0.957 ± 0.030 |
| Distinct-3 (word level) | 1.000 | 0.991 ± 0.014 |
| Repeated 4-gram rate (word level) | 0.000 | 0.002 ± 0.004 |
| Training-vocabulary word rate | 0.988 | 0.993 ± 0.008 |

The 15 draws are 5 prompts × 3 seeds, in `outputs/generation_sweep_*.csv`. The prompts
vary what the model has to do rather than merely differing: a story opener, a
mid-sentence noun phrase, a causal clause demanding a reason, a dialogue opening, and a
plural subject that constrains agreement.

**Reporting a single draw was actively misleading at low temperature, and this is the
clearest thing the sweep bought.** Greedy decoding scored a repeated-4-gram rate of
**0.060** on one sample; over 15 draws it is **0.235 ± 0.319**. The single sample
understated the degenerate-looping failure mode by a factor of four, and the standard
deviation larger than the mean says the behaviour is bimodal — greedy decoding either
escapes the loop or collapses into it, depending on the prompt. No single number
describes that, which is why the full sweep is reported:

| Temperature | Distinct-2 | Repeated 4-gram | Training-vocabulary word rate |
|---|---|---|---|
| 0.0 (greedy) | 0.690 ± 0.311 | **0.235 ± 0.319** | 1.000 ± 0.000 |
| 0.5 | 0.881 ± 0.059 | 0.016 ± 0.023 | 0.998 ± 0.005 |
| 0.8 | 0.957 ± 0.030 | 0.002 ± 0.004 | 0.993 ± 0.008 |
| 1.0 | 0.973 ± 0.028 | 0.001 ± 0.003 | 0.992 ± 0.011 |
| 1.2 | 0.983 ± 0.016 | 0.001 ± 0.003 | 0.970 ± 0.021 |

**The repetition penalty fixes the loop without the fluency cost of raising
temperature.** Five decoding strategies at T = 0.8, same 15 draws each
(`outputs/decoding_comparison_*.csv`):

| Decoding | Distinct-3 | Repeated 4-gram | Training-vocabulary word rate |
|---|---|---|---|
| plain | 0.989 ± 0.017 | 0.004 ± 0.010 | 0.991 ± 0.007 |
| top-k 40 | 0.988 ± 0.018 | 0.003 ± 0.007 | 0.994 ± 0.006 |
| top-p 0.9 | 0.983 ± 0.013 | 0.004 ± 0.009 | 0.996 ± 0.006 |
| **repetition penalty 1.15** | **1.000 ± 0.000** | **0.000 ± 0.000** | 0.994 ± 0.008 |
| top-p 0.9 + penalty 1.15 | 0.997 ± 0.006 | 0.000 ± 0.000 | **0.997 ± 0.005** |

A penalty of 1.15 over a 64-character window eliminates repeated 4-grams entirely
across all 15 draws *and* raises the training-vocabulary word rate slightly, so it is not trading
fluency for diversity — unlike temperature, which buys diversity at 1.2 by dropping the
training-vocabulary word rate to 0.970. The window matters: at character level the common letters
recur constantly by necessity, so penalising every character ever emitted would suppress
`e` and `t` and damage the text.

| Stability and cost | Value |
|---|---|
| NaN / inf count | 0 |
| Loss spikes | 0 |
| Gradient norm, mean / max | 0.701 / 11.128 (max is step 0, pre-warmup) |
| Parameter count | 4,822,016 |
| Training throughput | 24,990 tokens/sec |
| Generation throughput | 387 tokens/sec |
| Total training time | 6,144 s (102.4 min) |
| Peak GPU memory | 80.0 MB |

Diversity is reported at **word level**; character-level figures are in
`outputs/generation_metrics_*.csv` for completeness. At character level distinct-1 is
bounded by the 98-symbol vocabulary and saturates near 1.0 for any text of reasonable
length, so it cannot tell you whether the model is repeating itself. Word level can.

I added **training-vocabulary word rate** beyond the required list. Section 1.4 asks for failure
analysis, and the required diversity metrics score my temperature-0.8 sample as
essentially flawless (distinct-3 = 1.000, repeated-4-gram = 0.000) while it in fact
contains the invented word "tost". Training-vocabulary word rate is the metric that
catches the failure mode a character-level model is uniquely prone to.

To be precise about what it measures: the reference set is the vocabulary of the 20,000
TinyStories training documents, not an English dictionary. A genuine English word absent
from that corpus counts against the model, and a typo present in it counts for it.
TinyStories is written in deliberately simple vocabulary so the two are close in
practice, but the metric is a statement about the training data and not about English,
which is why it is no longer called "real-word rate".

## 5. Reading of the result

**The model is underfitting, not overfitting.** A generalization gap of 0.035 nats with
validation loss falling monotonically through epoch 12 says the run stopped while there
was still signal left to extract. Every drop of validation loss in the last four epochs
was 0.008 or smaller, so it was flattening, but it had not turned.

That matters for what the numbers mean. At 1.07 bits per character the model is well
short of a word-level model on the same corpus, and the constraint is capacity and
context rather than data or optimisation: 4.8M parameters and a 128-character window
against 12.8M characters of training text. The failure analysis supports the same
reading — case 3 is directly caused by the model losing sight of its own protagonist
once he passes out of the 128-character window.

**What I would try next**, in the order I would try it:

1. Widen the context to 256 characters. Case 3 is a context failure and this addresses
   it directly; the cost is quadratic in attention but the model is small.
2. Go deeper or wider (8 blocks, or `d_model=384`) and train past 12 epochs, since
   validation loss had not turned.
3. Add a repetition penalty or ban greedy decoding outright. Case 1 is purely a decoder
   problem and costs nothing to fix.

Item 3 is **done** and reported above: a repetition penalty of 1.15 removes repeated
4-grams entirely without costing fluency, so the decoder-side failure is closed. Items 1
and 2 are configured but not run, for a reason worth stating precisely rather than
calling it future work.

### Why items 1 and 2 are configured rather than run

Both are set up as GPU-lab presets, `src/config_gpu.yaml` and
`src/config_gpu_large.yaml`, and both were smoke-tested end to end on this Mac so a
config error cannot waste lab time. What stops them running here is arithmetic, not
choice. Each config was timed on this M4 twice via `--smoke`, against the 768M tokens a
30-epoch run at block 256 requires:

| Arm | Config | Parameters | Tokens/sec on MPS | 768M tokens on MPS |
|---|---|---|---|---|
| context + epochs | `config_gpu.yaml` | 4,854,784 | 13,939 and 24,255 | 9–15 hours |
| + capacity | `config_gpu_large.yaml` | 25,451,520 | 6,409 and 8,420 | 25–33 hours |

The ranges are wide, and deliberately reported as ranges: a smoke run is 15 steps, so
process startup and the first-step graph compile are a large share of the wall clock and
the throughput figure is not a benchmark. Quoting a single number here would imply a
precision the measurement does not have. It does not need that precision to decide
anything — the answer is "hours, not minutes" either way, which is why the reported run
is the 12-epoch, block-128 one. The timing logs are retained in
`reproducibility/raw_logs/` under `*_smoke_*`; no metric in this document comes from
them.

The two presets are built to answer a question rather than just to be bigger.
"Underfitting" has two causes that call for opposite fixes — too little training or too
little capacity — and a single scaled-up run cannot separate them. `config_gpu.yaml`
trains the *same* architecture for longer at a longer context; `config_gpu_large.yaml`
differs from it in the `model:` block and nothing else bar a batch-size reduction forced
by memory. So the gap between the two arms is attributable to capacity, and the gap
between the cheap arm and the reported run to training length and context. I verified
the configs differ only in those five fields by diffing them as parsed YAML rather than
trusting that I copied them correctly.

Two honest caveats. The cheap arm has 32,768 more parameters than the reported run, not
because capacity was changed but because the learned positional embedding is
`block_size x d_model` and the context doubled; it is 0.7% of the model. And the large
arm halves the batch size to fit memory, so its comparison is capacity-plus-batch-size
rather than capacity alone.

The trainer gained a CUDA fast path for this: bf16 autocast on the forward pass, TF32
matmuls, `cudnn.benchmark`, pinned memory and persistent workers. bf16 rather than fp16
because it keeps fp32's exponent range, so there is no loss scaler and no route to a
silent `inf` in a multi-hour run. **Evaluation always runs in fp32**, so the reported
metrics stay comparable with this run's rather than being measured in a different
precision. All of it is gated on `device.type == "cuda"`, and the MPS smoke run still
logs `amp=off (fp32)`, so the numbers in this document are unaffected by code that only
executes on hardware they were not produced on.

## 6. Constraint compliance

- No prebuilt Transformer or attention modules — verified by inspection of `src/model.py`.
- Character-level tokenisation with my own `char_to_idx` / `idx_to_char`.
- Causal masking applied before the softmax.
- Learnable token **and** positional embeddings.
- LM head projecting to vocabulary size for next-character prediction.
- Cross-entropy loss, LR warmup and scheduling, 12 epochs (>= 10 required).
- Train and validation loss curves plotted and reported.
- Text generated by both greedy decoding and temperature sampling.
- Three failure cases analysed in `failure_analysis.md`.
- No hard-coded personal paths: every path is config-driven and relative to the repo root.
