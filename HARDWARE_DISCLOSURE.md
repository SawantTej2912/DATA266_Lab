# Hardware disclosure

Every number in this project was produced on one of the machines below. Which machine
produced which number is recorded per row in each `metrics_report.csv` (`device_class`,
`device` or `hardware` column) and in full in the `runtime_*.json` written beside every
run's artifacts.

**No results table in this project mixes devices without labelling them.** The
`verify_gpu_ready.py` check "metrics_report.csv records the device" enforces this, and
fails if a single CSV contains rows from more than one device class.

---

## Machine A — development laptop (Parts 1 and 2 as submitted)

| Field | Value |
|---|---|
| Label used in tables | `Apple MPS` |
| GPU | Apple M4 integrated GPU, via the PyTorch MPS backend |
| GPU memory | none dedicated — 24 GB unified with the CPU |
| CUDA version | not applicable; no NVIDIA hardware |
| PyTorch | 2.14.0 |
| Operating system | macOS 26.7, arm64 |
| CPU | Apple M4 |
| RAM | 24 GB unified |
| Python | 3.11.9 |
| Precision | fp32 throughout |
| `num_workers` | Part 1: 0 · Part 2: 0 · Part 3: 4 |

### Measured on machine A

| Part | Duration | Peak memory | Throughput |
|---|---|---|---|
| 1 — character GPT, 12 epochs | 47.3 min | 80 MB reported by MPS | ~31,000 tokens/s |
| 2 — three models end to end | 31.6 min | 1,240 MB | 1,240–3,900 examples/s by arm |

Part 3 is absent from this table on purpose: no part of the submitted Part 3 result was
produced on machine A. It is a machine-B result end to end, and machine B is described
below.

One caveat about the figures above. The MPS peak-memory numbers come from
`torch.mps.current_allocated_memory()`, which reports allocator state on a unified pool
and is not comparable to CUDA's `max_memory_allocated()`. The two columns should not be
read against each other.

---

## Machine B — Colab A100 (all of Part 3 as submitted)

Part 3 was trained and scored on a Google Colab A100 instance, not on the GPU lab PC
that earlier drafts of this document anticipated. Every Part 3 number in this
repository — FID, MiFID, KID, LPIPS, the timings, the memory figure — is a machine-B
number, and `task3_gan/shriram_dundigalla/metrics_report.csv` labels all 43 rows
`NVIDIA A100 (CUDA)` so that the no-mixing check has something to read.

| Field | Value |
|---|---|
| Label used in tables | `NVIDIA A100 (CUDA)` |
| GPU | NVIDIA A100-SXM4 |
| GPU count | 1 |
| Precision | fp16 autocast with a `GradScaler` for training; fp32 for all evaluation |
| Training duration | 2.77 h for 100 epochs x 1,000 steps (99.6 s/epoch) |
| Throughput | 10.04 images/s |
| Peak GPU memory | 1,882 MB, measured during inference |
| Seed | 42 — Parts 1 and 2 use 9015; this run did not |
| `num_workers` | 2 |

Three of those rows need their scope stated rather than assumed. The peak-memory figure
was captured in the inference pass, not during training, so it is a lower bound on what
the run actually needed and should not be read as a training footprint. The duration
covers the single reported run; the two ablation runs beside it are additional time not
counted here.

And the precision row is split for a reason. Training ran under
`torch.amp.autocast('cuda')` with a `GradScaler`, set by `use_amp: true` in the
notebook's CONFIG — fp16 rather than the bf16 the precision policy below prefers,
because that is what the notebook was written with. Scoring does not autocast at all:
`evaluate_local.py` runs the Inception forward pass in fp32, so no reported metric
depends on the training precision even though the training precision was mixed.

Parts 1 and 2 were never rerun on CUDA. Their reported numbers remain machine-A results,
which is why this repository's tables carry a device label at all.

### Why the fields are thinner than machine A's

Colab does not expose a stable host to interrogate, and the notebook recorded the device
name, the timings and the memory figure rather than a full `runtime_*.json` block. The
fields that are present are the ones the notebook measured and printed. Rather than
filling in the rest from what a Colab A100 instance *typically* reports, they are
omitted — a plausible-looking CPU model or driver version that nobody measured is worse
than an absent row, because it reads as evidence.

---

## Precision policy

fp32 is the default everywhere. Mixed precision is opt-in through `train.amp: true`,
which is set in the Part 3 GPU config and both Part 1 GPU configs, and is honoured only
when the hardware genuinely supports it:

```
bf16_ok(device) = device is CUDA
                  and torch.cuda.is_bf16_supported()
                  and compute capability major >= 8
```

The capability check is not redundant. `is_bf16_supported()` answers `True` on some
pre-Ampere cards where bf16 is emulated, and emulated bf16 is slower than the fp32 it
replaced — so the check prevents a "speedup" that is actually a slowdown *and* a change
to the arithmetic. When mixed precision is requested but unavailable, the run continues
in fp32 and the log line reads
`precision=fp32 (amp requested; bf16 unavailable on cuda)`.

bf16 rather than fp16, where it is used: bf16 has the same exponent range as fp32, so it
needs no gradient scaler and cannot silently produce an `inf` partway through a
multi-hour run. The cost is mantissa bits, which these models do not need at these loss
values. Autocast wraps the forward pass only; parameters stay fp32 and gradients
accumulate in fp32, which is what makes this safe without a scaler.

**Evaluation always runs in fp32**, in all three parts, so no reported metric depends on
the training precision. This is what makes an MPS-trained and a CUDA-trained model
comparable at all — the weights differ, but the measurement of them does not.

---

## Determinism

Seeding is unconditional: `random`, `numpy`, `torch`, `torch.cuda` and `torch.mps` are
all seeded from the single `seed` field in each config. That field is 9015 for Parts 1
and 2 and **42 for Part 3**, which was written as a standalone notebook before the
repository convention existed. The value does not matter; having one field that every
generator reads from does. Part 3's
domain-B pairing uses SplitMix64 over integers rather than Python's `hash()`, which is
not guaranteed stable across processes.

`--deterministic` additionally pins cuDNN's algorithm selection. It is **off by default**
because it forbids the autotuner and costs real throughput on convolutions, which
matters for a CycleGAN run sized to a fixed wall-clock budget.

This project does **not** claim bit-exact reproducibility of a GPU run. Non-deterministic
cuDNN kernel selection, atomic accumulation order, and TF32 on the fp32 ops outside
autocast all mean two runs with the same seed can differ in the last few decimal places.
What is claimed, and what the seeding delivers, is that the data splits, the
initialisation, the shuffling order and the augmentation draws are identical — so a rerun
lands in the same place to within run-to-run noise, and the Part 2 seed sweep exists to
put a number on how large that noise is.
