# Reproducibility manifest — Shriram Dundigalla

Section 5 of the brief asks for the environment, the exact package versions, and a
mapping from each reported number back to the checkpoint that produced it. This is that
mapping. Every figure quoted in `results.md` and in the report traces to a row here.

## Environment

| | |
|---|---|
| Machine | Apple M4, 24 GB unified memory |
| Accelerator | Apple MPS (Metal) — **no CUDA was used anywhere in this project** |
| OS | macOS (Darwin 25.6.0) |
| Python | 3.11.9 |
| Environment file | [`requirements.txt`](../../requirements.txt) |

Package versions, as installed for every run below:

```
torch==2.14.0            torchvision==0.29.0      numpy==2.4.6
scipy==1.17.1            scikit-learn==1.9.1      pandas==3.0.5
matplotlib==3.11.1       pillow==12.3.0           nltk==3.10.3
pyyaml==6.0.3            datasets==5.0.1          pyarrow==25.0.1
lpips==0.1.4             nbconvert==7.17.1        ipykernel==7.3.0
kaggle==2.2.4
```

Device selection is automatic (CUDA, then MPS, then CPU) and every run records the
resolved hardware string in its log, so a re-run on different hardware is self-documenting
rather than silently different.

## Seeds

One seed per run, set for `random`, `numpy` and `torch` (including the MPS generator).
Mine is **9015**. Note two run-specific details that matter for exact reproduction:

- **Part 3 domain pairing** is seeded per `(seed, epoch, index)` rather than from a shared
  generator, because with `num_workers > 0` each DataLoader worker holds its own copy of
  the dataset; a single shared RNG would emit identical draws in every worker and the
  pairing would then depend on worker scheduling.
- **Inference is fully deterministic.** `SingleDomainDataset` disables the random crop and
  flip, so a given checkpoint reproduces its generated images exactly.

## Part 1 — character-level GPT

| | |
|---|---|
| Checkpoint | `task1_llm/shriram_dundigalla/checkpoints/gpt_char_tinystories_v1_20260928T204206Z.pt` |
| Config | `task1_llm/shriram_dundigalla/src/config.yaml` |
| Raw log | `reproducibility/raw_logs/task1_gpt_char_tinystories_v1_20260928T204206Z.log` |
| Metrics | `task1_llm/shriram_dundigalla/metrics_report.csv` |
| Notebook | `task1_llm/shriram_dundigalla/src/task1_gpt_from_scratch.ipynb` |
| Reproduce | `python task1_llm/shriram_dundigalla/src/train.py` |

Parameters 4,822,016; vocabulary 98; 12 epochs; 6,144 s; peak memory 80 MB.
Reported: train CE 0.7089 / val CE 0.7441, val perplexity 2.1045, val bpc 1.0735,
val top-1 0.7639, generalization gap 0.0352. Zero NaN/Inf, zero loss spikes.
The train-side figures cover all 99,968 training windows (100,000 less the 32 that
`drop_last=True` trims), not a sample of batches; `metrics_report.csv` records the
window count in `train_eval_windows` so the sample size cannot drift from the metric
again.

## Part 2 — Yelp Polarity

Config `task2_sentiment/shriram_dundigalla/src/config.yaml`; raw log
`reproducibility/raw_logs/task2_20260929T033024Z.log`; metrics
`task2_sentiment/shriram_dundigalla/metrics_report.csv`; notebook
`task2_yelp_sentiment.ipynb`. Reproduce with
`python task2_sentiment/shriram_dundigalla/src/train.py`.

All three models were trained on the same balanced 100,000-review subsample and scored on
the **full official 38,000-row test split**.

| Model | Checkpoint | Accuracy | Macro-F1 | Params |
|---|---|---|---|---|
| `baseline_bow` | `checkpoints/baseline_bow.pt` | 0.9245 | 0.9245 | 3,873,281 |
| `exp1_bilstm` | `checkpoints/exp1_bilstm.pt` | 0.9334 | 0.9334 | 6,605,953 |
| `exp2_textcnn` | `checkpoints/exp2_textcnn.pt` | 0.9344 | 0.9344 | 5,087,745 |

## Part 3 — CycleGAN

The architecture here is a **U-Net**, not a ResNet. An earlier ResNet-9 attempt exists in
the git history (commit `230337c` removed it) and is not part of this submission: it
duplicated the generator architecture the other team member used, and it was scored under
a FID convention the course evaluation script does not use, so neither its number nor its
architecture could stand next to a teammate's.

### Model

ResNet is replaced by a U-Net of depth 8 at `ngf=64`: 54,404,099 parameters per
generator, 2,763,841 per 70x70 PatchGAN discriminator, **114.34M** trainable in total.
Losses are LSGAN (MSE) adversarial, cycle L1 at lambda 10, identity L1 at lambda 5.
Adam, lr 2e-4, betas (0.5, 0.999), constant for 50 epochs then linear decay to zero over
the next 50. Batch size 1 at 256x256, resize-286-then-random-crop-256 with horizontal
flip, 50-image replay buffer per discriminator, Normal(0, 0.02) initialisation, AMP.

An epoch is **1000 steps**, so the 100-epoch run is 100,000 generator updates. Epoch
numbers are therefore not comparable to a teammate's run at a different steps-per-epoch;
compare total steps or wall clock.

### Run 1 — reported

| | |
|---|---|
| Notebook | `task3_gan/shriram_dundigalla/src/task3_cyclegan_unet.ipynb` |
| Checkpoint | `checkpoints/ckpt_epoch060.pth` (backup copy verified byte-identical) |
| Logs | `logs/metrics_per_epoch.csv`, `logs/train.log` (unedited copy at `reproducibility/raw_logs/task3_cyclegan_unet_run1.log`) |
| Metrics | `metrics_report.csv`, `full_metrics_report.csv` |
| Hardware | NVIDIA A100 (Colab), 2.77 h, 99.6 s/epoch, 10.04 img/s |
| Reported result | **FID 101.44, MiFID 0.4172**, averaged over both directions |
| Kaggle rank | **20** |

Per direction: FID 103.92 Monet->Photo and 98.95 Photo->Monet; MiFID 0.4241 and 0.4104.

**Which `submission.csv`, and why it matters.** The notebook's run-1 and run-2 scoring
cells both write to the same path, so the copy in the Colab Drive root is whichever cell
ran last -- on 2 Oct that was run 2 (101.512 / 0.41423), not the rank-20 run-1 result.
The committed file is the run-1 output, 101.43782932433302 / 0.417223796248436, which is
also preserved on Drive as `BACKUP_run1_unet/submission_run1.csv`. The discrepancy is
caught mechanically rather than by eye: `verify_submission.py` recomputes the mean of the
two per-direction FIDs and rejects a `submission.csv` that disagrees with it.

**Which checkpoint, and why.** Training ran the full 100 epochs but `ckpt_epoch060.pth`
scored best, so it is the one submitted. The last 40 epochs did not improve the graded
metric even though cycle loss was still falling, which is recorded in
`failure_analysis.md` as the reason training loss was not usable for model selection.

### Runs 2 and 3 — reported as negative results

| Run | Change | FID (avg) | Grad norm mean / max |
|---|---|---|---|
| 1 | baseline | **101.44** | 27.0 / 112.5 |
| 2 | loss-weight tuning, 80 epochs | 101.51 | — |
| 3 | DiffAugment on all discriminator inputs, 80 epochs | 112.97 | 137.0 / 640.5 |

No run produced a non-finite loss. Run 3 is kept because an 11-point regression with a
5x rise in gradient norm is the evidence for the failure analysis, not a discard.

### Reproducing the reported number

```bash
python task3_gan/shriram_dundigalla/evaluate_local.py \
    --real-monet task3_gan/data/monet_jpg \
    --real-photo task3_gan/data/photo_jpg \
    --gen-a2b    task3_gan/shriram_dundigalla/outputs/pred_A2B \
    --gen-b2a    task3_gan/shriram_dundigalla/outputs/pred_B2A
```

Training itself is the notebook, run top to bottom in Colab on an A100.

## Verification artifacts

These exist because the numbers above would otherwise be unfalsifiable.

| What is verified | How | Result |
|---|---|---|
| The scorer is the graded one | `evaluate_local.py` is the course evaluation script with its paths as arguments | ImageNet normalization, reference recomputed from the real folders, n = 300, both directions averaged |
| The scorer is correct | run it with the real folders passed as both real and generated | FID -0.000, MiFID 0.0000, as an exact implementation must return |
| The reported score is reproducible from the committed artifacts | re-score `outputs/pred_A2B` and `outputs/pred_B2A` locally on CPU | **101.389 / 0.41716** against the submitted 101.438 / 0.41722 — a 0.05 FID gap, consistent in both directions, attributable to CPU vs A100 float arithmetic rather than to the images; a wrong or re-encoded set moves FID by several points |
| The submitted images correspond to the source domains | compare filename stems against `data/monet_jpg` and `data/photo_jpg` | `pred_A2B` 300/300 matches the Monet stems, `pred_B2A` 7,038/7,038 matches the photo stems, no stale or extra files |
| Reported stability figures come from the log, not from memory | recompute from `logs/metrics_per_epoch.csv` | grad norm mean 27.0 / max 112.5, NaN 0, 99.6 s per epoch, 2.77 h total — all identical to `metrics_report.csv` |
| The checkpoint is a complete CycleGAN at the stated epoch | load `ckpt_epoch060.pth` and list its keys | `epoch=60`; two generators (17 tensors each), two discriminators (7 each), three optimizers, three schedulers, AMP scaler. 17 and 7 are exactly what these classes produce given `InstanceNorm2d(affine=False)` |
| Which FID convention the shipped `.npz` uses | self-FID of the real photos against `photo_stats.npz` under [-1,1] scaling | 0.0106 — confirms the cache is on a *different* scale from the graded metric, which is why it is not used |
| Generator / discriminator size | instantiate both classes from the notebook and count parameters | 54,404,099 and 2,763,841; output shapes (1,3,256,256) and (1,1,30,30) |
| Submitted value is not one direction | `verify_submission.py` recomputes the mean of the two directions | matches `submission.csv` to 0.01 |
| Causal masking (Part 1) | rewrite tokens 16-31, check logits 0-15 unchanged | bit-identical |

## Known deviations

- Part 2 trains on a documented 100,000-review subsample rather than all 560,000, for
  compute reasons. Scoring is always on the full official test split, so the comparison
  between models is unaffected.
- Part 3 trains for 100,000 generator updates rather than the paper's schedule, a
  wall-clock limit on a Colab session. FID stopped improving at 60,000 updates, so
  further training was not obviously the binding constraint.
- The course provides an evaluation script, and it is the authority for the reported
  number. It does **not** use the shipped `real_stats.npz`; it recomputes reference
  statistics from the image folders under ImageNet normalization and averages the two
  directions. A FID produced by any other convention is on a different scale and is not
  comparable to the leaderboard, so none is reported here.
- The human audit (30 blinded samples plus 5 real-Monet controls, two raters) is
  complete: style 3.37, content 3.83, artifacts 3.63, with quadratic-weighted Cohen's
  kappa 0.7925 / 0.6863 / 0.8047 and unweighted 0.2239 / 0.2975 / 0.5205. Both variants
  are recorded because within-1-point agreement is 100% on every dimension while exact
  agreement is 43-66%, which is what drives them apart. The controls did not separate
  from the generated images on style (gap 0.033), so they are reported as a flat-lining
  check rather than as evidence of realism.
