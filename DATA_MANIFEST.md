# Data and configuration manifest

Every path is relative to the directory holding this file, and is what the YAML
configs already expect — no path in any config needs editing.

Built 2026-09-30 23:01 UTC.

## Datasets

| Path | Expected | Found | Size | Status |
|---|---|---|---|---|
| `task1_llm/data/tinystories_train_first20000.jsonl` | 1 file | 1 | 17.6 MB | included |
| `task2_sentiment/data/yelp_polarity_train.parquet` | 1 file | 1 | 238.2 MB | included |
| `task2_sentiment/data/yelp_polarity_test.parquet` | 1 file | 1 | 16.5 MB | included |
| `task3_gan/data/monet_jpg` | 300 images | 300 | 4.7 MB | included |
| `task3_gan/data/photo_jpg` | 7,038 images | 7,038 | 93.6 MB | included |
| `task3_gan/data/real_stats.npz` | 1 file | 1 | 34.4 MB | included |
| `task3_gan/data/photo_stats.npz` | 1 file | 1 | 88.5 MB | included |

`real_stats.npz` and `photo_stats.npz` are the competition's own reference
statistics for the Monet and photograph domains. **Neither is used to produce any
reported number.** The course evaluation script recomputes its reference from the
image folders instead, so these two files are kept only because they ship with the
competition data and because the experiment that established how they were built
is cited in `TEAM_PROTOCOL.md`. Neither is derived from generated images, so both
are inputs rather than outputs.

## Configuration files

| Path | Purpose | Included |
|---|---|---|
| `task3_gan/shriram_dundigalla/src/task3_cyclegan_unet.ipynb` | **Part 3 in full** — the notebook holds its own CONFIG dict, so there is no separate YAML | yes |
| `task1_llm/shriram_dundigalla/src/config_gpu.yaml` | **Part 1 GPU run** — 30 epochs at a 256-character context | yes |
| `task1_llm/shriram_dundigalla/src/config_gpu_large.yaml` | Part 1 capacity arm — 25.4M parameters, otherwise identical | yes |
| `task1_llm/shriram_dundigalla/src/config.yaml` | Part 1 as reported on MPS | yes |
| `task2_sentiment/shriram_dundigalla/src/config.yaml` | Part 2, all three models; runs on CUDA unchanged | yes |
| `requirements-gpu.txt` | Dependencies for the lab PC | yes |

## What is not in this bundle, and why

The repository is 4.6 GB. Almost all of that is of no use on a lab PC:

- `.git/` (209 MB) — history is not needed to train, and the machine gets wiped.
- Part 3 full-state checkpoints under `task3_gan/*/checkpoints/` — these carry Adam state for four networks so a run can resume, and they are the single largest thing in the tree. The checkpoint behind the submitted score, `ckpt_epoch060.pth`, is the one that matters; it is gitignored by size but travels in the submission zip, which `scripts/package_submission.py` assembles by globbing that directory rather than by expecting a particular filename.
- `task3_gan/*/outputs/pred_A2B/` and `pred_B2A/` (7,338 JPEGs) — the *output* of a run. Regenerate by rerunning the notebook's inference cells from the checkpoint.
- `Lab1_datasets_Team_33.zip` (362 MB) — a packaged copy of data that already travels unpacked under `task*/data/`.
- `__pycache__/`, `.venv/`, `.DS_Store`, `*.pyc`, `.ipynb_checkpoints/` — build artifacts, and `.pyc` files compiled for the wrong platform.
- `smoke/` and `seed_sweep/` output directories — throwaway pipeline checks.
- No credentials, tokens, API keys or Google Drive addresses: none exist in the repository, and the bundle is scanned for eight patterns after it is built.

Nothing excluded is needed to train, evaluate or score any part. The generated
images and the full-state checkpoints are *outputs* of a run, not inputs to one.

## The two bundles that travel beside the repository

Build either with `python scripts/package_submission.py --datasets` or `--checkpoints`.
Both unzip at the repository root and need no renaming. Neither is committed.

| Bundle | Size | Files | Contents |
|---|---|---|---|
| `Lab1_datasets_Team_33.zip` | 450 MB | 7,343 | `task1_llm/data/`, `task2_sentiment/data/`, `task3_gan/data/` |
| `Lab1_checkpoints_Team_33.zip` | 459 MB | 6 | every weight behind a reported number |

The datasets bundle is what the 28 September announcement asks for, and its link belongs
in the top-level `README.md`.

The weights bundle exists for a narrower reason. Part 1 and Part 2 checkpoints are
15–27 MB and live in the repository; Part 3's cannot. One fp32 generator is 217.6 MB
against GitHub's 100 MB per-file limit, and fp16 is 108.8 MB — still over, and it would
no longer reproduce the reported FID of 101.44, which would leave the weights disagreeing
with `metrics_report.csv`. So the bundle carries `G_M2P` and `G_P2M` extracted from
`ckpt_epoch060.pth` in fp32, verified tensor-by-tensor as bitwise identical to the
checkpoint. It deliberately omits the optimizer state, which is roughly two thirds of
the 1.3 GB file and is needed only to *resume* training.

**No reported number depends on this bundle.** `evaluate_local.py` scores the committed
images under `outputs/pred_A2B/` and `outputs/pred_B2A/`, never the checkpoint. The
weights are required only to generate images that do not already exist.

## Checking all of this on the lab PC

```bash
python scripts/verify_gpu_ready.py
```

It re-checks every path and count in the table above against what is actually on
disk, and exits non-zero if anything is missing — so trust it over this file,
which is a snapshot of build time.
