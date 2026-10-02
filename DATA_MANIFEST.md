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

## Checking all of this on the lab PC

```bash
python scripts/verify_gpu_ready.py
```

It re-checks every path and count in the table above against what is actually on
disk, and exits non-zero if anything is missing — so trust it over this file,
which is a snapshot of build time.
