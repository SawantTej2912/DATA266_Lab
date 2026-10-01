# DATA266 Lab 1 — Team [Team Number]

**Members:** Tejas (`tejas/`), [Teammate] (`[teammate]/`)

## Contents

- **Task 1** — GPT from scratch (TinyStories) — `task1_llm/`
- **Task 2** — Yelp Polarity sentiment — `task2_sentiment/`
- **Task 3** — CycleGAN Monet ↔ Photo — `task3_gan/`

## Data

Datasets are not stored in git (size limits). Download them from [Google Drive link] and unzip so each task has this layout:

| Task | Expected location |
|---|---|
| Task 1 | `task1_llm/data/tinystories_hf/` |
| Task 2 | `task2_sentiment/data/yelp_review_polarity_csv/` or `task2_sentiment/data/yelp_polarity_hf/` |
| Task 3 | `task3_gan/data/monet_jpg/` + `task3_gan/data/photo_jpg/` |

## Setup

Install CUDA PyTorch first, then the rest of the requirements, so pip keeps the GPU build.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # macOS: plain `pip install torch torchvision`
pip install -r requirements.txt
```

## One-command smoke test (reproducibility)

A tiny end-to-end run of the whole notebook (a few iterations, small eval) to check the pipeline works.

**macOS / Linux**

```bash
cd task3_gan/tejas/src && SMOKE=1 jupyter nbconvert --to notebook --execute --output smoke_run.ipynb task3_cyclegan.ipynb
```

**Windows (PowerShell)**

```powershell
cd task3_gan\tejas\src; $env:SMOKE="1"; jupyter nbconvert --to notebook --execute --output smoke_run.ipynb task3_cyclegan.ipynb
```

Same pattern for `task1_llm/tejas/src/task1_llm.ipynb` and `task2_sentiment/tejas/src/task2_sentiment.ipynb`.

## Full runs

- Run the notebook from its `src/` folder without `SMOKE`.
- Training resumes from `checkpoints/last.pt` if interrupted.
- All settings live in the `CFG` cell; there are no hard-coded paths.

## Where results live

| What | Where |
|---|---|
| Code + outputs | `<task>/<member>/src/` |
| Metrics | `<task>/<member>/metrics_report.csv` (Task 3 also `full_metrics_report.csv`, `submission.csv`) |
| Plots / samples / predictions | `<task>/<member>/outputs/` |
| Weights | `<task>/<member>/checkpoints/` (resume checkpoints `last.pt` kept off-git: [Drive link]) |
| Raw logs (unedited) | `reproducibility/raw_logs/<task>/<member>/` |
| Manifests | `reproducibility/manifests/<task>/<member>/` |
| Write-ups | `results.md`, `failure_analysis.md` |
| Report | `report/DATA266_Lab1_Report_Team_[Team Number].pdf` |
