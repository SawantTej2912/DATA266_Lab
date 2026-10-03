#!/usr/bin/env python3
"""Re-train Part 2's three models across several seeds and report the spread.

    python scripts/seed_sweep_task2.py --seeds 9015,17,2718,31415,1234 --require-cuda
    python scripts/seed_sweep_task2.py --seeds 9015,17 --dry-run

WHY THIS EXISTS. Part 2's headline finding is that the BiLSTM and the TextCNN are
*statistically indistinguishable*: their bootstrap confidence intervals overlap and a
paired McNemar test gives p = 0.396. That conclusion is currently supported only from
within a single training run of each. Bootstrap CIs quantify sampling error in the test
set, and McNemar quantifies disagreement between two fixed sets of predictions; neither
says anything about **run-to-run variance from initialisation and shuffling order**.

So the claim has a hole in it, and it is the hole that matters. If retraining the same
architecture on a different seed moves macro-F1 by more than the 0.001 that separates
the two architectures, then the two models were never distinguishable by this experiment
and the correct conclusion is stronger than "no evidence of a difference" — it is "the
experiment cannot resolve a difference this small". If the seed spread is much *smaller*
than 0.001, that is a different and more interesting result: the tie is real rather than
an artefact of noise.

This is the single highest-value thing a GPU can do for Part 2, and note what it is
*not*: it is not re-running the reported result faster. Re-running on a different
backend produces different numbers for the same code, which would mean regenerating
`results.md`, every figure, the notebook, the manifest and the report — churn with no
new information. A seed sweep is purely **additive**: the reported single-seed result
stays exactly as it is, and this adds the variance estimate it is missing.

That churn has since been accepted deliberately -- the plan is to re-run all three
models on CUDA so that every Part 2 number comes from one backend. It changes what the
sweep's first seed means rather than what the sweep does: with seed 9015 run first and
on the same device, arm one of the sweep *is* the headline result, so the published
number sits inside its own variance estimate instead of beside it.

ISOLATION. Each seed writes to its own checkpoint and output directory. `train.py`
writes `<model>.pt` under fixed names, so seeds sharing a directory would silently
overwrite each other and the sweep would score the same weights several times. This is
the same trap that bites any sweep sharing one checkpoint filename, and it is not
hypothetical.

COST. On MPS the three models take 26 s, 211 s and 2,319 s, so one seed is about 43
minutes and five seeds is over three hours. On a lab GPU expect roughly 5-10 minutes per
seed, since the BiLSTM is what dominates and cuDNN's fused LSTM kernels are much faster
than the MPS path. Run it on CUDA.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER = "shriram_dundigalla"
BASE_CFG = REPO_ROOT / "task2_sentiment" / MEMBER / "src/config.yaml"
OUT_CSV = REPO_ROOT / "task2_sentiment" / MEMBER / "outputs/seed_sweep.csv"

MODELS = ["baseline_bow", "exp1_bilstm", "exp2_textcnn"]
# The metrics worth a spread. Accuracy and macro-F1 are the headline claims; the two
# ECEs because calibration is the property most likely to be seed-sensitive.
METRICS = ["accuracy", "f1_macro", "roc_auc", "mcc",
           "ece_predicted_class_confidence", "ece_positive_class"]


def seed_config(seed: int, work: Path) -> Path:
    """A config for one seed, with isolated output and checkpoint directories."""
    cfg = yaml.safe_load(BASE_CFG.read_text("utf-8"))
    cfg["seed"] = seed
    run = work / f"seed_{seed}"
    for key in ("checkpoints", "outputs"):
        d = run / key
        d.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = str(d.relative_to(REPO_ROOT))
    # Also the report, or every arm overwrites the committed metrics_report.csv in turn
    # and the sweep ends with the last seed masquerading as the reported result.
    cfg["paths"]["report_dir"] = str(run.relative_to(REPO_ROOT))
    # The preprocessing cache is deliberately NOT isolated. Preprocessing does not
    # depend on the seed -- vocabulary and cleaning are fixed by config -- so sharing it
    # means it runs once instead of once per seed, and guarantees every arm sees exactly
    # the same tokens. Only the model's initialisation and shuffling order vary, which
    # is precisely the variance this sweep is trying to measure.
    path = run / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), "utf-8")
    return path


def run_seed(seed: int, work: Path, dry: bool, smoke: bool = False,
             require_cuda: bool = False) -> float:
    cfg = seed_config(seed, work)
    cmd = [sys.executable,
           str(REPO_ROOT / "task2_sentiment" / MEMBER / "src/train.py"),
           "--config", str(cfg), "--seed", str(seed),
           "--tag-suffix", f"_seed{seed}"]
    if require_cuda:
        # Forwarded to every child rather than checked once here. A sweep is hours of
        # subprocesses, and on the lab PC the silent fallback is CPU -- there is no MPS
        # to make a wrong-device run merely slow. Each arm fails on its own device.
        cmd.append("--require-cuda")
    if smoke:
        # A cheap end-to-end rehearsal of the harness: tiny data, one epoch. Proves the
        # plumbing and the aggregation before a real sweep is worth an hour of GPU.
        cmd.append("--smoke")
    print(f"\n{'=' * 74}\n>>> seed {seed}\n{'=' * 74}", flush=True)
    if dry:
        print("  [dry-run]", " ".join(cmd))
        return 0.0
    t0 = time.time()
    r = subprocess.run(cmd, cwd=REPO_ROOT)
    if r.returncode != 0:
        raise SystemExit(f"seed {seed} failed with exit code {r.returncode}")
    return time.time() - t0


def collect(seed: int, work: Path) -> dict[str, dict]:
    """Read one seed's per-model metrics from the report it wrote."""
    report = work / f"seed_{seed}" / "metrics_report.csv"
    if not report.exists():          # --smoke redirects the report under outputs/smoke
        report = work / f"seed_{seed}/outputs/smoke" / "metrics_report.csv"
    if not report.exists():
        raise SystemExit(
            f"seed {seed} produced no metrics_report.csv (looked in "
            f"{report.parent})")
    with report.open(encoding="utf-8") as f:
        return {r["model"]: r for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="9015,17,2718,31415,1234",
                    help="comma-separated; the first should be the reported seed (9015) "
                         "so the sweep contains the run already published")
    ap.add_argument("--work", default="task2_sentiment/shriram_dundigalla/outputs/seed_sweep",
                    help="where per-seed configs, checkpoints and outputs go")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="rehearse the whole harness on tiny data; numbers are "
                         "meaningless but the plumbing is proven")
    ap.add_argument("--require-cuda", action="store_true",
                    help="fail every arm that cannot actually use CUDA, instead of "
                         "letting the sweep quietly spend hours on the CPU")
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    work = REPO_ROOT / args.work
    if not work.resolve().is_relative_to(REPO_ROOT):
        raise SystemExit("--work must be inside the repository")
    work.mkdir(parents=True, exist_ok=True)

    elapsed = {s: run_seed(s, work, args.dry_run, args.smoke, args.require_cuda)
               for s in seeds}
    if args.dry_run:
        print(f"\n[dry-run] would train {len(MODELS)} models x {len(seeds)} seeds")
        return

    results = {s: collect(s, work) for s in seeds}

    rows = []
    for model in MODELS:
        row: dict[str, object] = {"model": model, "n_seeds": len(seeds),
                                  "seeds": " ".join(map(str, seeds))}
        for metric in METRICS:
            vals = [float(results[s][model][metric]) for s in seeds]
            row[f"{metric}_mean"] = statistics.fmean(vals)
            row[f"{metric}_std"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
            row[f"{metric}_min"] = min(vals)
            row[f"{metric}_max"] = max(vals)
        rows.append(row)

    # A smoke rehearsal must not leave behind a file that looks like a result.
    out_csv = (OUT_CSV.with_name("seed_sweep_smoke.csv") if args.smoke else OUT_CSV)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ------------------------------------------------------------------ the verdict
    print(f"\n{'=' * 74}\nSEED SWEEP — {len(seeds)} seeds"
          f"{'  (SMOKE: numbers are meaningless)' if args.smoke else ''}\n{'=' * 74}")
    print(f"{'model':16} {'macro-F1 mean':>14} {'std':>9} {'min':>9} {'max':>9}")
    for r in rows:
        print(f"{r['model']:16} {r['f1_macro_mean']:14.4f} {r['f1_macro_std']:9.4f} "
              f"{r['f1_macro_min']:9.4f} {r['f1_macro_max']:9.4f}")

    # The comparison the sweep exists to make: is the gap between the two experimental
    # architectures larger than the noise within either of them?
    by = {r["model"]: r for r in rows}
    gap = abs(by["exp1_bilstm"]["f1_macro_mean"] - by["exp2_textcnn"]["f1_macro_mean"])
    noise = max(by["exp1_bilstm"]["f1_macro_std"], by["exp2_textcnn"]["f1_macro_std"])
    print(f"\nbetween-architecture gap (BiLSTM vs TextCNN) : {gap:.5f} macro-F1")
    print(f"largest within-architecture seed std          : {noise:.5f} macro-F1")
    if noise == 0:
        verdict = "only one seed: no variance estimate, re-run with more seeds"
    elif gap < noise:
        verdict = ("the architectures differ by LESS than seed noise, so this experiment "
                   "cannot resolve a difference between them -- a stronger and more "
                   "honest statement than 'no evidence of a difference'")
    elif gap < 2 * noise:
        verdict = ("the gap is within 2 standard deviations of seed noise: still not "
                   "resolvable, and the single-seed ranking should not be reported as one")
    else:
        verdict = ("the gap exceeds seed noise, so the single-seed difference survives "
                   "re-initialisation and the ranking is real")
    print(f"\nVERDICT: {verdict}")
    print(f"\nwrote {out_csv.relative_to(REPO_ROOT)}")
    total = sum(elapsed.values())
    print(f"total {total / 60:.1f} min for {len(seeds)} seeds x {len(MODELS)} models")


if __name__ == "__main__":
    main()
