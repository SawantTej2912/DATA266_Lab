"""Figures for Task 2, generated from the saved predictions so they never disagree
with the numbers in metrics_report.csv.

    python task2_sentiment/shriram_dundigalla/src/plots.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, roc_curve  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS = ["negative", "positive"]


def main():
    cfg = yaml.safe_load(
        (REPO_ROOT / "task2_sentiment/shriram_dundigalla/src/config.yaml").read_text("utf-8")
    )
    member_dir = REPO_ROOT / "task2_sentiment" / "shriram_dundigalla"
    out_dir = REPO_ROOT / cfg["paths"]["outputs"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    preds = np.load(out_dir / "test_predictions.npz")
    y_true = preds["y_true"]
    report = pd.read_csv(member_dir / "metrics_report.csv")
    models = report["model"].tolist()
    histories = json.loads((out_dir / "histories.json").read_text("utf-8"))

    # ------------------------------------------------------ confusion matrices
    fig, axes = plt.subplots(1, len(models), figsize=(4.4 * len(models), 4.2))
    for ax, name in zip(np.atleast_1d(axes), models):
        ConfusionMatrixDisplay.from_predictions(
            y_true, preds[f"pred_{name}"], display_labels=LABELS,
            ax=ax, colorbar=False, cmap="Blues", values_format="d",
        )
        acc = report.loc[report["model"] == name, "accuracy"].iloc[0]
        ax.set_title(f"{name}\naccuracy {acc:.4f}")
    fig.suptitle("Task 2 — confusion matrices on the full 38K Yelp Polarity test set")
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in models:
        p = preds[f"probs_{name}"]
        fpr, tpr, _ = roc_curve(y_true, p)
        auc = report.loc[report["model"] == name, "roc_auc"].iloc[0]
        axes[0].plot(fpr, tpr, label=f"{name} (AUC {auc:.4f})")

        prec, rec, _ = precision_recall_curve(y_true, p)
        ap = report.loc[report["model"] == name, "pr_auc"].iloc[0]
        axes[1].plot(rec, prec, label=f"{name} (AP {ap:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="chance")
    axes[0].set_xlabel("false positive rate"); axes[0].set_ylabel("true positive rate")
    axes[0].set_title("ROC"); axes[0].legend(loc="lower right"); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("recall"); axes[1].set_ylabel("precision")
    axes[1].set_title("Precision-Recall"); axes[1].legend(loc="lower left"); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "roc_pr_curves.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------ reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name in models:
        centres, accs, counts = M.reliability_bins(
            y_true, preds[f"probs_{name}"], cfg["eval"]["calibration_bins"]
        )
        # The positive-class ECE, not the confidence ECE. This axis bins by
        # p(positive) over [0, 1], which is exactly what positive_class_ece measures;
        # labelling it with the confidence ECE put a number from a different
        # quantity next to the curve.
        ece = report.loc[report["model"] == name, "ece_positive_class"].iloc[0]
        ok = ~np.isnan(accs)
        axes[0].plot(centres[ok], accs[ok], marker="o",
                     label=f"{name} (positive-class ECE {ece:.4f})")
        axes[1].plot(centres, counts, marker=".", label=name)

    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    axes[0].set_xlabel("predicted p(positive)"); axes[0].set_ylabel("observed positive rate")
    axes[0].set_title("Reliability (positive-class probability)")
    axes[0].legend(loc="upper left", fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("predicted p(positive)"); axes[1].set_ylabel("reviews in bin")
    axes[1].set_yscale("log"); axes[1].set_title("Confidence histogram")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "calibration.png", dpi=150)
    plt.close(fig)

    # ---------------------------------------------------------- training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for name, h in histories.items():
        axes[0].plot(h["epoch"], h["train_loss"], marker="o", label=f"{name} train")
        axes[0].plot(h["epoch"], h["val_loss"], marker="s", ls="--", label=f"{name} val")
        axes[1].plot(h["epoch"], h["val_macro_f1"], marker="^", label=name)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Training and validation loss"); axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("validation macro-F1")
    axes[1].set_title("Validation macro-F1 (early-stopping criterion)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "training_curves.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------ slice metrics
    sl = pd.read_csv(out_dir / "slice_metrics.csv")
    sl["label"] = sl["slice_family"] + " = " + sl["slice"]
    order = sl[sl["model"] == models[0]].sort_values("label")["label"].tolist()

    fig, ax = plt.subplots(figsize=(max(9, 0.8 * len(order)), 5))
    width = 0.8 / len(models)
    x = np.arange(len(order))
    for i, name in enumerate(models):
        sub = sl[sl["model"] == name].set_index("label").reindex(order)
        ax.bar(x + i * width, sub["macro_f1"], width, label=name)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_ylabel("macro-F1")
    # Zoom to the data range: all values sit in a ~0.04 band, so a 0-1 axis would
    # render every bar the same height and hide the effect being reported.
    lo, hi = sl["macro_f1"].min(), sl["macro_f1"].max()
    pad = (hi - lo) * 0.25
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_title("Task 2 — macro-F1 per data slice (robustness)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "slice_macro_f1.png", dpi=150)
    plt.close(fig)

    # -------------------------------------------- bootstrap CI comparison plot
    fig, ax = plt.subplots(figsize=(8, 4.6))
    metric_names = ["accuracy", "macro_f1", "mcc"]
    point_cols = {"accuracy": "accuracy", "macro_f1": "f1_macro", "mcc": "mcc"}
    x = np.arange(len(metric_names))
    for i, name in enumerate(models):
        row = report[report["model"] == name].iloc[0]
        vals = [row[point_cols[m]] for m in metric_names]
        lo = [row[f"{m}_ci95_low"] for m in metric_names]
        hi = [row[f"{m}_ci95_high"] for m in metric_names]
        err = np.vstack([np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)])
        ax.errorbar(x + i * 0.12 - 0.12, vals, yerr=err, fmt="o", capsize=5, label=name)
    ax.set_xticks(x); ax.set_xticklabels(["accuracy", "macro-F1", "MCC"])
    ax.set_title("Point estimate with 95% bootstrap confidence interval")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(fig_dir / "bootstrap_cis.png", dpi=150)
    plt.close(fig)

    print(f"wrote 6 figures to {fig_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
