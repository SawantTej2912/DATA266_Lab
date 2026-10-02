"""The full Task 2 metric battery, in one place so every team member scores identically.

The bootstrap works from 2x2 contingency counts rather than calling scikit-learn per
replicate: accuracy, macro-F1 and MCC all have closed forms in TP/FP/FN/TN, which makes
2000 replicates over 38K test rows a second of work instead of several minutes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from statsmodels.stats.contingency_tables import mcnemar


# ------------------------------------------------------- closed forms from counts

def _counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return (tn, fp, fn, tp) for binary labels in {0, 1}."""
    k = np.bincount(2 * y_true + y_pred, minlength=4)
    return k[0], k[1], k[2], k[3]


def _acc_f1_mcc(tn, fp, fn, tp):
    total = tn + fp + fn + tp
    acc = (tp + tn) / total

    f1_pos = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), 0.0)
    f1_neg = np.where(2 * tn + fn + fp > 0, 2 * tn / np.maximum(2 * tn + fn + fp, 1), 0.0)
    macro_f1 = (f1_pos + f1_neg) / 2

    denom = np.sqrt(
        (tp + fp).astype(float) * (tp + fn) * (tn + fp) * (tn + fn)
    )
    mcc = np.where(denom > 0, (tp * tn - fp * fn) / np.maximum(denom, 1e-12), 0.0)
    return acc, macro_f1, mcc


def bootstrap_cis(y_true, y_pred, n_boot: int = 2000, seed: int = 9015, chunk: int = 200) -> dict:
    """Percentile 95% CIs for accuracy, macro-F1 and MCC by case resampling."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    accs, f1s, mccs = [], [], []

    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        yt, yp = y_true[idx], y_pred[idx]
        k = 2 * yt + yp
        # One bincount per replicate, vectorised via offsetting each row into its own bin block.
        offs = (np.arange(b)[:, None] * 4 + k).ravel()
        counts = np.bincount(offs, minlength=4 * b).reshape(b, 4)
        a, f, m = _acc_f1_mcc(counts[:, 0], counts[:, 1], counts[:, 2], counts[:, 3])
        accs.append(a); f1s.append(f); mccs.append(m)

    out = {}
    for name, vals in (("accuracy", accs), ("macro_f1", f1s), ("mcc", mccs)):
        v = np.concatenate(vals)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[f"{name}_ci95_low"] = float(lo)
        out[f"{name}_ci95_high"] = float(hi)
    return out


# ------------------------------------------------------------------- calibration

def confidence_ece(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE on the confidence of the *predicted* class, binned over [0.5, 1].

    The standard multiclass formulation (Guo et al., 2017): each example contributes its
    predicted class's probability, and a bin's gap is |accuracy - mean confidence|. For
    binary problems the confidence is `max(p, 1-p)`, so it lives in [0.5, 1] and the bin
    edges have to span that range rather than [0, 1] -- using [0, 1] edges here would
    leave the lower half of the bins permanently empty and understate the error.

    This answers "when the model says it is 90% sure, is it right 90% of the time?" It
    is not the same question as `positive_class_ece` below, and reporting either one
    under the bare name "ECE" is ambiguous, so both are reported by name.
    """
    pred = (y_prob >= 0.5).astype(int)
    conf = np.where(pred == 1, y_prob, 1.0 - y_prob)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0.5 else (conf >= lo) & (conf <= hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def positive_class_ece(y_true, y_prob, n_bins: int = 15) -> float:
    """ECE on the positive-class probability, binned over [0, 1].

    The binary-specific formulation: bin by P(positive) and compare each bin's mean
    predicted probability against the observed positive rate. This answers "when the
    model says 30% positive, are 30% of those reviews actually positive?", which the
    confidence version cannot -- it folds p=0.05 and p=0.95 into the same bin.

    It is also the quantity the reliability diagram in `plots.py` draws, via
    `reliability_bins`, so this is the number that belongs beside that figure.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(y_true)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        if m.sum() == 0:
            continue
        ece += m.sum() / n * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins: int = 15):
    """(bin_centre, empirical_accuracy, count) for a reliability diagram."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    centres, accs, counts = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (y_prob > lo) & (y_prob <= hi) if lo > 0 else (y_prob >= lo) & (y_prob <= hi)
        centres.append((lo + hi) / 2)
        counts.append(int(m.sum()))
        accs.append(float(y_true[m].mean()) if m.sum() else np.nan)
    return np.array(centres), np.array(accs), np.array(counts)


# ------------------------------------------------------------------------ slices

SLICE_FAMILIES = {
    "length_bucket": lambda d: d["length_bucket"],
    "has_negation": lambda d: np.where(d["has_negation"], "negation", "no_negation"),
    "shouty": lambda d: np.where(d["shouty"], "shouty_caps", "normal_caps"),
    "exclamations": lambda d: np.where(d["exclamations"] > 0, "has_exclaim", "no_exclaim"),
}


def slice_values(slices_df, family: str) -> np.ndarray:
    """Per-row slice label for one family, as used by both scoring and error review."""
    return np.asarray(SLICE_FAMILIES[family](slices_df))


def slice_report(y_true, y_pred, slices_df, min_n: int = 200) -> list[dict]:
    """Macro-F1 and error rate per data slice.

    Slices with fewer than `min_n` rows are dropped: a macro-F1 computed on a handful
    of reviews swings wildly and would invite conclusions the data cannot support.
    """
    rows = []
    for name in SLICE_FAMILIES:
        values = slice_values(slices_df, name)
        for level in sorted(set(values.tolist())):
            m = values == level
            if m.sum() < min_n:
                continue
            yt, yp = y_true[m], y_pred[m]
            rows.append({
                "slice_family": name,
                "slice": str(level),
                "n": int(m.sum()),
                "positive_rate": float(yt.mean()),
                "macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
                "error_rate": float((yt != yp).mean()),
            })
    return rows


# --------------------------------------------------------------- paired McNemar

def mcnemar_test(y_true, y_pred_a, y_pred_b) -> dict:
    """Paired McNemar between two models on identical test rows.

    Uses the exact binomial when the discordant cells are small, the chi-square with
    continuity correction otherwise. b and c are the cells that carry the evidence:
    cases where exactly one of the two models is right.
    """
    a_correct = y_pred_a == y_true
    b_correct = y_pred_b == y_true
    n00 = int((~a_correct & ~b_correct).sum())
    n01 = int((~a_correct & b_correct).sum())   # only B right
    n10 = int((a_correct & ~b_correct).sum())   # only A right
    n11 = int((a_correct & b_correct).sum())

    table = [[n11, n10], [n01, n00]]
    exact = (n10 + n01) < 25
    res = mcnemar(table, exact=exact, correction=not exact)
    return {
        "only_a_correct": n10,
        "only_b_correct": n01,
        "both_correct": n11,
        "both_wrong": n00,
        "test": "exact binomial" if exact else "chi-square (continuity corrected)",
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


# -------------------------------------------------------------------- full report

def full_report(y_true, y_prob, threshold: float = 0.5, n_boot: int = 2000,
                n_bins: int = 15, seed: int = 9015, slices_df=None) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    p_ma, r_ma, f_ma, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    p_mi, r_mi, f_mi, _ = precision_recall_fscore_support(y_true, y_pred, average="micro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    report = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_ma), "recall_macro": float(r_ma), "f1_macro": float(f_ma),
        "precision_micro": float(p_mi), "recall_micro": float(r_mi), "f1_micro": float(f_mi),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        "confusion_tn": int(tn), "confusion_fp": int(fp),
        "confusion_fn": int(fn), "confusion_tp": int(tp),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        # Both, by name. "ECE" alone is ambiguous: these measure different things and
        # differ by more than 3x on these models. See the two functions above.
        "ece_predicted_class_confidence": confidence_ece(y_true, y_prob, n_bins),
        "ece_positive_class": positive_class_ece(y_true, y_prob, n_bins),
        **bootstrap_cis(y_true, y_pred, n_boot=n_boot, seed=seed),
    }
    if slices_df is not None:
        report["_slices"] = slice_report(y_true, y_pred, slices_df)
    report["_y_pred"] = y_pred
    return report
