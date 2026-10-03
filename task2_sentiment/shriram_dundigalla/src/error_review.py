"""Assemble the 20-error review that section 2.2.4 requires.

The four required buckets are 5 confident false positives, 5 confident false negatives,
5 near-threshold errors and 5 slice-specific failures. This script selects them and
attaches the diagnostics that make the cause checkable — negation presence, truncation,
out-of-vocabulary rate, review length — so the read-through is about judging the cause
rather than hunting for it.

The `suggested_error_type` column is a heuristic first pass from those diagnostics. It
is a starting point for the read-through, not the finding; confirm or overwrite it
against the review text before citing it in the report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

import metrics as M     # noqa: E402
import preprocess as P  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]


def diagnose(row) -> tuple[str, str]:
    """Heuristic error type and a matching testable fix, from the diagnostics."""
    if row["truncated"]:
        return (
            "truncation — verdict may fall outside the first max_len tokens",
            "Re-score this review with max_len raised to cover its full length; if the "
            "prediction flips, raise max_len or switch to last-n truncation.",
        )
    if row["oov_rate"] > 0.15:
        return (
            "out-of-vocabulary — a large share of tokens map to <unk>",
            "Lower min_freq / raise vocab_size so these tokens get their own embedding, "
            "and re-score; if it flips, the vocabulary cap is the binding constraint.",
        )
    if row["has_negation"] and row["n_tokens"] <= 60:
        return (
            "negation scope — polarity word present but negated nearby",
            "Add an explicit negation-marking pass (prefix the next 3 tokens after a "
            "negation with NOT_) and re-train; a bag-of-words model cannot see scope.",
        )
    if row["n_tokens"] <= 20:
        return (
            "insufficient evidence — very short review, little lexical signal",
            "Check whether accuracy on the shortest length decile improves when the "
            "classifier abstains below a confidence floor.",
        )
    if row["shouty"]:
        return (
            "register / emphasis — heavy capitalisation lost in lowercasing",
            "Add a binary SHOUTY feature (or keep a caps-ratio input) and re-train to "
            "test whether the signal was being discarded by the lowercasing step.",
        )
    if row["exclamations"] >= 3:
        return (
            "punctuation signal stripped — emphatic punctuation removed in cleaning",
            "Retain exclamation count as a numeric feature and re-train; if the slice "
            "error rate drops, punctuation removal was costing information.",
        )
    return (
        "mixed sentiment — review contains both praise and complaint",
        "Label a 50-review mixed-sentiment probe set by hand and measure accuracy on it "
        "separately; if it is far below overall accuracy, the single-logit framing is "
        "the limit, not the architecture.",
    )


def main(model: str | None = None, out_name: str = "failure_analysis.md"):
    cfg = yaml.safe_load(
        (REPO_ROOT / "task2_sentiment/shriram_dundigalla/src/config.yaml").read_text("utf-8")
    )
    member_dir = REPO_ROOT / "task2_sentiment" / "shriram_dundigalla"
    out_dir = REPO_ROOT / cfg["paths"]["outputs"]

    preds = np.load(out_dir / "test_predictions.npz")
    y_true = preds["y_true"]
    report = pd.read_csv(member_dir / "metrics_report.csv")

    if model is None:
        model = report.loc[report["f1_macro"].idxmax(), "model"]
    print(f"reviewing errors from the best model: {model}")

    probs = preds[f"probs_{model}"]
    y_pred = preds[f"pred_{model}"]

    # Rebuild the test split to recover raw text and the slice attributes.
    import pickle
    with (REPO_ROOT / cfg["data"]["cache_dir"] / "task2_data.pkl").open("rb") as fh:
        data = pickle.load(fh)["data"]
    test, vocab = data.test, data.vocab
    sw = P.build_stopwords(cfg["preprocess"]["keep_negations"])
    stemmer = cfg["preprocess"]["stemmer"]
    max_len = cfg["data"]["max_len"]

    err = np.flatnonzero(y_pred != y_true)
    print(f"{len(err)} errors out of {len(y_true)} test reviews ({len(err)/len(y_true):.2%})")

    slices = pd.read_csv(out_dir / "slice_metrics.csv")
    worst = slices[slices["model"] == model].sort_values("error_rate", ascending=False).iloc[0]
    print(f"worst slice: {worst['slice_family']}={worst['slice']} "
          f"(error rate {worst['error_rate']:.3f}, n={int(worst['n'])})")

    # ------------------------------------------------------------- bucket picks
    fp = err[(y_true[err] == 0)]
    fn = err[(y_true[err] == 1)]
    confident_fp = fp[np.argsort(-probs[fp])][:5]
    confident_fn = fn[np.argsort(probs[fn])][:5]
    near = err[np.argsort(np.abs(probs[err] - 0.5))][:5]

    sl_values = M.slice_values(test.slices, worst["slice_family"])
    in_slice = err[sl_values[err] == worst["slice"]]
    chosen = set(confident_fp) | set(confident_fn) | set(near)
    slice_specific = [i for i in in_slice if i not in chosen][:5]

    buckets = [
        ("confident false positive", confident_fp),
        ("confident false negative", confident_fn),
        ("near-threshold error", near),
        (f"slice-specific failure ({worst['slice_family']}={worst['slice']})", slice_specific),
    ]

    rows = []
    for bucket, idxs in buckets:
        for i in idxs:
            i = int(i)
            raw = test.raw[i]
            toks = P.clean(raw, sw, stemmer)
            ids = test.ids[i]
            real = ids[ids != P.PAD_ID]
            rec = {
                "bucket": bucket,
                "test_index": i,
                "true_label": int(y_true[i]),
                "predicted_label": int(y_pred[i]),
                "predicted_prob_positive": float(probs[i]),
                "confidence": float(max(probs[i], 1 - probs[i])),
                "n_tokens": int(len(toks)),
                "truncated": bool(len(toks) > max_len),
                "oov_rate": float((real == P.UNK_ID).mean()) if len(real) else 0.0,
                "has_negation": bool(test.slices["has_negation"].iloc[i]),
                "shouty": bool(test.slices["shouty"].iloc[i]),
                "exclamations": int(test.slices["exclamations"].iloc[i]),
                "length_bucket": test.slices["length_bucket"].iloc[i],
                "review_text": raw,
            }
            etype, fix = diagnose(rec)
            rec["suggested_error_type"] = etype
            rec["proposed_testable_fix"] = fix
            rows.append(rec)

    df = pd.DataFrame(rows)
    csv_path = out_dir / f"error_review_{model}.csv"
    df.to_csv(csv_path, index=False)
    print(f"wrote {csv_path.relative_to(REPO_ROOT)} ({len(df)} errors)")

    # ------------------------------------------------------------- markdown copy
    lines = [
        f"# Task 2 — error review ({model})",
        "",
        f"Model reviewed: **{model}** (highest macro-F1 of my three).",
        f"Errors: **{len(err)} of {len(y_true)}** test reviews "
        f"(**{len(err)/len(y_true):.2%}** error rate).",
        f"Worst slice: **{worst['slice_family']} = {worst['slice']}** "
        f"(error rate {worst['error_rate']:.3f} over {int(worst['n'])} reviews).",
        "",
        "`suggested_error_type` below is a heuristic first pass from the diagnostics; "
        "each one is confirmed or overwritten against the review text during the read-through.",
        "",
    ]
    for bucket, _ in buckets:
        sub = df[df["bucket"] == bucket]
        lines += [f"## {bucket.title()} ({len(sub)})", ""]
        for _, r in sub.iterrows():
            label = "positive" if r["true_label"] == 1 else "negative"
            pred = "positive" if r["predicted_label"] == 1 else "negative"
            lines += [
                f"### Test index {r['test_index']}",
                "",
                f"- True: **{label}** · Predicted: **{pred}** "
                f"(p(positive) = {r['predicted_prob_positive']:.4f})",
                f"- Tokens after cleaning: {r['n_tokens']} ({r['length_bucket']}) · "
                f"OOV rate {r['oov_rate']:.3f} · truncated: {r['truncated']}",
                f"- Negation present: {r['has_negation']} · shouty caps: {r['shouty']} · "
                f"exclamations: {r['exclamations']}",
                f"- Suggested error type: *{r['suggested_error_type']}*",
                f"- Proposed testable fix: {r['proposed_testable_fix']}",
                "",
                "> " + r["review_text"].replace("\\n", " ").replace("\n", " ")[:900],
                "",
            ]

    md_path = member_dir / out_name
    md_path.write_text("\n".join(lines), "utf-8")
    print(f"wrote {md_path.relative_to(REPO_ROOT)}")

    summary = {
        "model_reviewed": model,
        "n_errors": int(len(err)),
        "error_rate": float(len(err) / len(y_true)),
        "worst_slice": {k: (v.item() if hasattr(v, "item") else v)
                        for k, v in worst.to_dict().items()},
        "bucket_counts": {b: int((df["bucket"] == b).sum()) for b, _ in buckets},
    }
    (out_dir / "error_review_summary.json").write_text(json.dumps(summary, indent=2), "utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, help="defaults to the highest macro-F1 model")
    args = ap.parse_args()
    main(model=args.model)
