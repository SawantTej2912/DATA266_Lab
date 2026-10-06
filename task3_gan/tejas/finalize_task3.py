"""Finalize Task 3 without re-running training: human-audit agreement + Kaggle values -> metrics files.
Run from the repo root:  python task3_gan/tejas/finalize_task3.py
"""
import sys
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

KAGGLE_PUBLIC, KAGGLE_PRIVATE, KAGGLE_RANK = -49.55, None, 20      # update if the leaderboard changes before the deadline

M = Path("task3_gan/tejas"); O = M / "outputs"
r1, r2 = pd.read_csv(O / "audit_rater1.csv"), pd.read_csv(O / "audit_rater2.csv")
key = pd.read_csv(O / "audit_key.csv")
cols = ["style_1to5", "content_1to5", "artifacts_0or1"]
m = r1.merge(r2, on="sample_id", suffixes=("_r1", "_r2")).merge(key, on="sample_id")
missing = m[[f"{c}_{r}" for c in cols for r in ("r1", "r2")]].isna().any(axis=1)
if missing.any():
    sys.exit(f"Unrated samples: {', '.join(m.loc[missing, 'sample_id'])} - fill both sheets completely first.")
for c in cols:
    for r in ("r1", "r2"):
        m[f"{c}_{r}"] = m[f"{c}_{r}"].astype(int)
bad = [(c, r) for c in cols[:2] for r in ("r1", "r2") if not m[f"{c}_{r}"].between(1, 5).all()] + \
      [("artifacts", r) for r in ("r1", "r2") if not m[f"artifacts_0or1_{r}"].isin([0, 1]).all()]
if bad:
    sys.exit(f"Out-of-range ratings in {bad}: style/content must be 1-5, artifacts 0/1.")

res = {"audit_n": len(m)}
for c, w in (("style_1to5", "quadratic"), ("content_1to5", "quadratic"), ("artifacts_0or1", None)):
    a, b = m[f"{c}_r1"], m[f"{c}_r2"]
    res[f"audit_kappa_{c}"] = round(float(cohen_kappa_score(a, b, weights=w)), 4)
    res[f"audit_pct_agree_{c}"] = round(float((a == b).mean()), 4)
    res[f"audit_mean_{c}"] = round(float((a + b).mean() / 2), 4)
    for d, g in m.groupby("direction"):
        tag = "A2B" if d.split("→")[-1].strip() == "Photo" else "B2A"
        res[f"audit_mean_{c}_{tag}"] = round(float((g[f"{c}_r1"] + g[f"{c}_r2"]).mean() / 2), 4)
res.update(kaggle_public=KAGGLE_PUBLIC, kaggle_private=KAGGLE_PRIVATE, kaggle_rank=KAGGLE_RANK)
pd.DataFrame([res]).to_csv(O / "audit_results.csv", index=False)
for f in (M / "full_metrics_report.csv", M / "metrics_report.csv", M / "src" / "full_metrics_report.csv"):
    if f.exists():
        df = pd.read_csv(f)
        for k, v in res.items():
            df[k] = v
        df.to_csv(f, index=False)
        print("updated", f)
print("\n".join(f"  {k}: {v}" for k, v in res.items()))
