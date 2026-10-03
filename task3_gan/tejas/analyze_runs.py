import pandas as pd, numpy as np, matplotlib.pyplot as plt
from pathlib import Path
M = Path(".")
runs = {  # name: (folder, iters_per_epoch, total_epochs, decay_start, notes)
    "e200":  (M/"exploratory/e200", 300, 200, 100, "resize-conv, no DiffAug, D lr 2e-4"),
    "e300":  (M/"exploratory/e300", 300, 300, 150, "+DiffAug, D lr 1e-4, EMA"),
    "i180k": (M,                    900, 200, 100, "same as e300, 900 it/epoch"),
}
sel, hist = [], []
for name, (d, ipe, ep, dec, note) in runs.items():
    s = pd.read_csv(d/"outputs/checkpoint_selection.csv"); s["run"] = name; s["iters"] = s.epoch * ipe
    s["phase"] = np.where(s.epoch <= dec, "constant LR", "LR decay"); sel.append(s)
    h = pd.read_csv(d/"outputs/epoch_history.csv"); h["run"] = name; h["iters"] = h.epoch * ipe; hist.append(h)
    b = s.loc[s.kaggle_score.idxmax()]
    print(f"{name:6s} | {note:38s} | total {ep*ipe:>7,} it | best @ {int(b.iters):>7,} it (epoch {int(b.epoch)}): FID {b.FID:6.2f} (A2B {b.FID_A2B:6.1f}, B2A {b.FID_B2A:6.1f}) kaggle {b.kaggle_score:.3f}")
S, H = pd.concat(sel), pd.concat(hist)
print("\nquick30 (9,000 it, no snapshots): FID 146.97, kaggle -73.70")

print("\n=== FID by iterations, all snapshots ===")
print(S.pivot_table(index="iters", columns="run", values="FID").round(2).to_string())

print("\n=== Gain during constant-LR vs. decay phase ===")
for name, g in S.groupby("run"):
    g = g.sort_values("iters"); dec = runs[name][3] * runs[name][1]
    c = g[g.iters <= dec]; d_ = g[g.iters > dec]
    print(f"{name:6s}: end of constant phase FID {c.FID.iloc[-1]:6.2f} @ {int(c.iters.iloc[-1]):,} it -> best in decay {d_.FID.min():6.2f}  (decay gain {c.FID.iloc[-1]-d_.FID.min():5.2f})")

# trend: best FID vs log2(total iterations) for the DiffAug runs
x = np.log2([90_000, 180_000]); y = [S[S.run=="e300"].FID.min(), S[S.run=="i180k"].FID.min()]
slope = (y[1]-y[0])/(x[1]-x[0])
print(f"\n=== Rough extrapolation (2 points only — treat as optimistic) ===\nFID change per doubling of total iterations: {slope:.2f}")
for it in (270_000, 360_000, 540_000):
    print(f"  {it:>7,} it -> FID ~{y[1] + slope*np.log2(it/180_000):6.1f}   (~{it/180_000*329/60:4.1f} h at current speed)")

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
for name, g in S.groupby("run"):
    g = g.sort_values("iters"); ax[0].plot(g.iters/1000, g.FID, "o-", label=f"{name} ({runs[name][4]})")
ax[0].scatter([9], [146.97], c="k", label="quick30"); ax[0].set_xlabel("iterations (K)"); ax[0].set_ylabel("FID (mean, TA formula)")
ax[0].set_title("Submitted FID vs. training iterations"); ax[0].legend(fontsize=8)
for name, h in H.groupby("run"):
    ax[1].plot(h.iters/1000, h.loss_cyc, label=name); ax[2].plot(h.iters/1000, h.D_A_real - h.D_A_fake, label=f"{name} D_A"); 
ax[1].set_title("Cycle loss"); ax[1].set_xlabel("iterations (K)"); ax[1].legend()
ax[2].set_title("D_A real-fake gap (higher = D dominating)"); ax[2].set_xlabel("iterations (K)"); ax[2].legend()
plt.tight_layout(); plt.savefig("outputs/experiments_comparison.png", dpi=150)
S.to_csv("outputs/experiments_snapshot_table.csv", index=False)
print("\nSaved outputs/experiments_comparison.png and outputs/experiments_snapshot_table.csv")
