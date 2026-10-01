"""
evaluate_local.py — Task 3 (CycleGAN Monet <-> Photo) image-folder metrics.

Domains (same convention as the TA script):  A = Monet, B = Photo
  pred_A2B = generated photos from Monet inputs   (compare to real photo_jpg)
  pred_B2A = generated Monets from photo inputs   (compare to real monet_jpg)

Computes, both directions:
  - FID and "MiFID" exactly as the TA's Part3_Evaluation_Script does  -> submission.csv (the Kaggle file)
  - KID (unbiased MMD^2, cubic polynomial kernel, subset estimate, mean ± std)
  - Generative precision / recall / density / coverage (Naeem et al. 2020, k-NN on Inception features)
  - LPIPS (input vs. its translation; AlexNet) — perceptual change
  - Content-preservation cosine similarity (Inception features of input vs. its translation, paired by filename)
Pretrained networks here are used ONLY for evaluation, never to create or modify submitted images.

Usage:
  python evaluate_local.py --data ../data --pred ./outputs --out . [--n-eval 300]
"""
import argparse, glob, json, os
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine

NO_PRETRAINED = os.environ.get("NO_PRETRAINED", "0") == "1"   # offline CI only: random weights, numbers meaningless


def list_images(folder):
    exts = (".jpg", ".jpeg", ".png")
    paths = []
    for ext in exts:
        paths += glob.glob(os.path.join(folder, f"*{ext}")) + glob.glob(os.path.join(folder, f"*{ext.upper()}"))
    return sorted(set(paths))


def find_domain_dir(data_dir, name):
    """Accept monet_jpg / monet.jpg / monet (same for photo)."""
    for cand in (f"{name}_jpg", f"{name}.jpg", name):
        p = Path(data_dir) / cand
        if p.is_dir():
            return p
    raise FileNotFoundError(f"No {name}_jpg folder under {data_dir}")


# ── Inception features (identical model and preprocessing to the TA script) ──
_INC_TF = T.Compose([T.Resize(299), T.CenterCrop(299), T.ToTensor(),
                     T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
_inception = None


def get_inception(device):
    global _inception
    if _inception is None:
        w = None if NO_PRETRAINED else models.Inception_V3_Weights.IMAGENET1K_V1
        m = models.inception_v3(weights=w, transform_input=False, aux_logits=True, init_weights=NO_PRETRAINED)
        m.fc = nn.Identity()
        _inception = m.to(device).eval()
    return _inception


@torch.no_grad()
def inception_features(paths, device, batch_size=32):
    model, feats = get_inception(device), []
    for i in range(0, len(paths), batch_size):
        x = torch.stack([_INC_TF(Image.open(p).convert("RGB")) for p in paths[i:i + batch_size]]).to(device)
        feats.append(model(x).float().cpu().numpy())
    return np.concatenate(feats).astype(np.float64)


# ── TA-identical FID / MiFID ──
def frechet_distance(mu1, s1, mu2, s2, eps=1e-6):
    try:
        covmean, _ = scipy.linalg.sqrtm(s1.dot(s2), disp=False)       # SciPy < 1.18 (what the TA script uses)
    except TypeError:
        covmean = scipy.linalg.sqrtm(s1.dot(s2))                       # SciPy >= 1.18 removed `disp`
    if not np.isfinite(covmean).all():
        off = np.eye(s1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((s1 + off).dot(s2 + off))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    d = mu1 - mu2
    return float(d.dot(d) + np.trace(s1 + s2 - 2 * covmean))


def ta_fid_mifid(real_feat, gen_feat):
    n = min(len(real_feat), len(gen_feat)); r, g = real_feat[:n], gen_feat[:n]
    fid = frechet_distance(r.mean(0), np.cov(r, rowvar=False), g.mean(0), np.cov(g, rowvar=False))
    mifid = float(np.mean([cosine(r[i], g[i]) for i in range(n)]))
    return fid, mifid


# ── KID ──
def kid(real_feat, gen_feat, n_subsets=50, subset_size=100, seed=0):
    rng = np.random.default_rng(seed); d = real_feat.shape[1]
    m = min(subset_size, len(real_feat), len(gen_feat)); vals = []
    for _ in range(n_subsets):
        x = real_feat[rng.choice(len(real_feat), m, replace=False)]
        y = gen_feat[rng.choice(len(gen_feat), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        mmd = (kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean()
        vals.append(mmd)
    return float(np.mean(vals)), float(np.std(vals))


# ── Precision / recall / density / coverage ──
def _pdist(a, b):
    aa, bb = (a * a).sum(1)[:, None], (b * b).sum(1)[None, :]
    return np.sqrt(np.maximum(aa + bb - 2 * a @ b.T, 0))


def prdc(real_feat, gen_feat, k=5):
    rr = _pdist(real_feat, real_feat); gg = _pdist(gen_feat, gen_feat); rg = _pdist(real_feat, gen_feat)
    r_rad = np.sort(rr, 1)[:, k]; g_rad = np.sort(gg, 1)[:, k]          # index 0 is self-distance
    precision = float((rg < r_rad[:, None]).any(0).mean())
    recall = float((rg < g_rad[None, :]).any(1).mean())
    density = float((rg < r_rad[:, None]).sum(0).mean() / k)
    coverage = float((rg.min(1) < r_rad).mean())
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


# ── LPIPS and content cosine (input vs. its translation, paired by filename) ──
def paired(inputs, outputs):
    """Match each input to its translation by file stem (outputs are saved as <input stem>.jpg)."""
    out_by_stem = {Path(p).stem: p for p in outputs}
    pairs = [(i, out_by_stem[Path(i).stem]) for i in inputs if Path(i).stem in out_by_stem]
    return [a for a, _ in pairs], [b for _, b in pairs]


@torch.no_grad()
def lpips_mean(inputs, outputs, device, size=256):
    if NO_PRETRAINED:
        return float("nan")
    import lpips
    net = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    tf = T.Compose([T.Resize(size), T.CenterCrop(size), T.ToTensor(), T.Normalize((0.5,) * 3, (0.5,) * 3)])
    vals = []
    for a, b in zip(inputs, outputs):
        x = tf(Image.open(a).convert("RGB")).unsqueeze(0).to(device)
        y = tf(Image.open(b).convert("RGB")).unsqueeze(0).to(device)
        vals.append(net(x, y).item())
    return float(np.mean(vals))


def evaluate(data_dir, pred_dir, out_dir, n_eval=300, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    monet_dir, photo_dir = find_domain_dir(data_dir, "monet"), find_domain_dir(data_dir, "photo")
    real_monet = list_images(monet_dir)[:n_eval]; real_photo = list_images(photo_dir)[:n_eval]
    gen_a2b = list_images(Path(pred_dir) / "pred_A2B")[:n_eval]; gen_b2a = list_images(Path(pred_dir) / "pred_B2A")[:n_eval]
    assert gen_a2b and gen_b2a, "pred_A2B / pred_B2A are empty — run inference first"
    F = {k: inception_features(v, device) for k, v in
         dict(real_monet=real_monet, real_photo=real_photo, gen_a2b=gen_a2b, gen_b2a=gen_b2a).items()}

    fid_b2a, mifid_b2a = ta_fid_mifid(F["real_monet"], F["gen_b2a"])
    fid_a2b, mifid_a2b = ta_fid_mifid(F["real_photo"], F["gen_a2b"])
    sub = pd.DataFrame([{"ID": 1, "FID": (fid_a2b + fid_b2a) / 2, "MiFID": (mifid_a2b + mifid_b2a) / 2}])
    sub.to_csv(Path(out_dir) / "submission.csv", index=False)

    res = {"n_eval": n_eval, "FID_A2B": fid_a2b, "FID_B2A": fid_b2a, "MiFID_A2B": mifid_a2b, "MiFID_B2A": mifid_b2a,
           "submission_FID": float(sub.FID[0]), "submission_MiFID": float(sub.MiFID[0])}
    for tag, real, gen in (("A2B", "real_photo", "gen_a2b"), ("B2A", "real_monet", "gen_b2a")):
        m, s = kid(F[real], F[gen]); res[f"KID_{tag}_mean"], res[f"KID_{tag}_std"] = m, s
        res.update({f"{k}_{tag}": v for k, v in prdc(F[real], F[gen]).items()})
    # content preservation: input vs. its own translation (paired by filename)
    for tag, inputs, outs in (("A2B", list_images(monet_dir), gen_a2b), ("B2A", list_images(photo_dir), gen_b2a)):
        ins, outs_p = paired(inputs, outs)
        fi, fo = inception_features(ins, device), inception_features(outs_p, device)
        cs = (fi * fo).sum(1) / (np.linalg.norm(fi, axis=1) * np.linalg.norm(fo, axis=1) + 1e-12)
        res[f"content_cosine_{tag}"] = float(cs.mean())
        res[f"LPIPS_{tag}"] = lpips_mean(ins, outs_p, device)
        res[f"n_pairs_{tag}"] = len(ins)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../data"); ap.add_argument("--pred", default="./outputs")
    ap.add_argument("--out", default="."); ap.add_argument("--n-eval", type=int, default=300)
    a = ap.parse_args()
    r = evaluate(a.data, a.pred, a.out, a.n_eval)
    pd.DataFrame([r]).to_csv(Path(a.out) / "full_metrics_eval_only.csv", index=False)
    print(json.dumps(r, indent=2))
