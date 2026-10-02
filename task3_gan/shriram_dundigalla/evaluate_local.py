"""Score the generated images with the course evaluation metric.

The scoring logic below is the official evaluation script. It is reproduced
unchanged: the same Inception weights, the same ImageNet normalization, the same
Resize(299)+CenterCrop(299), reference statistics recomputed from the real image
folders rather than read from a cached .npz, both sets truncated to the first 300
sorted filenames, and the reported value averaged over the two directions. Only
the paths are different, and they are arguments instead of constants.

None of that is interchangeable with the convention used by a general-purpose FID
implementation. The reference .npz files shipped with the dataset were built by
scaling to [-1,1] ("tf" preprocessing); a self-FID check against photo_stats.npz
under that convention returns 0.0106, and under ImageNet normalization it does
not. Both conventions are internally consistent and neither is wrong, but they
are different scales, and this one is the graded one. A number produced by any
other pipeline cannot be compared against the leaderboard or against a teammate's
result, so it does not belong in the metrics report.

"MiFID" here is the official script's definition: the mean cosine distance
between the i-th sorted real feature and the i-th sorted generated feature. The
pairing is positional and therefore arbitrary, and there is no memorization
threshold, so despite the name it is not the Frechet-distance-penalized metric
from the Kaggle Monet competition. It is computed this way because this is what
the grader computes.

Usage:
    python evaluate_local.py \
        --real-monet  path/to/monet_jpg \
        --real-photo  path/to/photo_jpg \
        --gen-a2b     outputs/pred_A2B \
        --gen-b2a     outputs/pred_B2A \
        --out-csv     submission.csv
"""

from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import scipy.linalg
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from scipy.spatial.distance import cosine
from tqdm import tqdm

N_EVAL = 300
BATCH_SIZE = 32

# ImageNet normalization, not [-1,1]. See the module docstring.
INCEPTION_TF = T.Compose([
    T.Resize(299),
    T.CenterCrop(299),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def list_images(folder: str) -> list[str]:
    exts = (".jpg", ".jpeg", ".png")
    paths: list[str] = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    return sorted(set(paths))


def take_n(paths: list[str], n: int | None) -> list[str]:
    return paths if n is None else paths[:min(n, len(paths))]


def get_inception_model(device: torch.device) -> nn.Module:
    inception = models.inception_v3(
        weights=models.Inception_V3_Weights.IMAGENET1K_V1,
        transform_input=False,
    )
    inception.fc = nn.Identity()
    inception.to(device).eval()
    return inception


def load_batch(paths: list[str]) -> torch.Tensor:
    return torch.stack(
        [INCEPTION_TF(Image.open(p).convert("RGB")) for p in paths], dim=0
    )


@torch.no_grad()
def get_activations(model: nn.Module, image_paths: list[str],
                    device: torch.device, batch_size: int = BATCH_SIZE) -> np.ndarray:
    feats = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Inception activations"):
        x = load_batch(image_paths[i:i + batch_size]).to(device)
        feats.append(model(x).detach().cpu().numpy())
    return np.concatenate(feats, axis=0)


def frechet_distance(mu1, sigma1, mu2, sigma2, eps: float = 1e-6) -> float:
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    diff = mu1 - mu2
    return float(diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean))


def calculate_fid_mifid(real_paths: list[str], gen_paths: list[str],
                        device: torch.device, batch_size: int = BATCH_SIZE,
                        subsample_to_match: bool = True) -> tuple[float, float]:
    real_paths, gen_paths = sorted(real_paths), sorted(gen_paths)
    if subsample_to_match:
        n = min(len(real_paths), len(gen_paths))
        real_paths, gen_paths = real_paths[:n], gen_paths[:n]
    model = get_inception_model(device)
    real_act = get_activations(model, real_paths, device, batch_size)
    gen_act = get_activations(model, gen_paths, device, batch_size)

    mu_r, sig_r = real_act.mean(axis=0), np.cov(real_act, rowvar=False)
    mu_g, sig_g = gen_act.mean(axis=0), np.cov(gen_act, rowvar=False)
    fid = frechet_distance(mu_r, sig_r, mu_g, sig_g)

    m = min(len(real_act), len(gen_act))
    mifid = float(np.mean([cosine(real_act[i], gen_act[i]) for i in range(m)]))
    return fid, mifid


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-monet", required=True, help="real domain A (monet_jpg)")
    p.add_argument("--real-photo", required=True, help="real domain B (photo_jpg)")
    p.add_argument("--gen-a2b", required=True, help="generated Monet->Photo")
    p.add_argument("--gen-b2a", required=True, help="generated Photo->Monet")
    p.add_argument("--out-csv", default="submission.csv")
    p.add_argument("--n-eval", type=int, default=N_EVAL)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    for d in (args.real_monet, args.real_photo, args.gen_a2b, args.gen_b2a):
        if not os.path.isdir(d):
            raise SystemExit(f"Missing folder: {d}")

    real_monet = take_n(list_images(args.real_monet), args.n_eval)
    real_photo = take_n(list_images(args.real_photo), args.n_eval)
    gen_a2b = take_n(list_images(args.gen_a2b), args.n_eval)
    gen_b2a = take_n(list_images(args.gen_b2a), args.n_eval)

    print("\nCounts:")
    print("Real Monet:", len(real_monet), "| Gen Monet (B2A):", len(gen_b2a))
    print("Real Photo:", len(real_photo), "| Gen Photo (A2B):", len(gen_a2b))

    print("\n Evaluating Photo -> Monet (B2A) ")
    fid_b2a, mifid_b2a = calculate_fid_mifid(real_monet, gen_b2a, device, args.batch_size)
    print(f"[Photo->Monet] FID={fid_b2a:.3f}  MiFID={mifid_b2a:.4f}")

    print("\n Evaluating Monet -> Photo (A2B) ")
    fid_a2b, mifid_a2b = calculate_fid_mifid(real_photo, gen_a2b, device, args.batch_size)
    print(f"[Monet->Photo] FID={fid_a2b:.3f}  MiFID={mifid_a2b:.4f}")

    submission = pd.DataFrame([{
        "ID": 1,
        "FID": (fid_a2b + fid_b2a) / 2,
        "MiFID": (mifid_a2b + mifid_b2a) / 2,
    }])
    submission.to_csv(args.out_csv, index=False)
    print("\n" + "=" * 45)
    print(submission.to_string(index=False))
    print("=" * 45)
    print("submission.csv saved to", args.out_csv)


if __name__ == "__main__":
    main()
