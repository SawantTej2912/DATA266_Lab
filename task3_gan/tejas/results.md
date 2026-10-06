# Task 3 — CycleGAN Monet ↔ Photo (Tejas)

## Final model (i270k)

| Component | Choice |
|---|---|
| Generators (×2) | ResNet-9 (Johnson et al.; CycleGAN): c7s1-64 → d128 → d256 → 9 residual blocks → **2× resize-convolution upsampling** → c7s1-3 → tanh; InstanceNorm, reflection padding. 11,378,179 parameters each |
| Discriminators (×2) | 70×70 PatchGAN, InstanceNorm, LeakyReLU 0.2. 2,764,737 parameters each (28.3M total) |
| Losses | LSGAN adversarial + cycle L1 (λ_cyc = 10) + identity L1 |
| Identity weights | Monet→Photo generator: λ_id = 5 (constant). **Photo→Monet generator: 5, then linearly to 1.5 over the decay phase** |
| Optimisation | Adam (β = 0.5, 0.999); generator LR **2e-4**, discriminator LR **7.5e-5**; batch 1; image pool of 50 |
| Regularisation | **DiffAugment** (colour, translation, cutout) on every discriminator input; **EMA** of generator weights (decay 0.999) used for all outputs |
| Schedule | 900 iterations/epoch × 300 epochs = **270,000 iterations**; constant LR for 45% (until epoch 135), then linear decay to 0 |
| Data / augmentation | 300 Monets, 7,038 photos; resize 286 → random crop 256 → horizontal flip; each step pairs the next Monet with a random photo |
| Model selection | EMA snapshots every 10 epochs, scored with the TA's FID formula; **selected: epoch 280** (epochs 270–300 all score 98.7–99.2, so it isn't a one-off) |

## Experiment history: how the FID went from 147 to 98.7

| Run | Iterations | Change, and why | Best FID (Monet→Photo / Photo→Monet) | Kaggle |
|---|---|---|---|---|
| quick30 | 9K | Baseline: paper configuration, transposed-conv upsampling | 146.97 (142.8 / 151.1) | −73.70 |
| e200 | 60K | **Resize-convolution** (quick30 showed checkerboard artifacts); longer schedule; snapshot selection | 114.23 (122.3 / 106.2) | −57.33 |
| e300 | 90K | **DiffAugment + slower D (1e-4) + EMA.** In e200 the discriminator dominated: D loss ~0.06, real/fake 0.86/0.14, generator adversarial loss rising | 110.70 (115.6 / 105.8) | −55.56 |
| i180k | 180K | Same setup, 2× longer. e300's "plateau" turned out to be its LR reaching 0, not a capacity limit | 103.09 (104.7 / 101.5) | −51.75 |
| **i270k (final)** | **270K** | **Direction-specific identity ramp** (Photo→Monet had plateaued at ~101 while Monet→Photo kept improving) + D LR 7.5e-5 (slow D creep) + 45/55 schedule | **98.70 (100.92 / 96.49)** | **−49.55** |

Comparison plot of all runs: `outputs/experiments_comparison.png` · per-run evidence in `exploratory/`.

Some cells in the exploratory e200, e300 and i180k notebooks have no execution count because they were skipped on purpose during those runs (e.g. export/backup cells); none produced errors.

## Final metrics (selected snapshot, both directions)

| Metric | Monet→Photo (A2B) | Photo→Monet (B2A) |
|---|---|---|
| FID (TA formula, n = 300) | 100.92 | **96.49** |
| "MiFID" (TA definition) | mean 0.4106 | |
| KID | 0.0181 | 0.0066 |
| Precision / recall | 0.747 / 0.483 | 0.543 / 0.720 |
| Cycle-reconstruction L1 | 0.0785 | 0.0849 |
| LPIPS (input vs. output) | 0.322 | 0.364 |
| Content cosine similarity | 0.812 | 0.796 |

**Submission:** FID 98.703, MiFID 0.4106 → **Kaggle public score −49.55, rank 20** (at submission time).
**Training:** 0 NaNs; 507 min on an RTX 4090 (~18 img/s); final cycle loss 0.146, identity loss 0.140.
**Curves:** `outputs/loss_curves.png` (adversarial, cycle/identity, discriminator outputs, gradient norms, LR), `outputs/fid_vs_epoch.png`, `outputs/epoch_history.csv`.

## Cycle-consistency verification
`outputs/cycle_verification.png` shows Monet → photo → Monet and photo → Monet → photo for fixed samples: scenes are reconstructed closely. The cycle L1 on 300 held images is 0.078 / 0.085 (in [−1, 1] pixel space), the lowest of all runs.

## Training stability
- **Discriminator balance:** with DiffAugment and the slower discriminator, the D losses stayed at 0.14–0.21 and D_A's real/fake outputs widened only slowly, from 0.62/0.38 to **0.70/0.30** over 270K iterations. Without the fix (e200), they reached 0.86/0.14 within 60K. A slow creep remains: a limitation for even longer runs.
- **Peak-LR noise:** snapshots during the constant-LR phase fluctuate (e.g. epoch 140: 111.9 after 110.5). The decay phase gives smooth, steady gains: FID 106.3 → 98.7.
- **The identity ramp worked:** after the Photo→Monet identity weight began falling (epoch 136), its FID dropped from ~104 to **96.5**, breaking i180k's ~101 plateau. Monet→Photo, whose weight stayed at 5, improved more slowly (108 → 101), consistent with the longer schedule alone.

## Human audit (blinded, 30 fixed samples, 2 raters)
Raters saw anonymised `input | output` pairs (S01–S30) without the direction key.

| Criterion | Mean | Monet→Photo / Photo→Monet | Agreement | Cohen's κ |
|---|---|---|---|---|
| Style (1–5) | 4.33 | 4.10 / 4.57 | 73% | 0.84 (quadratic) |
| Content (1–5) | 4.70 | 4.60 / 4.80 | 100% | 1.00 (ceiling effect) |
| Visible artifacts | 55% of samples | 53% / 57% | 83% | 0.66 |

Human judgments agree with FID: Photo→Monet is rated more convincing (4.57 vs. 4.10). Content preservation is rated very high, matching the low cycle L1. Perfect content agreement partly reflects a ceiling effect, since almost every image scored 4–5. Visible artifacts in about half the samples (speckle in bright skies, halos around the sun) are the main remaining weakness. With n = 30 and two raters, these κ values have wide uncertainty.

## Evaluation integrity
- `submission.csv` was regenerated with the TA's `Part3_Evaluation_Script.ipynb`, run **unmodified** on the final outputs. It matches our `evaluate_local.py` to < 1e-6 (`outputs/ta_eval_script_executed.ipynb`, `outputs/submission_ta_script.csv`).
- Submitted images are the direct output of this CycleGAN. Pretrained networks (Inception-v3, AlexNet for LPIPS) are used **only for evaluation**.
- **Caveats:** evaluation is **in-sample** (the scored Monets and photos are also training data, per the competition protocol), and the snapshot was **selected by that same FID** among 30 candidates, so the reported score is slightly optimistic as an estimate of generalisation.

## Limitations and next steps
- Speckle and halo artifacts in bright regions; a slow late-training discriminator advantage.
- No held-out photos to measure generalisation; next: reserve a held-out photo set.
- Next experiments: paper-scale training (~1.4M iterations, in progress as an exploratory run), stronger discriminator regularisation (R1 or spectral norm), a lower identity weight for Photo→Monet.

## Reproducibility
Final notebook `src/task3_cyclegan.ipynb` · evaluation `evaluate_local.py` · weights `checkpoints/G_AB.pt`, `G_BA.pt` (EMA, weights only) · raw logs and manifests in `reproducibility/*/task3_gan/tejas/` (one subfolder per exploratory run) · `last.pt` and snapshots kept off-git (backup).