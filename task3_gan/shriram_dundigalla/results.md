# Task 3 — CycleGAN, Monet <-> Photo (U-Net generator)

Member: Shriram Dundigalla
Notebook: `src/task3_cyclegan_unet.ipynb`
Scoring: `evaluate_local.py` (the official evaluation script, paths parameterized)

## Headline

| | |
|---|---|
| Submitted FID (averaged over both directions) | **101.44** |
| Submitted MiFID | **0.4172** |
| Kaggle class competition rank | **20** (leaderboard band 11-20 = 9 points) |
| Selected checkpoint | `ckpt_epoch060.pth` |

## Architecture

Two generators and two discriminators, trained jointly.

**Generator — U-Net, depth 8, `ngf=64`, 54,404,099 parameters each (54.40M).** Eight
stride-2 encoder blocks take 256x256 down to 1x1, eight transposed-convolution
decoder blocks bring it back, and each decoder block concatenates the mirrored
encoder feature map along the channel axis. InstanceNorm throughout except the
first encoder block and the bottleneck; LeakyReLU(0.2) in the encoder, ReLU in
the decoder; dropout 0.5 in the first three decoder blocks; Tanh output.

The choice of U-Net over ResNet is the substantive architectural decision here,
and it is not cosmetic. Both architectures have to solve the same problem --
change the style without losing the content -- and they solve it by different
mechanisms. A ResNet generator preserves content because every residual block
learns a *delta* on top of an identity path, so the default behaviour of an
untrained block is to pass the image through. A U-Net preserves content because
the decoder is handed the encoder's spatial feature maps directly, so location
information never has to survive the bottleneck at all. The practical
consequence is visible in the metrics below: the U-Net's skips keep geometry
very well and make it comparatively reluctant to repaint large regions, which
helps content preservation and hurts the more aggressive stylization that the
Monet direction rewards.

**Discriminator — 70x70 PatchGAN, `ndf=64`, 2,763,841 parameters each.**
C64-C128-C256-C512 then a 1-channel convolution, producing a 30x30 score map
rather than a scalar. No sigmoid, because the adversarial loss is LSGAN (MSE)
and operates on raw outputs.

Total trainable parameters: **114.34M**.

## Training setup

| | |
|---|---|
| Loss | LSGAN (MSE) adversarial + cycle L1 (lambda 10) + identity L1 (lambda 5) |
| Optimizer | Adam, lr 2e-4, beta1 0.5, beta2 0.999 |
| Schedule | constant lr for 50 epochs, then linear decay to 0 over the next 50 |
| Epochs / steps | 100 epochs x 1000 steps = 100,000 generator updates |
| Batch size | 1 (standard for CycleGAN; InstanceNorm behaves badly at larger batch) |
| Augmentation | resize 286, random crop 256, random horizontal flip |
| Stability aids | 50-image replay buffer per discriminator, Normal(0, 0.02) init |
| Precision | AMP |
| Seed | 42 (Python, NumPy, torch, CUDA all seeded) |
| Hardware | NVIDIA A100 (Colab) |

Note on comparing against a teammate: an "epoch" here is 1000 steps, not 300.
Epoch 60 of this run is 60,000 generator updates. Any cross-member comparison
should be made on total steps or wall clock, not on epoch number.

## Results

### Distribution metrics (official script, n=300 per direction)

| | Monet -> Photo (A2B) | Photo -> Monet (B2A) | averaged |
|---|---|---|---|
| FID | 103.92 | 98.95 | **101.44** |
| MiFID | 0.4241 | 0.4104 | **0.4172** |
| KID | 0.0191 | 0.0082 | 0.0136 |

### Fidelity and diversity

| | Monet -> Photo | Photo -> Monet |
|---|---|---|
| Precision | 0.6833 | 0.5000 |
| Recall | 0.4433 | 0.7300 |
| Density | 0.8627 | 0.3847 |
| Coverage | 0.7900 | 0.6933 |
| LPIPS (input vs translation) | 0.1806 | 0.2498 |

### Cycle and content

| | |
|---|---|
| Cycle-reconstruction L1 | 0.0482 |
| Content-preservation cosine (input vs translation) | 0.7756 |
| Cycle loss, final epoch | 0.093547 |
| Identity loss, final epoch | 0.048619 |
| Adversarial loss, final epoch | 1.140634 |

### Efficiency

| | |
|---|---|
| Training time | 2.77 h |
| Seconds per epoch | 99.6 |
| Throughput | 10.04 images/sec |
| Peak GPU memory (inference) | 1882 MB |

Training-time peak memory was not logged; only the inference figure is
measured, and it is reported as such rather than estimated.

### Stability

Gradient norm on the generator averaged 27.0 with a maximum of 112.5 across all
100 epochs, and there were **zero** non-finite loss values. The generator and
discriminator losses settle rather than diverge: G falls from 5.06 at epoch 0 to
roughly 2.3 and stays there, D sits near 0.26, and neither collapses toward 0 or
runs away. This is the signature of a GAN that reached a stable equilibrium
rather than one where the discriminator won.

- `outputs/figures/loss_curves_run1.png` — generator vs discriminator loss, the
  three generator loss components, generator gradient norm, and seconds per
  epoch, with the LR-decay point at epoch 50 marked.
- `outputs/figures/run1_vs_run3_stability.png` — run 1 against run 3, which is
  where the DiffAugment gradient blow-up is visible.

## Experiment comparison

Three full training runs were done, all scored with the same official script.

| Run | Change | FID (avg) | Grad norm mean / max | NaN |
|---|---|---|---|---|
| 1 | baseline | **101.44** | 27.0 / 112.5 | 0 |
| 2 | loss-weight tuning (new lambdas), 80 epochs | 101.51 | - | 0 |
| 3 | DiffAugment on all discriminator inputs, 80 epochs | 112.97 | 137.0 / 640.5 | 0 |

Run 1 was submitted. The reasoning for each is in `failure_analysis.md`.

## A note on which FID this is

The number above comes from the official evaluation script and is not
interchangeable with a general-purpose FID. That script uses ImageNet
normalization, `Resize(299)` + `CenterCrop(299)`, reference statistics
recomputed from the real image folders, 300 images per side, and the average of
both directions. A standard implementation that scales inputs to [-1,1] and
reads the shipped `real_stats.npz` produces a different number on a different
scale -- neither is wrong, but only one is graded. `evaluate_local.py`
implements the graded one, and every figure in this document was produced by it.

The quantity the script calls MiFID is the mean cosine distance between the
i-th sorted real feature and the i-th sorted generated feature. The pairing is
positional and there is no memorization threshold, so despite the name it is not
the memorization-penalized MiFID from the Kaggle Monet competition. It is
reported here as defined because that is what the leaderboard scores.

## Outstanding

- **Human audit.** 30 fixed samples, two raters, blinded, scored on style /
  content / artifacts, with Cohen's kappa. Not yet run; it needs both team
  members. Placeholders in the metrics files read `pending human audit` rather
  than carrying a fabricated value.
