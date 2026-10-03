# Task 3 — Failure analysis

What did not work, and what the numbers say about why.

## 1. DiffAugment made the model measurably worse

Differentiable augmentation applied to every discriminator input is the standard
remedy for a discriminator that memorizes a small real set, and with only 300
Monet paintings against 7,038 photos that looked like exactly the right tool.
It was not.

| | baseline (run 1) | DiffAugment (run 3) |
|---|---|---|
| FID (averaged) | 101.44 | **112.97** |
| Generator gradient norm, mean | 27.0 | **137.0** |
| Generator gradient norm, max | 112.5 | **640.5** |
| Non-finite losses | 0 | 0 |

FID got 11.4 points worse, and the gradient norms tell the story: the mean rose
5x and the maximum rose 5.7x. The run never produced a NaN, so this is not a
numerical blow-up -- it is a genuinely harder optimization problem. Augmenting
the discriminator's inputs means the generator is chasing a target that moves
under colour, translation and cutout jitter on every step, and the gradient it
receives is correspondingly noisier. With a 54M-parameter U-Net at batch size 1
there is no averaging to damp that noise.

The lesson is that DiffAugment addresses discriminator *overfitting*, and this
baseline was not overfitting. The run-1 discriminator loss sits near 0.26 and
never approaches 0, which means it was never winning in the first place.
Applying a remedy for a problem the model did not have cost 11 FID points.

## 2. Loss-weight tuning changed essentially nothing

Run 2 altered the cycle and identity weights and retrained from scratch for 80
epochs. FID went from 101.44 to 101.51 -- a 0.07 difference, far inside the
run-to-run noise that re-encoding the same images at a different JPEG quality
produces.

This is a negative result worth stating plainly: within the range tried, the
lambda values were not the binding constraint. Two full retraining runs bought
no improvement, which suggests the bottleneck was architectural or in the
quantity of Monet data, not in the balance between the loss terms.

## 3. The best checkpoint was epoch 60, not epoch 100

Training ran the full 100 epochs with linear LR decay from epoch 50, and the
checkpoint that scored best was `ckpt_epoch060.pth`. The last 40 epochs of
training -- roughly 40% of a 2.77-hour run -- did not improve the graded metric.

The training losses do not show this. Cycle loss is still drifting gently
downward at epoch 99 (0.0935) and the generator loss is flat. Lower
reconstruction error simply does not imply a better match between the generated
and real feature distributions, which is what FID measures. Training loss was
not a usable model-selection signal here, and periodic FID evaluation on
checkpoints was the only thing that caught it.

## 4. The two directions fail in opposite ways

| | Monet -> Photo | Photo -> Monet |
|---|---|---|
| Precision | 0.6833 | 0.5000 |
| Recall | 0.4433 | 0.7300 |
| Density | 0.8627 | 0.3847 |
| LPIPS | 0.1806 | 0.2498 |

Monet -> Photo has high precision and low recall: the generated photos land
convincingly inside the real-photo distribution, but they cover only part of
it. That is the expected signature of a conservative generator -- it has found a
safe region of photo-space and stays there.

Photo -> Monet is the mirror image, and the density of 0.3847 is the weakest
number in the entire report. Low density with high recall means the generated
"Monets" are spread widely but sit in sparse regions of the real Monet manifold
-- they are varied, but they are not often convincingly Monet-like. The higher
LPIPS in this direction (0.2498 vs 0.1806) says the model is changing the image
more, and the density says those changes are not landing on target.

This is the clearest limitation of the U-Net choice. Skip connections hand the
decoder the encoder's spatial features directly, which is excellent for keeping
the composition intact and is why content cosine is a healthy 0.7756. But
turning a photograph into a Monet requires discarding fine detail and replacing
it with brushwork, and a skip connection's entire purpose is to stop fine detail
from being discarded. The architecture that makes the easy direction easy is the
one that makes the hard direction hard.

## 5. Two incompatible FID conventions, and the cost of finding out late

The reference `.npz` files shipped with the dataset were built by scaling inputs
to [-1,1]; a self-FID check against `photo_stats.npz` under that convention
returns 0.0106, confirming it. The official evaluation script does not use those
files at all. It uses ImageNet normalization, recomputes reference statistics
from the real image folders, truncates both sides to 300 sorted filenames, and
averages the two directions.

Both pipelines are internally consistent and neither is incorrect, but they
produce numbers on different scales, and only one of them is graded. Any FID
computed by the other route cannot be compared against the leaderboard or
against a teammate's figure. Every number in this submission comes from
`evaluate_local.py`, which is the official script with its paths parameterized,
so this ambiguity cannot recur.

## 6. Known gaps

- **Human audit not yet run.** It requires two raters and is the one required
  metric still missing. The metrics files carry `pending human audit` rather
  than a placeholder number.
- **Training-time peak GPU memory was never logged.** Only the inference figure
  (1882 MB) is measured, and it is labelled as such.
- **No seed sweep.** All three runs used seed 42, so the 0.07 FID gap between
  runs 1 and 2 cannot be formally separated from seed variance. The conclusion
  that lambda tuning did not help is therefore suggestive rather than
  established.
