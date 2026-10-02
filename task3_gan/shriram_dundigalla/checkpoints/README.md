# Part 3 checkpoint — why the weights are not in git

The checkpoint behind every Part 3 number in this repository is not committed, because
GitHub rejects any single file over 100 MB and this one cannot be made to fit without
changing the weights.

| | |
|---|---|
| File | `ckpt_epoch060.pth` |
| Size | 1,372,095,715 bytes (1.3 GB) |
| SHA-256 | `d44cb9e7e68096b06e9dd36f8b1971e87584b53018964df2131794b70f1e4ec6` |
| Contents | `epoch`, `G_M2P`, `G_P2M`, `D_P`, `D_M`, `opt_G`, `opt_D_P`, `opt_D_M`, `sched_G`, `sched_D_P`, `sched_D_M`, `scaler` |
| Epoch | 60 |

Stripping it to inference weights does not help enough. Each generator is 54,404,099
parameters, so one generator alone is 217.6 MB in fp32 — still over the limit. Casting
to fp16 brings a generator to 108.8 MB, which is *also* over the limit, and would change
the weights: the reported FID of 101.44 was produced by the fp32 values, so an fp16 copy
would not reproduce it and would make this folder disagree with `metrics_report.csv`.
Splitting the file into sub-100 MB parts would work mechanically but adds roughly 435 MB
of binary fragments to a repository that is already near GitHub's 1 GB guidance.

So the weights travel outside git, and the things that make them verifiable travel
inside it:

- `src/task3_cyclegan_unet.ipynb` — the code that produced them, with outputs
- `src/config.yaml` — the exact settings, extracted from the notebook's `CONFIG`
- `outputs/pred_A2B/` (300) and `outputs/pred_B2A/` (7,038) — what the generators
  actually emitted, which is what `evaluate_local.py` scores
- `logs/metrics_per_epoch.csv` and `logs/train.log` — the unedited training record
- `../../reproducibility/manifests/shriram_dundigalla_manifest.md` — maps this
  checkpoint to the results it produced

A grader does not need the weights to check any reported number: `evaluate_local.py`
reads the committed images, not the checkpoint. The weights are needed only to generate
*new* images, and they are available on request and will be brought to the demo.

Tejas's Part 3 generators *are* committed, at 45.5 MB each. That is not a difference in
practice — his ResNet-9 generator is 11,378,179 parameters against this U-Net's
54,404,099, so his fits and this one does not.
