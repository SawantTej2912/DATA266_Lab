# Reproducibility

What is guaranteed, what is not, and how to check either.

---

## The short version

```bash
python scripts/verify_gpu_ready.py        # is this checkout runnable?
python scripts/verify_submission.py       # do the reported numbers agree with each other?
```

The first should pass before a run. The second passes only after results are folded back
into the prose, notebooks and report — so it failing after a fresh run is the expected
state, and the list of failures is the to-do list.

---

## What is claimed

**Configuration.** Every run is driven entirely by a YAML file. No hyperparameter is
hardcoded in a training script, and no path in any config is absolute — `verify_gpu_ready`
walks every `config*.yaml` and fails on any value starting with `/` or `C:\`.

**Seeding.** One `seed` field per config seeds `random`, `numpy`,
`torch`, `torch.cuda` and `torch.mps`. The field is 9015 for Parts 1 and 2 and 42 for
Part 3, whose notebook predates the repository convention; what matters is that every
generator reads from one field rather than that the three agree. Part 3 draws from the
two domains independently — an epoch is a fixed 1,000 steps, not one pass over the
smaller domain — so there is no pairing between a Monet and a photograph to keep stable
across processes.

**Splits.** Part 1 splits at the *story* level before cutting windows, so no story's
characters appear on both sides; `verify_gpu_ready` probes eight 200-character windows of
the validation text against the training text to confirm. Part 2's vocabulary is fitted on
training documents only — checked both by reading the argument passed to
`Vocab.from_docs` from the AST and by confirming that an unseen token encodes to `UNK`
rather than acquiring an id.

**Provenance.** Every run writes a `runtime_<tag>.json` beside its artifacts containing
the device, device class, torch and CUDA versions, GPU name, capability and memory,
precision mode, seed, batch size, worker count, dataset sizes, parameter count, Python
version, platform, and the git commit with a dirty flag. Nothing about a reported number
has to be remembered.

**Checkpoint self-sufficiency.** A checkpoint contains everything needed to score it
without the repository's state:

- Part 1: weights, model config, full run config, seed, block size, **both directions**
  of the character↔index mapping, history, and the provenance block. The `*_resume.pt`
  variant adds the optimiser state, global step, epoch and the schedule description.
- Part 2: weights, the `itos` vocabulary list, the preprocessing configuration, the data
  configuration, the seed, the architecture spec, the **label mapping**, and the config
  fingerprint.
- Part 3: both generators plus, in the epoch checkpoints, both discriminators, both
  optimisers, both schedulers, the EMA shadow weights and the image pool state.

The reason is specific: a state dict plus a vocabulary size cannot score a single review,
because token ids are meaningless without `itos` and the same text tokenises differently
under a different stemmer or stopword policy. Keeping them in separate files means a
checkpoint can be paired with the wrong vocabulary and then produces confident nonsense
rather than an error.

**Caches are verified, not trusted.** Part 2's preprocessing cache carries a SHA-256
fingerprint of the data config, the preprocessing config and the seed. On load, the
fingerprint is checked *and* the tensors inside are checked against the config's split
sizes, sequence width and vocabulary cap. The fingerprint alone is not enough: it is
computed from the config, so a pickle that was truncated or written by older code still
carries a matching key, and evaluating against it would silently report metrics for a
different dataset.

---

## What is not claimed

**Bit-exact GPU reproducibility.** Two runs with the same seed on the same GPU can differ
in the last few decimal places. Non-deterministic cuDNN kernel selection, atomic
accumulation order in reductions, and TF32 on the fp32 operations outside autocast all
contribute. `--deterministic` pins the cuDNN algorithm choice and narrows this, but is off
by default because it forbids the autotuner and costs throughput on a run sized to a
wall-clock budget.

**Cross-device reproducibility.** An MPS-trained model and a CUDA-trained model from the
same seed are different models. They are *comparable*, because evaluation always runs in
fp32 in all three parts, but they are not the same. This is why `device_class` is a column
in every metrics CSV and why `HARDWARE_DISCLOSURE.md` separates machine A from machine B.

**Part 2 resume.** There is none. At 25–40 minutes for all three arms it is cheaper to
restart, and claiming otherwise would be a false capability statement. Parts 1 and 3 both
have working `--resume auto`.

**Run-to-run variance in the Part 2 comparison.** The reported bootstrap confidence
intervals measure test-set sampling error, and the McNemar tests measure disagreement
between two fixed prediction sets. Neither measures variance from initialisation and
shuffling. `scripts/seed_sweep_task2.py` exists to measure that and has not yet been run
at full scale — until it has, the architecture ranking is a single-seed observation.

---

## Reproducing each part from scratch

Datasets first. `DATA_MANIFEST.md` lists every path with its expected count, and
`python scripts/fetch_data.py` restores anything missing.

```bash
pip install -r requirements.txt
# On a CUDA machine install torch from an index new enough for the card first --
# an RTX 5090 is sm_120 and needs cu128 or newer:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python scripts/verify_gpu_ready.py
```

### Part 1 — character-level TinyStories GPT

```bash
python task1_llm/shriram_dundigalla/src/train.py \
    --config task1_llm/shriram_dundigalla/src/config_gpu.yaml --require-cuda
python task1_llm/shriram_dundigalla/src/evaluate.py --checkpoint <tag>.pt
```

Produces `metrics_report.csv`, the loss and stability figures, the generation sweep and
the decoding comparison. The tokenizer, the causal mask and every Transformer block are
implemented in `model.py`; `verify_gpu_ready` confirms by AST that no prebuilt attention
or Transformer module is used, which a text search cannot do because `model.py`'s own
docstring names `nn.MultiheadAttention` in order to say it is unused.

### Part 2 — Yelp Polarity sentiment

```bash
python task2_sentiment/shriram_dundigalla/src/train.py --require-cuda
python task2_sentiment/shriram_dundigalla/src/error_review.py
```

One command trains all three arms from the same cached preprocessing and writes the
three-row report, the McNemar tests, the slice metrics and the figures. Run it this way
rather than as three `--model` invocations: the paired test needs arms that saw
byte-identical inputs.

### Part 3 — CycleGAN photo→Monet

```bash
# Training is the notebook, run top to bottom on a CUDA machine:
#   task3_gan/shriram_dundigalla/src/task3_cyclegan_unet.ipynb
python task3_gan/shriram_dundigalla/evaluate_local.py \
    --real-monet task3_gan/data/monet_jpg \
    --real-photo task3_gan/data/photo_jpg \
    --gen-a2b    task3_gan/shriram_dundigalla/outputs/pred_A2B \
    --gen-b2a    task3_gan/shriram_dundigalla/outputs/pred_B2A
```

`evaluate_local.py` is the course evaluation script with its paths as arguments, and it
is the authority for the reported FID and MiFID. It does not read the shipped
`real_stats.npz`; it recomputes its reference from the real image folders under ImageNet
normalization, caps each side at 300 sorted filenames, and averages the two directions.
A FID computed any other way is on a different scale and is not comparable to the
leaderboard.

---

## The evidence trail

`reproducibility/raw_logs/` holds every run's log, including the MPS runs, the smoke
runs and the failed runs. **Nothing in it is edited or deleted after the fact.** A log
that contradicts a later reported number is kept and explained rather than removed — for
example, the Part 1 baseline log shows a training CE of 0.7110 measured over the first
100 batches, while the reported 0.7089 is the full-split figure; the manifest states both
and why they differ.

`reproducibility/manifests/` holds the per-member manifest: what was run, on what, with
which seed, and what the headline numbers were.

Three metric CSVs are the machine-written record. Every number in every `results.md`,
notebook and in `Report.pdf` is read from them at build time or checked against them by
`verify_submission.py`, which compares the prose at the prose's own rounding rather than
against a hardcoded expected value. That check has caught three real drifts: a stale
cross-entropy in the manifest, an ECE quoted as 0.0204 where the CSV said 0.0203, and a
notebook still capping training metrics at 100 batches.

---

## Superseding an MPS result with a CUDA result

Do not overwrite. Add the CUDA row, keep the MPS row, and label both — the
`device` column exists for this. `verify_gpu_ready` fails if one CSV ends up containing rows from two device
classes without them being labelled, and `verify_submission.py` fails if `Report.pdf` is
older than any CSV it quotes, so a forgotten rebuild cannot pass silently.
