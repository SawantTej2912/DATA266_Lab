# Working on this repository

DATA266 Fall 2026 Lab 1, a three-part graded submission. Parts 1 and 2 have complete
results already; Part 3 is scored on Kaggle and is the part still being improved.

If you are seeing this project for the first time, read
[`README.md`](README.md) first and [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) second.
Everything runs from this directory, and every path in every config is relative to it.

---

## Rules that carry academic consequences

These are not style preferences. Breaking any of them invalidates the submission.

**Part 3's submitted images must be the direct output of the trained CycleGAN in this
repository.** No hand-picked, manually edited, copied or externally sourced images. No
hardcoded or looked-up outputs. No training on or inspecting test-set pairings. **No
pretrained or foundation image model may generate or touch a submitted image.**
Violating this scores zero for Part 3. Pretrained Inception and AlexNet are used to
*score* FID, KID and LPIPS, which the rubric requires and which is fine — the
distinction is scoring versus generating.

**Part 1 must stay from scratch.** No `nn.Transformer`, `nn.TransformerEncoder`,
`nn.MultiheadAttention` or any prebuilt attention module. The char tokenizer, the
token/position embeddings, the causal mask and the Pre-LN blocks are all hand-written
and must remain so. Note that `model.py`'s docstring *names* `nn.MultiheadAttention` in
order to say it is unused — a text search will flag that file; an AST check will not.

**Part 2 must not use pretrained embeddings or pretrained language models.** The
vocabulary is built from the training split only.

**No personal file paths, credentials, API keys or tokens in any file in this
repository** — including logs, notebook outputs and commit messages. A Kaggle token
exists on the author's machine outside this tree and must stay there. Log paths relative
to the repository root, never absolute. The bundle builder scans for secrets and will
refuse to package if it finds any.

---

## Rules about evidence

**Never fabricate a metric.** If a run has not happened, write `pending CUDA run`. Do
not copy a number from a different device, a different config or a different seed into
a table as though it were measured.

**Never delete or overwrite a raw log.** Everything in `reproducibility/raw_logs/` is
the evidence trail for the report, including the smoke-test logs. If an old result is
superseded, label it historical and add the new one beside it — do not replace it.
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) describes how to supersede a result properly.

**Never mix devices in one table without labelling.** Every result currently in this
repository was produced on Apple MPS and is labelled `Apple MPS`. CUDA results are
labelled `CUDA GPU`. Both rows stay, both labelled, with the submitted one identified.
`device_class` exists in all three `metrics_report.csv` files for exactly this reason.

**Do not change model logic, hyperparameters or architecture** unless a broken path or
missing dependency prevents execution, or you are asked to. If you do change one,
say so plainly and say whether it invalidates the existing results.

---

## Running anything

The booked GPU is an **RTX 5090** — Blackwell, compute capability 12.0, `sm_120`, 32 GB.

Put `--require-cuda` on every command. It fails immediately with exit code 2 rather than
silently falling back to the CPU, which on this machine is the difference between a run
that finishes and a slot that is wasted.

`cu128` is the floor for the torch wheel. `cu124` and `cu126` install cleanly, report
`cuda_available = True`, and then fail at the first kernel launch because they contain no
`sm_120` kernels. `scripts/runtime.py` diagnoses this specifically.

Shared device selection, seeding and provenance live in `scripts/runtime.py`. Use it
rather than writing a new `pick_device()`; three near-duplicates were consolidated into
it so that `--require-cuda` means one thing everywhere.

## Before claiming anything works

```bash
python scripts/verify_gpu_ready.py          # 111 checks; add --smoke for all three tasks
python scripts/verify_submission.py         # submission completeness
```

`verify_gpu_ready.py` is written to prove rather than assert — it imports modules, runs
kernels, probes for train/validation leakage with real substrings, and reads the AST
instead of grepping text. It found three bugs in itself and several in the project. If
you add a check, make it measure something.

Four warnings are expected and not failures: three Part 2 checkpoints predate the
`label_map` field and the Part 1 checkpoint predates the dual-direction tokenizer field.
They clear on the next training run and must not be resolved by editing the checkpoints.
