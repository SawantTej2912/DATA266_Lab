"""Task 1 training driver: config-driven, logged, resumable.

Run from the repo root:
    python task1_llm/shriram_dundigalla/src/train.py --config .../config.yaml
    python task1_llm/shriram_dundigalla/src/train.py --smoke --require-cuda
    python task1_llm/shriram_dundigalla/src/train.py --resume auto --require-cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import data as data_mod           # noqa: E402
import metrics as metrics_mod     # noqa: E402
import runtime as rt              # noqa: E402
from model import GPT, GPTConfig  # noqa: E402

# Device selection, seeding and the provenance block all live in scripts/runtime.py so
# that all three tasks behave identically -- in particular so `--require-cuda` means the
# same thing everywhere and cannot drift between them.
pick_device = rt.select_device
device_name = rt.device_label
set_seed = rt.set_seed


def lr_at(step: int, total_steps: int, peak_lr: float, warmup_steps: int, min_lr_frac: float) -> float:
    """Linear warm-up, then cosine decay to min_lr_frac * peak_lr.

    Warm-up matters even with Pre-LN blocks: Adam's second-moment estimate is close to
    zero for the first handful of steps, so a full-size learning rate there produces
    an enormous effective step before the optimiser state has settled.
    """
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return peak_lr * (min_lr_frac + (1 - min_lr_frac) * cosine)


def build_optimizer(model, lr, weight_decay, betas):
    """Decay matmul weights only.

    LayerNorm gains, biases and embeddings are excluded: weight decay on them pulls
    the normalisation scale and unused embedding rows toward zero for no benefit.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 and "emb" not in name else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=tuple(betas)), len(decay), len(no_decay)


class RunLogger:
    """Writes an unedited append-only log; the lab grades this as the evidence trail."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def load_resume_state(resume: str, ckpt_dir: Path, run_name: str, log,
                      smoke: bool = False, force_fp32: bool = False) -> dict:
    """Load a resume checkpoint, either an explicit path or the newest one.

    `auto` orders by modification time, and a real run never resumes from a smoke
    checkpoint while a smoke run only ever resumes from one. Both halves matter: a
    lexical sort puts "s" after every digit, so a 2-epoch smoke checkpoint written
    minutes ago outranks a real epoch-10 one, and since the GPU runbook has you
    smoke-test before every session that is the normal state of the directory rather
    than a corner case. Part 3 shipped with exactly this bug.

    Precision is filtered the same way and for the same reason. The bf16 and fp32 arms
    of the precision comparison share a run_name, and resuming one from the other would
    silently make the comparison meaningless rather than fail.
    """
    if resume == "auto":
        cands = sorted(
            (p for p in ckpt_dir.glob(f"{run_name}_*resume.pt")
             if ("_smoke_" in p.name) == smoke
             and ("_fp32_" in p.name) == force_fp32),
            key=lambda p: p.stat().st_mtime)
        if not cands:
            raise FileNotFoundError(
                f"--resume auto found no {'smoke ' if smoke else ''}resume checkpoint "
                f"matching {run_name}_*resume.pt in {ckpt_dir}. One is written at the "
                f"end of every epoch, so this means no epoch has finished yet.")
        path = cands[-1]
    else:
        path = Path(resume)
        if not path.is_absolute():
            path = REPO_ROOT / path
    # Logged relative to the repo root. The lab forbids personal absolute paths in any
    # committed file, and a log is a committed file -- `--resume auto` resolves to an
    # absolute path, so printing it raw would write the home directory into the
    # evidence trail.
    try:
        shown = path.resolve().relative_to(REPO_ROOT)
    except ValueError:
        shown = f"<outside repo>/{path.name}"
    log(f"resume checkpoint={shown}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"optimizer_state_dict", "global_step", "epoch"} - set(state)
    if missing:
        raise ValueError(
            f"{path.name} is a final checkpoint, not a resume checkpoint: it is missing "
            f"{sorted(missing)}. Final checkpoints carry weights for evaluation only.")
    return state


def checkpoint_payload(*, model, optimizer, gcfg, cfg, tok, history, tag, epoch,
                       global_step, total_steps, tokens_seen, step_losses, grad_norms,
                       provenance) -> dict:
    """Everything needed to either evaluate or resume this run.

    The tokenizer goes inside the checkpoint rather than only in the sibling
    tokenizer.json, in both directions. A character-level GPT's weights are meaningless
    without the exact id->char mapping, and a checkpoint separable from its tokenizer can
    be paired with the wrong one and then generates plausible garbage instead of
    failing. Both directions of the mapping are stored because JSON keys are strings:
    reconstructing char->idx by inverting idx->char requires knowing to cast the keys
    back to int, and that is the sort of detail that gets missed.
    """
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": gcfg.to_dict(),
        "run_config": cfg,
        "seed": cfg["seed"],
        "block_size": gcfg.block_size,
        "history": history,
        "tag": tag,
        "epoch": epoch,
        "global_step": global_step,
        "total_steps": total_steps,
        "tokens_seen": tokens_seen,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
        "tokenizer": {"idx_to_char": dict(tok.idx_to_char),
                      "char_to_idx": {c: i for i, c in tok.idx_to_char.items()}},
        "vocab_size": tok.vocab_size,
        # No torch LR scheduler object exists to save: `lr_at()` is a closed-form
        # function of the global step, so `global_step` *is* the scheduler state.
        "lr_schedule": {"kind": "closed-form warmup+cosine, see lr_at()",
                        "peak_lr": cfg["train"]["lr"],
                        "warmup_frac": cfg["train"]["warmup_frac"],
                        "min_lr_frac": cfg["train"]["min_lr_frac"]},
        "provenance": provenance,
    }


def run(cfg: dict, smoke: bool = False, require_cuda: bool = False,
        forced_device: str | None = None, resume: str | None = None,
        deterministic: bool = False, force_fp32: bool = False) -> dict:
    set_seed(cfg["seed"], deterministic=deterministic)
    device = pick_device(require_cuda=require_cuda, forced=forced_device)
    rt.enforce_cuda_used(device, require_cuda)

    # Precision. fp32 is the default and mixed precision is opt-in through
    # `train.amp: true`, because a wrong guess here is not a small effect: on a
    # pre-Ampere card bf16 is emulated and slower than the fp32 it replaced, and on any
    # card it changes the arithmetic of a run whose numbers are being reported.
    #
    # When it is asked for, bf16 rather than fp16: same exponent range as fp32, so no
    # loss scaler and no chance of a silent inf across a multi-hour run. The trade is
    # mantissa bits, which a 5M-parameter model at these loss values does not need.
    # Autocast wraps the forward only -- the parameters stay fp32 and gradients
    # accumulate in fp32, which is what makes it safe without a scaler. `bf16_ok` also
    # checks compute capability, not just `is_bf16_supported()`, which answers True on
    # some cards where the dtype is emulated.
    amp_requested = bool(cfg.get("train", {}).get("amp", False)) and not force_fp32
    amp = amp_requested and rt.bf16_ok(device)
    if force_fp32:
        # The overriding arm of the bf16-vs-fp32 comparison. Recorded in the precision
        # string rather than only in the shell history, because the whole point of the
        # comparison is that the two runs are otherwise identical and the logs are the
        # only thing that distinguishes them afterwards.
        precision = "fp32 (forced by --fp32)"
    elif amp_requested and not amp:
        precision = f"fp32 (amp requested; bf16 unavailable on {device.type})"
    else:
        precision = "bf16 autocast" if amp else "fp32"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Smoke runs go to a scratch subdirectory. They are throwaway pipeline checks, and
    # writing them beside the reported artifacts means a glob for "the training summary"
    # can pick one up -- which it did, until this changed. `smoke/` is gitignored, so a
    # smoke run also cannot dirty the working tree before a commit.
    scratch = Path("smoke") if smoke else Path()
    out_dir = REPO_ROOT / cfg["paths"]["outputs"] / scratch
    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"] / scratch
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # `_fp32` goes in the tag as well as the resume filename: the precision arms share a
    # run_name, so without it the two runs' checkpoints, curves, generations and logs
    # differ only by a timestamp, and telling them apart later means opening each one.
    tag = (f"{cfg['run_name']}{'_smoke' if smoke else ''}"
           f"{'_fp32' if force_fp32 else ''}_{stamp}")
    log = RunLogger(REPO_ROOT / cfg["paths"]["raw_logs"] / f"task1_{tag}.log")

    dcfg, mcfg, tcfg = cfg["data"], cfg["model"], cfg["train"]
    if smoke:
        dcfg = {**dcfg, "n_train_windows": 2000, "n_val_windows": 500}
        tcfg = {**tcfg, "epochs": 2}

    log(f"precision={precision} tf32={device.type == 'cuda'} "
        f"note=evaluation always runs in fp32, so reported metrics are not affected "
        f"by the training precision")
    log(f"config={json.dumps({'data': dcfg, 'model': mcfg, 'train': tcfg})}")

    # ------------------------------------------------------------------ data
    bundle = data_mod.build(
        REPO_ROOT / dcfg["stories_jsonl"],
        block_size=dcfg["block_size"],
        n_train_windows=dcfg["n_train_windows"],
        n_val_windows=dcfg["n_val_windows"],
        val_story_frac=dcfg["val_story_frac"],
        seed=cfg["seed"],
    )
    tok = bundle.tokenizer
    log(
        f"stories train/val={bundle.n_train_stories}/{bundle.n_val_stories} "
        f"chars train/val={len(bundle.train_text)}/{len(bundle.val_text)} "
        f"vocab={tok.vocab_size} windows train/val={len(bundle.train_ds)}/{len(bundle.val_ds)}"
    )
    tok.save(REPO_ROOT / cfg["paths"]["checkpoints"] / "tokenizer.json")

    # drop_last=True is deliberate, and costs 32 of 100,000 windows per epoch
    # (100000 = 781*128 + 32). Kept rather than switched to False for two reasons: a
    # ragged final batch of 32 would contribute a 4x-noisier gradient at full learning
    # rate at the end of every epoch, and because shuffle=True reshuffles each epoch the
    # dropped 32 are a *different* 32 every time -- over 12 epochs the chance a given
    # window is never seen is (32/100000)^12, so nothing is permanently excluded. The
    # post-hoc training metrics use the same drop_last=True loader, so the reported
    # number describes exactly the windows the model trained on.
    # pin_memory is what makes the non_blocking copies in the training loop actually
    # asynchronous; without pinned host memory `non_blocking=True` is silently a no-op.
    # persistent_workers only pays for itself when there are workers to keep alive, and
    # the dataset here holds no per-epoch state, so re-forking would cost with no
    # correctness benefit either way.
    loader_kw = {"pin_memory": device.type == "cuda"}
    if tcfg["num_workers"] > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 4
    train_loader = DataLoader(
        bundle.train_ds, batch_size=tcfg["batch_size"], shuffle=True,
        drop_last=True, num_workers=tcfg["num_workers"], **loader_kw,
    )
    val_loader = DataLoader(
        bundle.val_ds, batch_size=tcfg["batch_size"], shuffle=False,
        num_workers=tcfg["num_workers"], **loader_kw,
    )

    # ----------------------------------------------------------------- model
    gcfg = GPTConfig(vocab_size=tok.vocab_size, block_size=dcfg["block_size"], **mcfg)
    model = GPT(gcfg).to(device)
    n_params = model.num_params()
    log(f"parameters={n_params:,}")

    optimizer, n_decay, n_nodecay = build_optimizer(
        model, tcfg["lr"], tcfg["weight_decay"], tcfg["betas"]
    )
    log(f"optimizer=AdamW decay_tensors={n_decay} nodecay_tensors={n_nodecay}")

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * tcfg["epochs"]
    warmup_steps = max(1, int(total_steps * tcfg["warmup_frac"]))
    log(f"steps_per_epoch={steps_per_epoch} total_steps={total_steps} warmup_steps={warmup_steps}")

    # ----------------------------------------------------- run provenance
    # Written to disk, not only logged, and with the device class as an explicit field.
    # This project now has results from two devices, and a metrics table that mixes them
    # without saying so is not a result.
    prov = rt.report(
        device, seed=cfg["seed"], batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"], precision=precision,
        dataset_sizes={"train_windows": len(bundle.train_ds),
                       "val_windows": len(bundle.val_ds),
                       "train_stories": bundle.n_train_stories,
                       "val_stories": bundle.n_val_stories,
                       "train_chars": len(bundle.train_text),
                       "val_chars": len(bundle.val_text),
                       "vocab_size": tok.vocab_size},
        n_params=n_params,
        extra={"task": "task1_llm", "run_name": cfg["run_name"], "tag": tag,
               "smoke": smoke, "require_cuda": require_cuda,
               "deterministic": deterministic,
               "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
               "epochs": tcfg["epochs"], "drop_last": True})
    for line in rt.format_report(prov):
        log(line)
    log(f"runtime_report={rt.write_report(prov, out_dir, tag).relative_to(REPO_ROOT)}")

    # -------------------------------------------------------------- training
    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
    step_losses, grad_norms, lr_trace = [], [], []
    peak_mem, step, t_start = 0.0, 0, time.time()
    # Two counters, because they answer different questions. `tokens_seen` is the
    # model's cumulative training exposure and must survive a resume. `tokens_this_run`
    # is the denominator for throughput, and mixing them reports a resumed run at
    # double its real tokens/second.
    tokens_seen = tokens_this_run = 0
    start_epoch = 1

    if resume:
        # Resume exists because the module docstring used to claim it did. The
        # scheduler here is a closed-form function of `step` rather than a torch
        # scheduler object, so restoring the step counter restores the learning-rate
        # schedule exactly -- there is no `last_epoch` to get out of sync.
        state = load_resume_state(resume, ckpt_dir, cfg["run_name"], log, smoke=smoke,
                                  force_fp32=force_fp32)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        step = state["global_step"]
        start_epoch = state["epoch"] + 1
        history = state["history"]
        step_losses, grad_norms = state["step_losses"], state["grad_norms"]
        tokens_seen = state.get("tokens_seen", 0)
        lr_trace = [lr_at(step, total_steps, tcfg["lr"], warmup_steps,
                          tcfg["min_lr_frac"])]
        if state["total_steps"] != total_steps:
            log(f"WARNING resume total_steps={state['total_steps']} but this config "
                f"says {total_steps}; the LR schedule will not match the original run")
        log(f"resumed from epoch {state['epoch']} step {step}; continuing at "
            f"epoch {start_epoch}")
        if start_epoch > tcfg["epochs"]:
            log(f"nothing left to train: the checkpoint is already at epoch "
                f"{state['epoch']} and this config asks for {tcfg['epochs']}. "
                f"Re-evaluating and re-writing artifacts from the loaded weights.")

    for epoch in range(start_epoch, tcfg["epochs"] + 1):
        model.train()
        epoch_loss, epoch_tokens, t_epoch = 0.0, 0, time.time()

        for x, y in train_loader:
            lr = lr_at(step, total_steps, tcfg["lr"], warmup_steps, tcfg["min_lr_frac"])
            for g in optimizer.param_groups:
                g["lr"] = lr

            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Measured before clipping, so the number reflects the raw gradient.
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"]).item()
            optimizer.step()

            n_tok = y.numel()
            epoch_loss += loss.item() * n_tok
            epoch_tokens += n_tok
            tokens_seen += n_tok
            tokens_this_run += n_tok
            step_losses.append(loss.item())
            grad_norms.append(gnorm)
            lr_trace.append(lr)

            if device.type == "mps":
                peak_mem = max(peak_mem, torch.mps.current_allocated_memory() / 1024**2)

            if step % tcfg["log_every"] == 0:
                log(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} "
                    f"grad_norm={gnorm:.3f} lr={lr:.2e}")
            step += 1

        train_ce = epoch_loss / epoch_tokens
        val = metrics_mod.evaluate_loss_and_accuracy(model, val_loader, device)
        epoch_secs = time.time() - t_epoch

        history["epoch"].append(epoch)
        history["train_loss"].append(train_ce)
        history["val_loss"].append(val["cross_entropy"])
        history["val_acc"].append(val["top1_accuracy"])
        history["lr"].append(lr_trace[-1])

        # Epoch 1 on unfamiliar hardware is the only cheap estimate of what the whole
        # run costs, and the decision it informs -- whether the slot has room for the
        # large arm too -- has to be made in the first few minutes.
        remaining = tcfg["epochs"] - epoch
        eta = f" eta_min={remaining * epoch_secs / 60:.1f}" if remaining else ""
        log(f"EPOCH {epoch} train_ce={train_ce:.4f} val_ce={val['cross_entropy']:.4f} "
            f"val_ppl={val['perplexity']:.3f} val_bpc={val['bits_per_character']:.4f} "
            f"val_top1={val['top1_accuracy']:.4f} secs={epoch_secs:.1f} "
            f"tokens_per_sec={epoch_tokens / epoch_secs:,.0f}{eta}")

        # One rolling resume checkpoint, overwritten each epoch rather than one file per
        # epoch: this run is under an hour even on a modest GPU, so the value of resume
        # is surviving a crash, not archiving a trajectory, and 12 copies of a 5M-
        # parameter model plus AdamW's two moment buffers is 700 MB of disk for nothing.
        torch.save(
            checkpoint_payload(
                model=model, optimizer=optimizer, gcfg=gcfg, cfg=cfg, tok=tok,
                history=history, tag=tag, epoch=epoch, global_step=step,
                total_steps=total_steps, tokens_seen=tokens_seen,
                step_losses=step_losses, grad_norms=grad_norms, provenance=prov),
            # `--fp32` gets its own resume file. The precision arms share a run_name, so
            # a single rolling file would mean the second arm destroys the first arm's
            # crash recovery at its own epoch 1.
            ckpt_dir / (f"{cfg['run_name']}{'_smoke' if smoke else ''}"
                        f"{'_fp32' if force_fp32 else ''}_resume.pt"))

    total_secs = time.time() - t_start
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    elif device.type == "cpu":
        peak_mem = metrics_mod.peak_memory_mb(device)

    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save(
        checkpoint_payload(
            model=model, optimizer=optimizer, gcfg=gcfg, cfg=cfg, tok=tok,
            history=history, tag=tag, epoch=tcfg["epochs"], global_step=step,
            total_steps=total_steps, tokens_seen=tokens_seen,
            step_losses=step_losses, grad_norms=grad_norms, provenance=prov),
        ckpt_path,
    )
    log(f"checkpoint={ckpt_path.relative_to(REPO_ROOT)}")

    # The whole training split by default. See the `eval` block in config.yaml: a
    # 100-batch cap measured 12.8% of it and was reported as if it were the training
    # loss.
    train_eval_batches = cfg.get("eval", {}).get("train_eval_batches")
    train_eval = metrics_mod.evaluate_loss_and_accuracy(
        model, train_loader, device, max_batches=train_eval_batches)
    log(f"train metrics computed over "
        f"{train_eval_batches if train_eval_batches else len(train_loader)} batches "
        f"of {len(train_loader)}")
    val_eval = metrics_mod.evaluate_loss_and_accuracy(model, val_loader, device)
    stability = metrics_mod.stability_report(step_losses, grad_norms)

    summary = {
        "tag": tag,
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
        "device": str(device),
        "device_name": device_name(device),
        "device_class": prov["device_class"],
        "precision": precision,
        "provenance": prov,
        "parameter_count": n_params,
        "train_cross_entropy": train_eval["cross_entropy"],
        "val_cross_entropy": val_eval["cross_entropy"],
        "train_perplexity": train_eval["perplexity"],
        "val_perplexity": val_eval["perplexity"],
        "train_bits_per_character": train_eval["bits_per_character"],
        "val_bits_per_character": val_eval["bits_per_character"],
        "generalization_gap": metrics_mod.generalization_gap(
            train_eval["cross_entropy"], val_eval["cross_entropy"]
        ),
        "train_top1_accuracy": train_eval["top1_accuracy"],
        "val_top1_accuracy": val_eval["top1_accuracy"],
        "total_training_seconds": total_secs,
        # Null rather than a number when this process trained nothing: dividing a
        # resumed token count by the few seconds spent re-evaluating produces a
        # throughput figure in the millions, which would be reported as if measured.
        "training_tokens_per_sec": (tokens_this_run / total_secs
                                    if tokens_this_run else None),
        "tokens_this_process": tokens_this_run,
        "tokens_seen_cumulative": tokens_seen,
        "epochs_trained_this_process": max(0, tcfg["epochs"] - start_epoch + 1),
        "resumed_from": resume,
        "peak_memory_mb": peak_mem,
        "vocab_size": tok.vocab_size,
        **stability,
        "history": history,
        "step_losses": step_losses,
        "grad_norms": grad_norms,
    }

    (out_dir / f"train_summary_{tag}.json").write_text(json.dumps(summary, indent=2), "utf-8")
    tps = summary["training_tokens_per_sec"]
    log(f"train_ce={summary['train_cross_entropy']:.4f} "
        f"val_ce={summary['val_cross_entropy']:.4f} "
        f"gap={summary['generalization_gap']:.4f} "
        f"tokens_per_sec={f'{tps:,.0f}' if tps else 'n/a (no epoch trained)'} "
        f"peak_mem_mb={peak_mem:.1f} total_secs={total_secs:.1f}")
    log.close()

    summary["_model"] = model
    summary["_tokenizer"] = tok
    summary["_bundle"] = bundle
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="task1_llm/shriram_dundigalla/src/config.yaml")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to check the pipeline end to end, on the same "
                         "device the full run would use")
    ap.add_argument("--resume", default=None,
                    help="path to a *_resume.pt, or 'auto' for the newest one")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN algorithms; slower, and does not make a GPU run "
                         "bit-exact on its own")
    ap.add_argument("--fp32", action="store_true",
                    help="ignore train.amp and train in fp32; the control arm when "
                         "checking whether bf16 is numerically safe on this GPU")
    rt.add_device_args(ap)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = REPO_ROOT / cfg_path
    try:
        run(yaml.safe_load(cfg_path.read_text("utf-8")), smoke=args.smoke,
            require_cuda=args.require_cuda, forced_device=args.device,
            resume=args.resume, deterministic=args.deterministic,
            force_fp32=args.fp32)
    except rt.CudaUnavailable as exc:
        rt.die(exc)
