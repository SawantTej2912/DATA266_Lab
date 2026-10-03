"""Task 2 driver: preprocess once, train all three models, score them identically.

    python task2_sentiment/shriram_dundigalla/src/train.py
    python task2_sentiment/shriram_dundigalla/src/train.py --smoke --require-cuda
    python task2_sentiment/shriram_dundigalla/src/train.py --model bilstm --require-cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, TensorDataset

REPO_ROOT = Path(__file__).resolve().parents[3]

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import metrics as M            # noqa: E402
import preprocess as P         # noqa: E402
import runtime as rt           # noqa: E402
from models import architecture_summary, build_model  # noqa: E402

# Shared with Parts 1 and 3 so `--require-cuda` cannot mean three different things.
pick_device = rt.select_device


def hardware_string(device) -> str:
    """The exact CPU/GPU used, which section 2.2.5 asks to be recorded per model."""
    import subprocess

    cpu = platform.processor() or "unknown"
    if sys.platform == "darwin":
        try:
            cpu = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            pass
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(0)} (CUDA) / host CPU {cpu}"
    if device.type == "mps":
        return f"{cpu} integrated GPU via Apple MPS"
    return f"{cpu} (CPU only)"


set_seed = rt.set_seed

# The label convention, in one place. `preprocess.py` maps Yelp Polarity's 1/2 to 0/1
# and every metric downstream treats 1 as the positive class -- ROC-AUC, PR-AUC, the
# positive-class ECE and the reliability diagram all assume it. Recording it in the
# checkpoint means a loaded model cannot be scored with the polarity inverted, which
# produces a plausible-looking accuracy near 1 - a and is easy to miss.
LABEL_MAP = {0: "negative", 1: "positive"}


class RunLogger:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = path.open("a", encoding="utf-8")

    def __call__(self, msg):
        line = f"[{datetime.now(timezone.utc).isoformat(timespec='seconds')}] {msg}"
        print(line, flush=True)
        self.fh.write(line + "\n")
        self.fh.flush()

    def close(self):
        self.fh.close()


def config_fingerprint(cfg: dict) -> str:
    """A short stable hash of everything that changes the preprocessed tensors.

    Canonical JSON with sorted keys, so the hash depends on the values and not on the
    order PyYAML happened to build the dict in.
    """
    key = {"data": cfg["data"], "preprocess": cfg["preprocess"], "seed": cfg["seed"]}
    return hashlib.sha256(
        json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def data_manifest(data: P.Task2Data, cfg: dict) -> dict:
    """What the models were actually trained on, in a form that can be diffed.

    Split sizes and class counts are computed from the tensors rather than copied from
    the config, so a manifest that disagrees with the config is evidence the cache is
    stale rather than something that silently agrees with itself.
    """
    def split_of(s):
        counts = np.bincount(s.labels.astype(int), minlength=2)
        return {"n": int(len(s.labels)),
                "class_counts": {LABEL_MAP[i]: int(c) for i, c in enumerate(counts)},
                "positive_fraction": round(float(counts[1] / max(1, counts.sum())), 6),
                "mean_length": round(float(s.lengths.mean()), 2),
                "max_length": int(s.lengths.max())}

    return {
        "dataset": "Yelp Polarity (yelp_review_polarity_csv), parquet mirror",
        "source_files": {
            "train": cfg["data"]["train_parquet"],
            "test": cfg["data"]["test_parquet"]},
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "splits": {"train": split_of(data.train), "val": split_of(data.val),
                   "test": split_of(data.test)},
        "vocabulary": {"size": len(data.vocab),
                       "max_size_configured": cfg["data"]["vocab_size"],
                       "min_freq": cfg["data"]["min_freq"],
                       "pad_index": 0, "unk_index": 1,
                       "pad_token": data.vocab.itos[0],
                       "unk_token": data.vocab.itos[1]},
        "preprocess_config": dict(cfg["preprocess"]),
        "config_fingerprint": config_fingerprint(cfg),
        "seed": cfg["seed"],
        "preprocess_stats": data.stats,
    }


def get_data(cfg, log, force: bool = False) -> P.Task2Data:
    """Preprocess once and cache. Cleaning 148K reviews takes a few minutes; the three
    model builds must see byte-identical inputs for the paired McNemar test to be valid.

    The cache is only ever trusted after its fingerprint is checked against the current
    config *and* the tensors inside it are checked against that config's split sizes.
    The fingerprint alone is not enough: it is computed from the config, so a pickle
    whose contents were truncated or written by an older preprocessing code path still
    carries a matching key. Evaluating against a silently wrong cache would not raise
    anything -- it would just report metrics for a different dataset.
    """
    cache = REPO_ROOT / cfg["data"]["cache_dir"] / "task2_data.pkl"
    fingerprint = config_fingerprint(cfg)

    if cache.exists() and not force:
        with cache.open("rb") as fh:
            blob = pickle.load(fh)
        if blob.get("fingerprint") == fingerprint:
            data = blob["data"]
            want_val = cfg["data"]["n_val"]
            problems = []
            if len(data.val.labels) != want_val:
                problems.append(f"val split has {len(data.val.labels)} rows, config "
                                f"says n_val={want_val}")
            if data.train.ids.shape[1] != cfg["data"]["max_len"]:
                problems.append(f"sequences are {data.train.ids.shape[1]} tokens wide, "
                                f"config says max_len={cfg['data']['max_len']}")
            if len(data.vocab) > cfg["data"]["vocab_size"] + 2:
                problems.append(f"vocabulary has {len(data.vocab)} entries, config caps "
                                f"it at {cfg['data']['vocab_size']} + pad + unk")
            if problems:
                log("cache fingerprint matches but its contents do not: "
                    + "; ".join(problems))
                log("re-running preprocessing rather than trusting it")
            else:
                log(f"loaded cached preprocessing from "
                    f"{cache.relative_to(REPO_ROOT)} "
                    f"(fingerprint {fingerprint}, verified against config)")
                return data
        else:
            log(f"cache fingerprint {blob.get('fingerprint')} != config fingerprint "
                f"{fingerprint}, re-running preprocessing")

    t0 = time.time()
    data = P.build(cfg, REPO_ROOT, log=log)
    log(f"preprocessing took {time.time() - t0:.1f}s")
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as fh:
        pickle.dump({"fingerprint": fingerprint, "data": data}, fh)
    return data


def loaders(split, batch_size, shuffle, workers):
    ds = TensorDataset(
        torch.from_numpy(split.ids.astype(np.int64)),
        torch.from_numpy(split.lengths.astype(np.int64)),
        torch.from_numpy(split.labels.astype(np.float32)),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs = []
    for ids, lengths, _ in loader:
        logits = model(ids.to(device), lengths.to(device))
        probs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(probs)


def train_one(name, spec, data, cfg, device, log, provenance=None):
    tcfg = cfg["train"]
    set_seed(cfg["seed"])

    model = build_model(spec, len(data.vocab)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"--- {name}: {architecture_summary(spec, model)}")

    opt = torch.optim.AdamW(model.parameters(), lr=spec["lr"], weight_decay=tcfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    tr = loaders(data.train, spec["batch_size"], True, tcfg["num_workers"])
    va = loaders(data.val, spec["batch_size"], False, tcfg["num_workers"])
    te = loaders(data.test, spec["batch_size"], False, tcfg["num_workers"])

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": []}
    best_f1, best_state, stale = -1.0, None, 0
    peak_mem, examples_seen = 0.0, 0
    t_start = time.time()

    for epoch in range(1, spec["epochs"] + 1):
        model.train()
        running, n = 0.0, 0
        for step, (ids, lengths, y) in enumerate(tr):
            ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
            logits = model(ids, lengths)
            loss = loss_fn(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            opt.step()

            running += loss.item() * y.size(0)
            n += y.size(0)
            examples_seen += y.size(0)
            if device.type == "mps":
                peak_mem = max(peak_mem, torch.mps.current_allocated_memory() / 1024**2)
            if step % tcfg["log_every"] == 0:
                log(f"{name} epoch={epoch} step={step}/{len(tr)} loss={loss.item():.4f}")

        train_loss = running / n

        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for ids, lengths, y in va:
                ids, lengths, y = ids.to(device), lengths.to(device), y.to(device)
                vloss += loss_fn(model(ids, lengths), y).item() * y.size(0)
                vn += y.size(0)
        val_probs = predict(model, va, device)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(data.val.labels, (val_probs >= 0.5).astype(int), average="macro")

        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(vloss / vn)
        history["val_macro_f1"].append(float(val_f1))
        log(f"{name} EPOCH {epoch} train_loss={train_loss:.4f} val_loss={vloss/vn:.4f} "
            f"val_macro_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, stale = val_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= tcfg["early_stop_patience"]:
                log(f"{name} early stop at epoch {epoch} (best val_macro_f1={best_f1:.4f})")
                break

    train_secs = time.time() - t_start
    if best_state is not None:
        model.load_state_dict(best_state)

    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    elif device.type == "cpu":
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_mem = rss / 1024**2 if sys.platform == "darwin" else rss / 1024

    ckpt_dir = REPO_ROOT / cfg["paths"]["checkpoints"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"{name}.pt"
    # The vocabulary and the preprocessing settings go *inside* the checkpoint, not
    # just alongside it. A state dict plus a vocab_size is not enough to score a single
    # review: token ids are meaningless without the itos list, and the same text
    # tokenises differently under a different stemmer or stopword policy. Keeping them
    # in separate files means a checkpoint can be paired with the wrong vocabulary and
    # produce confident nonsense rather than an error.
    torch.save({"model_state_dict": model.state_dict(), "spec": spec,
                "vocab_size": len(data.vocab), "history": history,
                "vocab_itos": data.vocab.itos,
                "preprocess": dict(cfg["preprocess"]),
                "data_config": {k: cfg["data"][k] for k in
                                ("max_len", "vocab_size", "min_freq",
                                 "n_train", "n_val", "use_full_test")},
                "label_map": LABEL_MAP,
                "config_fingerprint": config_fingerprint(cfg),
                "provenance": provenance,
                "seed": cfg["seed"]}, ckpt_path)

    test_probs = predict(model, te, device)
    return {
        "name": name,
        "spec": spec,
        "architecture": architecture_summary(spec, model),
        "parameter_count": n_params,
        "training_seconds": train_secs,
        "examples_per_sec": examples_seen / train_secs,
        "peak_memory_mb": peak_mem,
        "hardware": hardware_string(device),
        "best_val_macro_f1": float(best_f1),
        "epochs_run": history["epoch"][-1],
        "history": history,
        "test_probs": test_probs,
        "checkpoint": str(ckpt_path.relative_to(REPO_ROOT)),
    }


def main(smoke: bool = False, force_preprocess: bool = False,
         config: str | None = None, seed: int | None = None,
         tag_suffix: str = "", require_cuda: bool = False,
         forced_device: str | None = None, models: list[str] | None = None,
         deterministic: bool = False):
    # --config and --seed exist so a seed sweep can drive this without editing the
    # committed config, and to match how Parts 1 and 3 are invoked. Defaults are
    # unchanged, so `python train.py` still reproduces the reported run exactly.
    cfg_path = (Path(config) if config else
                REPO_ROOT / "task2_sentiment/shriram_dundigalla/src/config.yaml")
    cfg = yaml.safe_load(cfg_path.read_text("utf-8"))
    if seed is not None:
        cfg["seed"] = seed

    if smoke:
        # A smoke run must not be able to touch a reported artifact. It previously
        # overwrote the committed metrics_report.csv and the cached vocabulary, because
        # the report path was hardcoded and the cache path was shared -- so a two-minute
        # pipeline check destroyed the published results. Everything a smoke run writes
        # is redirected under a `smoke/` subdirectory, all of which is gitignored.
        cfg["paths"] = {**cfg["paths"],
                        "checkpoints": str(Path(cfg["paths"]["checkpoints"]) / "smoke"),
                        "outputs": str(Path(cfg["paths"]["outputs"]) / "smoke")}
        cfg["paths"]["report_dir"] = cfg["paths"]["outputs"]
        cfg["data"] = {**cfg["data"],
                       "cache_dir": str(Path(cfg["data"]["cache_dir"]) / "smoke")}
        cfg["data"] = {**cfg["data"], "n_train": 4000, "n_val": 1000, "use_full_test": False}
        cfg["eval"] = {**cfg["eval"], "bootstrap_n": 200}
        for name in cfg["models"]:
            cfg["models"][name] = {**cfg["models"][name], "epochs": 1}

    # --model trains a subset. The three-way comparison and the paired McNemar tests
    # need all three present, so a subset run is explicitly labelled as a partial run:
    # it still writes checkpoints and per-model metrics, but the comparison artifacts it
    # would produce would be missing arms, and a metrics_report.csv with one row must
    # not silently replace the three-row one the report quotes.
    partial = False
    if models:
        unknown = [m for m in models if m not in cfg["models"]]
        if unknown:
            raise SystemExit(
                f"unknown model(s) {unknown}; config defines "
                f"{sorted(cfg['models'])}")
        if set(models) != set(cfg["models"]):
            partial = True
            cfg["models"] = {k: v for k, v in cfg["models"].items() if k in models}
            cfg["paths"] = {**cfg["paths"],
                            "outputs": str(Path(cfg["paths"]["outputs"]) / "partial")}
            cfg["paths"]["report_dir"] = cfg["paths"]["outputs"]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    tag = (f"task2{'_smoke' if smoke else ''}"
           f"{'_partial_' + '-'.join(sorted(models)) if partial else ''}"
           f"{tag_suffix}_{stamp}")
    log = RunLogger(REPO_ROOT / cfg["paths"]["raw_logs"] / f"{tag}.log")

    device = pick_device(require_cuda=require_cuda, forced=forced_device)
    rt.enforce_cuda_used(device, require_cuda)
    set_seed(cfg["seed"], deterministic=deterministic)
    log(f"device={device} hardware={hardware_string(device)} torch={torch.__version__}")
    if partial:
        log(f"PARTIAL RUN: only {sorted(models)}. Artifacts go to "
            f"{cfg['paths']['outputs']} so the full three-model report is not "
            f"overwritten. McNemar comparisons need all three arms.")

    data = get_data(cfg, log, force=force_preprocess or smoke)
    out_dir = REPO_ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Part 2 has no single model or batch size, so the provenance block records the
    # per-model values rather than pretending there is one of each.
    prov = rt.report(
        device, seed=cfg["seed"],
        batch_size={n: s["batch_size"] for n, s in cfg["models"].items()},
        num_workers=cfg["train"]["num_workers"],
        precision="fp32",
        dataset_sizes={"train": int(len(data.train.labels)),
                       "val": int(len(data.val.labels)),
                       "test": int(len(data.test.labels)),
                       "vocab_size": len(data.vocab)},
        extra={"task": "task2_sentiment", "tag": tag, "smoke": smoke,
               "partial": partial, "models": sorted(cfg["models"]),
               "require_cuda": require_cuda, "deterministic": deterministic,
               "config_fingerprint": config_fingerprint(cfg),
               "label_map": {str(k): v for k, v in LABEL_MAP.items()}})
    for line in rt.format_report(prov):
        log(line)
    rt.write_report(prov, out_dir, tag)

    # The manifest is written before training, so it exists even if a run is killed.
    manifest = data_manifest(data, cfg)
    (out_dir / "data_manifest.json").write_text(
        json.dumps(manifest, indent=2), "utf-8")
    log(f"data_manifest fingerprint={manifest['config_fingerprint']} "
        f"train={manifest['splits']['train']['n']} "
        f"(pos {manifest['splits']['train']['positive_fraction']:.4f}) "
        f"val={manifest['splits']['val']['n']} "
        f"test={manifest['splits']['test']['n']} "
        f"(pos {manifest['splits']['test']['positive_fraction']:.4f}) "
        f"vocab={manifest['vocabulary']['size']}")

    results = [train_one(n, s, data, cfg, device, log, provenance=prov)
               for n, s in cfg["models"].items()]

    # ------------------------------------------------------------------ scoring
    ecfg = cfg["eval"]
    rows, slice_rows = [], []
    for r in results:
        rep = M.full_report(
            data.test.labels, r["test_probs"], threshold=ecfg["threshold"],
            n_boot=ecfg["bootstrap_n"], n_bins=ecfg["calibration_bins"],
            seed=cfg["seed"], slices_df=data.test.slices,
        )
        r["y_pred"] = rep.pop("_y_pred")
        for s in rep.pop("_slices"):
            slice_rows.append({"model": r["name"], **s})
        r["report"] = rep
        rows.append({
            "member": "shriram_dundigalla", "task": "task2_sentiment",
            "model": r["name"], "role": "baseline" if r["name"].startswith("baseline") else "experimental",
            "architecture": r["architecture"],
            "hyperparameters": json.dumps(r["spec"]),
            "checkpoint": r["checkpoint"],
            **rep,
            "parameter_count": r["parameter_count"],
            "training_seconds": r["training_seconds"],
            "examples_per_sec": r["examples_per_sec"],
            "peak_memory_mb": r["peak_memory_mb"],
            "epochs_run": r["epochs_run"],
            "best_val_macro_f1": r["best_val_macro_f1"],
            "hardware": r["hardware"],
            # So a table can never mix devices without saying so.
            "device_class": prov["device_class"],
            "precision": prov["precision"],
        })
        log(f"{r['name']}: acc={rep['accuracy']:.4f} macroF1={rep['f1_macro']:.4f} "
            f"ROC-AUC={rep['roc_auc']:.4f} PR-AUC={rep['pr_auc']:.4f} MCC={rep['mcc']:.4f} "
            f"Brier={rep['brier_score']:.4f} "
            f"ECEconf={rep['ece_predicted_class_confidence']:.4f} "
            f"ECEpos={rep['ece_positive_class']:.4f}")

    # ------------------------------------------------- paired McNemar vs baseline
    # McNemar compares two sets of predictions on the same examples, so it needs the
    # baseline arm to exist in *this* run -- predictions from a different run are not
    # paired even on the same test set, because the arms would have seen different
    # preprocessing. A partial run that excludes the baseline therefore skips the test
    # rather than comparing against something it should not.
    mc_rows = []
    baseline = next((r for r in results if r["name"].startswith("baseline")), None)
    if baseline is None:
        log("skipping McNemar: no baseline arm in this run, and pairing against a "
            "different run's predictions would not be a paired test")
    else:
        for r in results:
            if r is baseline:
                continue
            mc = M.mcnemar_test(data.test.labels, baseline["y_pred"], r["y_pred"])
            mc_rows.append({"model_a": baseline["name"], "model_b": r["name"], **mc})
            log(f"McNemar {baseline['name']} vs {r['name']}: "
                f"only_a={mc['only_a_correct']} only_b={mc['only_b_correct']} "
                f"{mc['test']} p={mc['p_value']:.3e}")

    # ------------------------------------------------------------------ artefacts
    # Config-driven so a smoke run or a seed-sweep arm writes its report beside its own
    # outputs instead of over the committed one. Defaults to the member directory, which
    # is where the reported metrics_report.csv lives.
    member_dir = REPO_ROOT / cfg["paths"].get(
        "report_dir", "task2_sentiment/shriram_dundigalla")
    member_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(member_dir / "metrics_report.csv", index=False)
    pd.DataFrame(slice_rows).to_csv(out_dir / "slice_metrics.csv", index=False)
    pd.DataFrame(mc_rows).to_csv(out_dir / "mcnemar_tests.csv", index=False)

    np.savez_compressed(
        out_dir / "test_predictions.npz",
        y_true=data.test.labels,
        **{f"probs_{r['name']}": r["test_probs"] for r in results},
        **{f"pred_{r['name']}": r["y_pred"] for r in results},
    )
    (out_dir / "histories.json").write_text(
        json.dumps({r["name"]: r["history"] for r in results}, indent=2), "utf-8"
    )
    (out_dir / "preprocess_stats.json").write_text(json.dumps(data.stats, indent=2), "utf-8")

    log(f"wrote {(member_dir / 'metrics_report.csv').relative_to(REPO_ROOT)}")
    log.close()
    return results, data


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force-preprocess", action="store_true")
    ap.add_argument("--config", default=None,
                    help="defaults to src/config.yaml; used by the seed sweep")
    ap.add_argument("--seed", type=int, default=None,
                    help="override the config's seed; used by the seed sweep")
    ap.add_argument("--tag-suffix", default="",
                    help="appended to the run tag, so parallel runs do not collide")
    ap.add_argument("--model", action="append", dest="models", default=None,
                    choices=["baseline_bow", "exp1_bilstm", "exp2_textcnn",
                             "baseline", "bilstm", "textcnn"],
                    help="train only these models; repeatable. A subset run writes to "
                         "outputs/partial/ and skips McNemar, which needs all arms")
    ap.add_argument("--deterministic", action="store_true",
                    help="pin cuDNN algorithms; slower")
    rt.add_device_args(ap)
    args = ap.parse_args()

    # Short aliases, because `--model bilstm` is what anyone types and what the run
    # plan documents, while the config keys carry the experiment numbering.
    alias = {"baseline": "baseline_bow", "bilstm": "exp1_bilstm",
             "textcnn": "exp2_textcnn"}
    chosen = [alias.get(m, m) for m in args.models] if args.models else None

    try:
        main(smoke=args.smoke, force_preprocess=args.force_preprocess,
             config=args.config, seed=args.seed, tag_suffix=args.tag_suffix,
             require_cuda=args.require_cuda, forced_device=args.device,
             models=chosen, deterministic=args.deterministic)
    except rt.CudaUnavailable as exc:
        rt.die(exc)
