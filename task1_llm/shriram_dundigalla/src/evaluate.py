"""Task 1 scoring: load a checkpoint, generate, and write the full metrics report.

Kept separate from training so the metrics can be recomputed from a checkpoint without
retraining, and so every team member scores their model with the same code.

    python task1_llm/shriram_dundigalla/src/evaluate.py --checkpoint <path>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import hashlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data as data_mod           # noqa: E402
import metrics as metrics_mod     # noqa: E402
from model import GPT, GPTConfig  # noqa: E402
from train import pick_device, device_name, set_seed  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
_WORD_RE = re.compile(r"[a-z']+")


def load_checkpoint(path: Path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = GPT(GPTConfig(**ckpt["model_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def sample(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float,
           top_k, device, top_p=None, repetition_penalty: float = 1.0,
           seed: int | None = None) -> tuple[str, float]:
    """Return the generated continuation and the generation rate in tokens/sec."""
    if seed is not None:
        torch.manual_seed(seed)
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    if device.type == "mps":
        torch.mps.synchronize()
    t0 = time.time()
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k,
                         top_p=top_p, repetition_penalty=repetition_penalty)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.time() - t0
    return tokenizer.decode(out[0].tolist()), max_new_tokens / elapsed


def _seed_for(*parts) -> int:
    """A reproducible seed from arbitrary values.

    Not `hash()`: Python randomises string hashing per process unless PYTHONHASHSEED is
    set, so a sweep seeded that way would produce different draws on every run and the
    reported standard deviations would not be reproducible -- which is the one thing
    they need to be.
    """
    key = "|".join(map(str, parts)).encode()
    return int(hashlib.sha256(key).hexdigest()[:8], 16)


# Metrics worth aggregating across draws. The rest of generation_report is either a
# count or is implied by these.
_AGG_KEYS = ("distinct_1_word", "distinct_2_word", "distinct_3_word",
             "repeated_4gram_rate_word", "train_vocab_word_rate")


def _sweep(model, tok, train_vocab, gcfg, device, log=print) -> list[dict]:
    """Sample every (prompt, temperature) pair `n_samples` times and aggregate.

    A single sample is one draw from a stochastic decoder. On this model the
    repeated-4-gram rate ranges from 0.00 to 0.06 across temperatures in a single draw
    each, which is indistinguishable from noise -- so a mean and a standard deviation
    over several seeds and several prompts is the only honest way to report it.
    """
    rows = []
    for temp in gcfg["temperatures"]:
        per_metric = {k: [] for k in _AGG_KEYS}
        rates = []
        for prompt in gcfg["prompts"]:
            for s in range(gcfg["n_samples"]):
                # Seed derived from the setting, so the sweep is reproducible and each
                # draw is a genuinely different one.
                text, tps = sample(model, tok, prompt, gcfg["max_new_tokens"], temp,
                                   gcfg["top_k"], device,
                                   seed=_seed_for(prompt, temp, s))
                cont = text[len(prompt):]
                rep = {**metrics_mod.generation_report(cont),
                       "train_vocab_word_rate": metrics_mod.train_vocab_word_rate(cont, train_vocab)}
                for k in _AGG_KEYS:
                    per_metric[k].append(rep[k])
                rates.append(tps)
        n = len(rates)
        row = {"temperature": temp, "n_draws": n,
               "n_prompts": len(gcfg["prompts"]), "n_seeds": gcfg["n_samples"],
               "generation_tokens_per_sec_mean": float(np.mean(rates))}
        for k in _AGG_KEYS:
            row[f"{k}_mean"] = float(np.mean(per_metric[k]))
            row[f"{k}_std"] = float(np.std(per_metric[k], ddof=1)) if n > 1 else 0.0
        rows.append(row)
        log(f"  T={temp:<4} n={n:<3} "
            + "  ".join(f"{k.replace('_word', '')}={row[f'{k}_mean']:.3f}"
                        f"+/-{row[f'{k}_std']:.3f}" for k in _AGG_KEYS))
    return rows


def _decoding_comparison(model, tok, train_vocab, gcfg, device, log=print) -> list[dict]:
    """Compare decoding strategies at one temperature, aggregated the same way."""
    temp = gcfg["decoding_temperature"]
    rows = []
    for var in gcfg["decoding_variants"]:
        per_metric = {k: [] for k in _AGG_KEYS}
        for prompt in gcfg["prompts"]:
            for s in range(gcfg["n_samples"]):
                text, _ = sample(model, tok, prompt, gcfg["max_new_tokens"], temp,
                                 var["top_k"], device, top_p=var["top_p"],
                                 repetition_penalty=var["repetition_penalty"],
                                 seed=_seed_for(prompt, var["name"], s))
                cont = text[len(prompt):]
                rep = {**metrics_mod.generation_report(cont),
                       "train_vocab_word_rate": metrics_mod.train_vocab_word_rate(cont, train_vocab)}
                for k in _AGG_KEYS:
                    per_metric[k].append(rep[k])
        n = len(per_metric[_AGG_KEYS[0]])
        row = {"decoding": var["name"], "temperature": temp, "n_draws": n,
               "top_k": var["top_k"], "top_p": var["top_p"],
               "repetition_penalty": var["repetition_penalty"]}
        for k in _AGG_KEYS:
            row[f"{k}_mean"] = float(np.mean(per_metric[k]))
            row[f"{k}_std"] = float(np.std(per_metric[k], ddof=1)) if n > 1 else 0.0
        rows.append(row)
        log(f"  {var['name']:<20} "
            + "  ".join(f"{k.replace('_word', '')}={row[f'{k}_mean']:.3f}"
                        f"+/-{row[f'{k}_std']:.3f}" for k in _AGG_KEYS))
    return rows


def plot_curves(history, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    ax.plot(history["epoch"], history["train_loss"], marker="o", label="train")
    ax.plot(history["epoch"], history["val_loss"], marker="s", label="validation")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy (nats/char)")
    ax.set_title("Task 1 — training and validation loss"); ax.legend(); ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["val_acc"], marker="^", color="tab:green")
    ax.set_xlabel("epoch"); ax.set_ylabel("top-1 next-character accuracy")
    ax.set_title("Validation top-1 accuracy"); ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_stability(summary, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(summary["step_losses"], lw=0.6)
    axes[0].set_xlabel("step"); axes[0].set_ylabel("batch loss")
    axes[0].set_title("Per-step training loss"); axes[0].grid(alpha=0.3)

    axes[1].plot(summary["grad_norms"], lw=0.6, color="tab:red")
    axes[1].axhline(summary["run_config"]["train"]["grad_clip"], ls="--", color="k",
                    label=f"clip = {summary['run_config']['train']['grad_clip']}")
    axes[1].set_xlabel("step"); axes[1].set_ylabel("gradient norm (pre-clip)")
    axes[1].set_title("Gradient norm"); axes[1].legend(); axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(checkpoint: Path) -> dict:
    device = pick_device()
    model, ckpt = load_checkpoint(checkpoint, device)
    cfg = ckpt["run_config"]

    # Data, model and training settings come from the checkpoint, so evaluation always
    # rebuilds the split the model was actually trained on. Generation settings are an
    # evaluation choice rather than a property of the run, so those come from the config
    # on disk -- otherwise a decoding strategy added after training could never be
    # applied to an existing checkpoint without retraining it.
    disk_cfg_path = Path(__file__).resolve().parent / "config.yaml"
    if disk_cfg_path.exists():
        import yaml
        disk_cfg = yaml.safe_load(disk_cfg_path.read_text("utf-8"))
        cfg["generate"] = {**cfg.get("generate", {}), **disk_cfg.get("generate", {})}
    set_seed(cfg["seed"])

    out_dir = REPO_ROOT / cfg["paths"]["outputs"]
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = ckpt["tag"]

    summary_path = out_dir / f"train_summary_{tag}.json"
    summary = json.loads(summary_path.read_text("utf-8"))
    summary["run_config"] = cfg

    # Rebuild the same split to recover the tokenizer and the training vocabulary.
    dcfg = cfg["data"]
    bundle = data_mod.build(
        REPO_ROOT / dcfg["stories_jsonl"], dcfg["block_size"],
        dcfg["n_train_windows"], dcfg["n_val_windows"], dcfg["val_story_frac"], cfg["seed"],
    )
    tok = bundle.tokenizer
    train_vocab = set(_WORD_RE.findall(bundle.train_text.lower()))

    # ------------------------------------------------------------- generation
    gcfg = cfg["generate"]
    samples = []
    for temp in gcfg["temperatures"]:
        text, tps = sample(model, tok, gcfg["prompt"], gcfg["max_new_tokens"],
                           temp, gcfg["top_k"], device)
        continuation = text[len(gcfg["prompt"]):]
        row = {
            "temperature": temp,
            "decoding": "greedy" if temp <= 0 else "temperature sampling",
            "generation_tokens_per_sec": tps,
            "train_vocab_word_rate": metrics_mod.train_vocab_word_rate(continuation, train_vocab),
            **metrics_mod.generation_report(continuation),
        }
        samples.append({**row, "text": text})
        print(f"T={temp}: {tps:,.0f} tok/s  distinct-2(word)={row['distinct_2_word']:.3f}  "
              f"rep-4gram={row['repeated_4gram_rate_word']:.3f}  invocab-word={row['train_vocab_word_rate']:.3f}")

    (out_dir / f"generations_{tag}.json").write_text(json.dumps(samples, indent=2), "utf-8")
    with (out_dir / f"generations_{tag}.txt").open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(f"{'=' * 78}\nTEMPERATURE {s['temperature']} ({s['decoding']})\n{'=' * 78}\n")
            fh.write(s["text"] + "\n\n")

    pd.DataFrame([{k: v for k, v in s.items() if k != "text"} for s in samples]).to_csv(
        out_dir / f"generation_metrics_{tag}.csv", index=False
    )

    # ------------------------------------- multi-prompt, multi-seed aggregation
    # The single-draw table above is what the qualitative samples come from. These two
    # are what any *claim* about generation quality should rest on.
    print("\nmean +/- std across prompts x seeds:")
    sweep = _sweep(model, tok, train_vocab, gcfg, device)
    pd.DataFrame(sweep).to_csv(out_dir / f"generation_sweep_{tag}.csv", index=False)

    print("\ndecoding strategies at T="
          f"{gcfg['decoding_temperature']}:")
    decoding = _decoding_comparison(model, tok, train_vocab, gcfg, device)
    pd.DataFrame(decoding).to_csv(out_dir / f"decoding_comparison_{tag}.csv", index=False)

    # ------------------------------------------------------------------ plots
    plot_curves(summary["history"], out_dir / f"loss_curves_{tag}.png")
    plot_stability(summary, out_dir / f"stability_{tag}.png")

    # --------------------------------------------------- single metrics report
    # The report row uses temperature 0.8 as the headline sampling setting; the full
    # sweep lives in generation_metrics_*.csv.
    headline = min(samples, key=lambda s: abs(s["temperature"] - 0.8))
    sweep_at_headline = min(sweep, key=lambda r: abs(r["temperature"] - 0.8))
    # Lowest degenerate-repetition rate, tie-broken toward higher trigram diversity.
    best_decoding = min(decoding, key=lambda r: (r["repeated_4gram_rate_word_mean"],
                                                 -r["distinct_3_word_mean"]))
    report = {
        "member": "shriram_dundigalla",
        "task": "task1_llm",
        "model": cfg["run_name"],
        "checkpoint": summary["checkpoint"],
        "architecture": (
            f"decoder-only GPT, {cfg['model']['n_layer']} pre-LN blocks, "
            f"{cfg['model']['n_head']} heads, d_model={cfg['model']['d_model']}, "
            f"d_ff={cfg['model']['d_ff_mult'] * cfg['model']['d_model']}, "
            f"block_size={dcfg['block_size']}, learned token + positional embeddings"
        ),
        "hyperparameters": (
            f"AdamW lr={cfg['train']['lr']} betas={tuple(cfg['train']['betas'])} "
            f"wd={cfg['train']['weight_decay']} dropout={cfg['model']['dropout']} "
            f"batch={cfg['train']['batch_size']} epochs={cfg['train']['epochs']} "
            f"warmup={cfg['train']['warmup_frac']} cosine to {cfg['train']['min_lr_frac']}x, "
            f"grad_clip={cfg['train']['grad_clip']}"
        ),
        "vocab_size": summary["vocab_size"],
        "parameter_count": summary["parameter_count"],
        "train_cross_entropy": summary["train_cross_entropy"],
        "val_cross_entropy": summary["val_cross_entropy"],
        "train_perplexity": summary["train_perplexity"],
        "val_perplexity": summary["val_perplexity"],
        "train_bits_per_character": summary["train_bits_per_character"],
        "val_bits_per_character": summary["val_bits_per_character"],
        "generalization_gap": summary["generalization_gap"],
        "train_top1_accuracy": summary["train_top1_accuracy"],
        "val_top1_accuracy": summary["val_top1_accuracy"],
        # How much of the training split the train_* figures above were measured on.
        # Stating it is the point: an earlier version of this row was computed from 100
        # batches (12.8% of the split) and reported as "the training loss".
        "train_eval_windows": summary.get("train_eval_windows"),
        "distinct_1": headline["distinct_1_word"],
        "distinct_2": headline["distinct_2_word"],
        "distinct_3": headline["distinct_3_word"],
        "repeated_4gram_rate": headline["repeated_4gram_rate_word"],
        "train_vocab_word_rate": headline["train_vocab_word_rate"],
        "generation_temperature": headline["temperature"],
        # The single-draw figures above are one sample and should not be quoted on
        # their own -- at T=0.0 one draw gave a repeated-4-gram rate of 0.060 while the
        # mean over 15 draws is 0.235. These are the aggregates any claim should rest on.
        "distinct_2_mean": sweep_at_headline["distinct_2_word_mean"],
        "distinct_2_std": sweep_at_headline["distinct_2_word_std"],
        "repeated_4gram_rate_mean": sweep_at_headline["repeated_4gram_rate_word_mean"],
        "repeated_4gram_rate_std": sweep_at_headline["repeated_4gram_rate_word_std"],
        "train_vocab_word_rate_mean": sweep_at_headline["train_vocab_word_rate_mean"],
        "train_vocab_word_rate_std": sweep_at_headline["train_vocab_word_rate_std"],
        "generation_draws": sweep_at_headline["n_draws"],
        "best_decoding": best_decoding["decoding"],
        "best_decoding_repeated_4gram_rate_mean":
            best_decoding["repeated_4gram_rate_word_mean"],
        "nan_or_inf_count": summary["nan_or_inf_count"],
        "loss_spike_count": summary["loss_spike_count"],
        "grad_norm_mean": summary["grad_norm_mean"],
        "grad_norm_max": summary["grad_norm_max"],
        "training_tokens_per_sec": summary["training_tokens_per_sec"],
        "generation_tokens_per_sec": headline["generation_tokens_per_sec"],
        "total_training_seconds": summary["total_training_seconds"],
        "peak_memory_mb": summary["peak_memory_mb"],
        "device": summary["device_name"],
    }

    csv_path = REPO_ROOT / "task1_llm" / "shriram_dundigalla" / "metrics_report.csv"
    pd.DataFrame([report]).to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path.relative_to(REPO_ROOT)}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    args = ap.parse_args()
    p = Path(args.checkpoint)
    main(p if p.is_absolute() else REPO_ROOT / p)
