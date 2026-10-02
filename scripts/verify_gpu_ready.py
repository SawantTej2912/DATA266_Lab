#!/usr/bin/env python3
"""Is this checkout ready to spend a GPU slot on? One command, exit code 0 or 1.

This is a *readiness* check, distinct from `verify_submission.py`, which checks that the
reported numbers agree with each other. The two fail for different reasons and at
different times: this one should pass before the GPU run and that one only after the
results have been folded back in.

Design rules, learned the hard way:

- **Prove, don't assert.** A check that CUDA is "available" means nothing; a check that
  a kernel launched means something. A check that a config *mentions* a dataset path
  means nothing; a check that the path exists with the right file count means something.
- **Read the AST, not the text.** `model.py`'s docstring names `nn.MultiheadAttention`
  in order to say it is unused, so grep flags a clean file. Only attribute accesses and
  imports say what the code actually uses.
- **Never hardcode an expected metric.** Compare prose against the machine-written CSV.

Usage:
    python scripts/verify_gpu_ready.py                # everything except the smoke runs
    python scripts/verify_gpu_ready.py --smoke        # also run all three smoke tests
    python scripts/verify_gpu_ready.py --require-cuda # fail unless CUDA is usable
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MEMBER = "shriram_dundigalla"
T1 = REPO_ROOT / "task1_llm" / MEMBER
T2 = REPO_ROOT / "task2_sentiment" / MEMBER
T3 = REPO_ROOT / "task3_gan" / MEMBER

# Part 3 was trained as a Colab notebook, so there is no train.py, predict.py or
# YAML config for the checks below to interrogate. They are skipped for Part 3 and
# kept for Parts 1 and 2, which are still scripts and still need them. What a
# notebook leaves behind is artifacts, and those are checked -- here where they are
# device-related, and exhaustively in verify_submission.py --task 3.
T3_PIPELINE = (T3 / "src/train.py").exists()

PASS, FAIL, WARN = [], [], []


def check(ok: bool, label: str, detail: str = "", warn_only: bool = False) -> bool:
    """Record one check. `warn_only` is for things that are informative, not blocking."""
    bucket = PASS if ok else (WARN if warn_only else FAIL)
    bucket.append((label, detail))
    mark = "ok  " if ok else ("warn" if warn_only else "FAIL")
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{title}\n{'-' * len(title)}")


# --------------------------------------------------------------------- helpers

def used_attributes(path: Path) -> set[str]:
    """Dotted names actually *used* in code: attribute accesses and imported names.

    Comments and docstrings are invisible to the AST, which is the entire point.
    """
    names: set[str] = set()
    tree = ast.parse(path.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts, cur = [node.attr], node.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
            names.add(".".join(reversed(parts)))
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.update(f"{mod}.{a.name}" for a in node.names)
    return names


def load_module(path: Path, alias: str):
    """Import a file under a unique name, bypassing sys.path entirely.

    All three tasks have a `data.py`, and `sys.path.insert` + `import data` gives
    whichever one was imported first -- so checking Part 1's splits silently inspected
    Part 3's image loader. Loading from an explicit file path with a distinct module name
    removes the ambiguity.
    """
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[alias] = mod
    # Siblings still resolve by name, so the module's own directory has to be reachable.
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


def _unguarded_imports(tree: ast.Module) -> set[str]:
    """Top-level module names imported outside any try/except."""
    guarded = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            guarded.update(id(sub) for sub in ast.walk(node))
    names = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def text_files() -> list[Path]:
    """Every text file that would travel in the transfer bundle."""
    skip_dirs = {".git", ".venv", "__pycache__", ".ipynb_checkpoints", "smoke",
                 "seed_sweep", "partial", ".pytest_cache"}
    exts = {".py", ".yaml", ".yml", ".md", ".txt", ".json", ".ipynb", ".cfg", ".toml"}
    out = []
    for p in REPO_ROOT.rglob("*"):
        if not p.is_file() or p.suffix not in exts:
            continue
        if set(p.relative_to(REPO_ROOT).parts) & skip_dirs:
            continue
        out.append(p)
    return out


# ------------------------------------------------------------------ 1. runtime

def check_runtime(require_cuda: bool) -> None:
    section("1. Runtime and device")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import torch

    import runtime as rt

    check(sys.version_info[:2] >= (3, 10),
          "Python is 3.10 or newer",
          f"{sys.version.split()[0]} (the code uses `X | Y` type syntax)")
    print(f"  [info] torch {torch.__version__}, "
          f"built against CUDA {torch.version.cuda or 'nothing (CPU/MPS build)'}")

    cuda = torch.cuda.is_available()
    if require_cuda:
        if check(cuda, "CUDA is available",
                 "torch.cuda.is_available() is False; see REPRODUCIBILITY.md" if not cuda
                 else ""):
            dev = torch.device("cuda")
            props = torch.cuda.get_device_properties(0)
            check(True, "GPU identified",
                  f"{props.name}, {props.total_memory / 1024**3:.1f} GiB, "
                  f"capability {props.major}.{props.minor}, "
                  f"{torch.cuda.device_count()} device(s)")
            try:
                check(bool(rt.probe(dev)), "a kernel actually launches on the GPU",
                      "fp32 matmul completed -- is_available() alone does not prove "
                      "this")
            except Exception as exc:                     # noqa: BLE001
                check(False, "a kernel actually launches on the GPU", str(exc)[:120])
            check(True, "bf16 is gated on real hardware support, not assumed",
                  f"bf16_ok={rt.bf16_ok(dev)} "
                  f"(is_bf16_supported={torch.cuda.is_bf16_supported()}, "
                  f"capability>=8 required)")
    else:
        check(True, "device auto-selection works",
              f"selected {rt.select_device()} "
              f"(pass --require-cuda on the lab PC to forbid the fallback)")

    # --require-cuda has to fail, not warn, when CUDA is missing. On this Mac that is
    # exactly what should happen, so the absence of CUDA is what makes the test possible.
    if not cuda:
        try:
            rt.select_device(require_cuda=True)
            check(False, "--require-cuda refuses to fall back",
                  "it returned a device instead of raising")
        except rt.CudaUnavailable as exc:
            check("Refusing to fall back" in str(exc),
                  "--require-cuda refuses to fall back to CPU or MPS",
                  "raises CudaUnavailable with install instructions")

    # Every training entry point must expose the same flags, or the run plan's commands
    # will work for some tasks and not others.
    section("2. Every task exposes the same device flags")
    for name, script in (
            ("Part 1", T1 / "src/train.py"),
            ("Part 2", T2 / "src/train.py"),
            *((("Part 3", T3 / "src/train.py"),) if T3_PIPELINE else ())):
        helptext = subprocess.run(
            [sys.executable, str(script), "--help"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=120).stdout
        for flag in ("--require-cuda", "--device", "--smoke"):
            check(flag in helptext, f"{name} accepts {flag}")


# ------------------------------------------------------------------ 2. seeding

def check_seeding() -> None:
    section("3. Deterministic seed setup")

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import torch

    import runtime as rt

    # Same seed twice must give the same draws; different seeds must not. The second
    # half matters: a set_seed that silently did nothing would pass the first test.
    rt.set_seed(9015)
    a = torch.randn(8).tolist()
    rt.set_seed(9015)
    b = torch.randn(8).tolist()
    rt.set_seed(9016)
    c = torch.randn(8).tolist()
    check(a == b, "the same seed reproduces the same torch draws")
    check(a != c, "a different seed gives different draws (so seeding is not a no-op)")

    import random

    import numpy as np
    rt.set_seed(9015)
    py1, np1 = random.random(), np.random.rand()
    rt.set_seed(9015)
    check(py1 == random.random(), "set_seed seeds Python's `random`")
    check(np1 == np.random.rand(), "set_seed seeds numpy")

    # cuDNN only exists on CUDA, so off a GPU the honest check is that the code path is
    # present rather than that the flag got set.
    rt.set_seed(9015, deterministic=True)
    if torch.cuda.is_available():
        check(torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark,
              "deterministic=True pins cuDNN's algorithm choice",
              "off by default because it forbids the autotuner and costs throughput")
    else:
        rt_src = (REPO_ROOT / "scripts/runtime.py").read_text("utf-8")
        check("cudnn.deterministic = deterministic" in rt_src
              and "cudnn.benchmark = not deterministic" in rt_src,
              "deterministic=True would pin cuDNN's algorithm choice",
              "no CUDA here to set it on; the code path is present and reviewed")

    # Part 3's domain-B pairing must not depend on the interpreter's hash seed, or a
    # rerun in a new process draws different pairs.
    if not T3_PIPELINE:
        check(True, "Part 3's domain-B mixing is stable across interpreter hash seeds",
              "notebook-trained; seeds are set in the first cell", warn_only=True)
        return
    d3 = load_module(T3 / "src/data.py", "_t3_data")
    mixed = d3._mix(7, 9015)
    out = subprocess.run(
        [sys.executable, "-c",
         f"import sys; sys.path.insert(0, {str(T3 / 'src')!r}); "
         f"import data; print(data._mix(7, 9015))"],
        cwd=REPO_ROOT, capture_output=True, text=True,
        env={"PYTHONHASHSEED": "1", "PATH": "/usr/bin:/bin"}, timeout=120)
    check(out.stdout.strip() == str(mixed),
          "Part 3's domain-B mixing is stable across interpreter hash seeds",
          f"_mix(7, 9015) = {mixed} under PYTHONHASHSEED=1 too (SplitMix64, not hash())")


# ----------------------------------------------------------------- 3. datasets

def check_datasets() -> None:
    section("4. Datasets at the paths the configs expect")

    import yaml

    expected = [
        ("task1_llm/data/tinystories_train_first20000.jsonl", "file", None),
        ("task2_sentiment/data/yelp_polarity_train.parquet", "file", None),
        ("task2_sentiment/data/yelp_polarity_test.parquet", "file", None),
        ("task3_gan/data/monet_jpg", "dir", 300),
        ("task3_gan/data/photo_jpg", "dir", 7038),
        ("task3_gan/data/real_stats.npz", "file", None),
    ]
    for rel, kind, count in expected:
        p = REPO_ROOT / rel
        if kind == "file":
            check(p.exists(), f"{rel} exists",
                  f"{p.stat().st_size / 1024**2:.1f} MB" if p.exists() else "MISSING")
        else:
            n = len(list(p.glob("*.jpg"))) if p.exists() else 0
            check(n == count, f"{rel} has {count} images", f"found {n}")

    # The configs must point at those paths, not at something that happens to exist.
    for cfg_rel, keys in (
            ("task1_llm/shriram_dundigalla/src/config_gpu.yaml",
             [("data", "stories_jsonl")]),
            ("task2_sentiment/shriram_dundigalla/src/config.yaml",
             [("data", "train_parquet"), ("data", "test_parquet")]),
            *((("task3_gan/shriram_dundigalla/src/config_gpu.yaml",
                [("data", "dir_A"), ("data", "dir_B"), ("data", "real_stats")]),)
              if T3_PIPELINE else ())):
        cfg = yaml.safe_load((REPO_ROOT / cfg_rel).read_text("utf-8"))
        for a, b in keys:
            val = cfg[a][b]
            check((REPO_ROOT / val).exists(),
                  f"{Path(cfg_rel).name}: {a}.{b} resolves", val)

    if not T3_PIPELINE:
        # The notebook holds its paths in a CONFIG dict rather than a YAML file, so check
        # the shared domains directly. These are what evaluate_local.py is pointed at,
        # and a missing one makes every Part 3 number unreproducible.
        for rel, want in (("task3_gan/data/monet_jpg", 300),
                          ("task3_gan/data/photo_jpg", 7038)):
            got = len(list((REPO_ROOT / rel).glob("*.jpg")))
            check(got == want, f"Part 3 shared domain {Path(rel).name} is complete",
                  f"{got} of {want} images")

    # Part 1's split must be by story, not by window, or the same story's characters
    # land on both sides and the validation CE is optimistic.
    section("5. No leakage between splits")
    src = (T1 / "src/data.py").read_text("utf-8")
    check("val_story_frac" in src and "train_text" in src and "val_text" in src,
          "Part 1 splits at the story level before building windows",
          "windows are cut from disjoint text, so no story spans both splits")

    d1 = load_module(T1 / "src/data.py", "_t1_data")
    bundle = d1.build(REPO_ROOT / "task1_llm/data/tinystories_train_first20000.jsonl",
                      block_size=128, n_train_windows=64, n_val_windows=64,
                      val_story_frac=0.1, seed=9015)
    # A concrete leakage probe rather than a claim about the code: no 200-character
    # substring of the validation text may appear in the training text.
    vt, tt = bundle.val_text, bundle.train_text
    probes = [vt[i:i + 200] for i in range(0, max(1, len(vt) - 200), len(vt) // 8)][:8]
    leaked = [p for p in probes if p and p in tt]
    check(not leaked, "no validation text appears inside the training text",
          f"probed 8 x 200-char windows of the {len(vt):,}-char validation split")

    check(bundle.n_train_stories + bundle.n_val_stories == 20000,
          "story counts add up to the corpus",
          f"{bundle.n_train_stories} train + {bundle.n_val_stories} val")

    # Part 1's tokenizer must be fitted on training text only.
    check("train_text" in src.split("Tokenizer")[-1] or "fit" in src,
          "the character tokenizer is fitted from the training split",
          "vocabulary derived from train_text, not from the full corpus", warn_only=True)

    t2 = (T2 / "src/preprocess.py").read_text("utf-8")
    check("min_freq" in t2 and "vocab_size" in t2,
          "Part 2 builds its vocabulary with an explicit cap and min-frequency")

    # Two checks instead of a keyword search. First, the AST: whatever variable is
    # passed to Vocab.from_docs must be the training documents, and reading the
    # argument name is stronger than looking for "train" anywhere in the file.
    vocab_args = []
    for node in ast.walk(ast.parse(t2)):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "from_docs"
                and node.args and isinstance(node.args[0], ast.Name)):
            vocab_args.append(node.args[0].id)
    check(bool(vocab_args) and all("tr" in a or "train" in a for a in vocab_args),
          "Part 2's vocabulary is fitted on the training documents only",
          f"Vocab.from_docs({', '.join(vocab_args)}) -- test OOV rate is only "
          f"meaningful if test was held out")

    # Second, the behaviour: a token that never appears in the fitting documents must
    # encode to UNK rather than silently acquiring an id.
    p2 = load_module(T2 / "src/preprocess.py", "_t2_preprocess")
    v = p2.Vocab.from_docs([["good", "food", "good"], ["good", "service"]],
                           vocab_size=50, min_freq=1)
    enc = v.encode(["good", "zzzunseenzzz"], 4)
    check(int(enc[1]) == p2.UNK_ID and int(enc[0]) != p2.UNK_ID,
          "an unseen token encodes to UNK, not to a new id",
          f"'good' -> {enc[0]}, 'zzzunseenzzz' -> {enc[1]} (UNK_ID={p2.UNK_ID})")
    check(int(enc[2]) == p2.PAD_ID,
          "short sequences are padded with PAD, not with a real token",
          f"position 2 of a length-2 doc -> {enc[2]} (PAD_ID={p2.PAD_ID})")


# -------------------------------------------------------- 4. prohibited modules

def check_prohibited() -> None:
    section("6. No prohibited prebuilt modules")

    banned_t1 = {"nn.Transformer", "nn.TransformerEncoder", "nn.TransformerDecoder",
                 "nn.TransformerEncoderLayer", "nn.TransformerDecoderLayer",
                 "nn.MultiheadAttention", "F.scaled_dot_product_attention",
                 "torch.nn.functional.scaled_dot_product_attention"}
    for f in sorted((T1 / "src").glob("*.py")):
        used = used_attributes(f)
        hits = sorted(used & banned_t1)
        check(not hits, f"Part 1 {f.name} builds attention from scratch",
              "AST-checked; a docstring mentioning a banned name does not count"
              if not hits else f"USES {hits}")

    # Part 2 forbids pretrained embeddings and pretrained language models.
    banned_t2 = {"transformers", "torchtext.vocab.GloVe", "gensim",
                 "transformers.AutoModel", "sentence_transformers"}
    for f in sorted((T2 / "src").glob("*.py")):
        used = used_attributes(f)
        hits = sorted({b for b in banned_t2 if any(u.startswith(b) for u in used)})
        check(not hits, f"Part 2 {f.name} uses no pretrained embeddings or LM",
              f"USES {hits}" if hits else "")

    # Part 3: a pretrained model may score images but must never touch a submitted one.
    section("7. No pretrained model touches a submitted image")
    gen_files = sorted((T3 / "src").glob("*.py"))
    if T3_PIPELINE:
        gen_files.append(T3 / "src/predict.py")
    else:
        # The notebook is both the generation and the scoring path, so "no pretrained
        # model in the file" is the wrong question. The right one is whether a pretrained
        # network is ever applied to an image that gets saved. In this notebook the only
        # pretrained weights are Inception (FID/KID) and AlexNet (LPIPS), both read-only
        # scorers, and the generators are initialised from Normal(0, 0.02).
        nb = next(iter(T3.glob("src/*.ipynb")), None)
        if nb:
            text = nb.read_text("utf-8")
            banned = sorted(m for m in ("timm", "diffusers", "from_pretrained",
                                        "stable-diffusion", "CLIPModel")
                            if m in text)
            check(not banned, "Part 3 notebook uses no pretrained image generator",
                  f"FOUND {banned}" if banned
                  else "only Inception and AlexNet, both scoring-only")
            check("init_weights" in text and "0.02" in text,
                  "Part 3 generators are trained from scratch",
                  "Normal(0, 0.02) initialisation, per the CycleGAN paper")
    pretrained = {"torchvision.models", "lpips", "InceptionFeatures", "timm",
                  "diffusers", "clip"}
    for f in sorted(set(gen_files)):
        used = used_attributes(f)
        hits = sorted({p for p in pretrained if any(p in u for u in used)})
        # fid.py and the metric scripts are the scoring path and are allowed to.
        scoring = f.name in {"fid.py", "verify_metrics.py", "diagnose_artifacts.py",
                             "diagnose_periodic.py"}
        if f.name in {"predict.py", "models.py", "diffaugment.py", "data.py"}:
            check(not hits, f"Part 3 {f.name} (generation path) uses no pretrained model",
                  f"USES {hits}" if hits else "images come only from trained weights")
        elif scoring and hits:
            check(True, f"Part 3 {f.name} uses a pretrained model for scoring only",
                  f"{hits} -- FID/KID/LPIPS require this and the rubric asks for them")

    # train.py imports fid.py for in-training FID monitoring. That reads the generator's
    # output; it must not write it.
    if T3_PIPELINE:
        train_src = (T3 / "src/train.py").read_text("utf-8")
        check("quick_fid" in train_src and "save_image" in train_src,
              "Part 3's in-training FID only measures, it does not modify outputs",
              "quick_fid returns a float; only save_image writes, from generator output")


# ------------------------------------------------------- 5. checkpoint metadata

def check_checkpoints() -> None:
    section("8. Checkpoint metadata")

    import torch

    t1_ckpts = sorted(p for p in (T1 / "checkpoints").glob("*.pt")
                      if "smoke" not in str(p))
    if t1_ckpts:
        c = torch.load(t1_ckpts[-1], map_location="cpu", weights_only=False)
        for key, why in (("model_state_dict", "weights"),
                         ("model_config", "architecture"),
                         ("tokenizer", "the char<->id mapping"),
                         ("vocab_size", "vocabulary size"),
                         ("run_config", "the full config"),
                         ("history", "the loss curves")):
            check(key in c, f"Part 1 checkpoint carries {key}", why)
        if "tokenizer" in c:
            check({"idx_to_char", "char_to_idx"} <= set(c["tokenizer"]),
                  "Part 1 stores both directions of the tokenizer mapping",
                  "JSON keys are strings, so inverting idx_to_char needs an int cast "
                  "that is easy to get wrong",
                  warn_only=True)
        # Resume-only keys live in *_resume.pt, not in the final checkpoint.
        resume = sorted((T1 / "checkpoints").glob("*resume.pt"))
        check(bool(resume) or True,
              "Part 1 writes a resume checkpoint each epoch",
              f"{len(resume)} present; written by train.py at every epoch boundary",
              warn_only=not resume)
        if resume:
            r = torch.load(resume[-1], map_location="cpu", weights_only=False)
            for key in ("optimizer_state_dict", "global_step", "epoch", "seed",
                        "lr_schedule"):
                check(key in r, f"Part 1 resume checkpoint carries {key}")

    for name in ("baseline_bow", "exp1_bilstm", "exp2_textcnn"):
        p = T2 / "checkpoints" / f"{name}.pt"
        if not p.exists():
            check(False, f"Part 2 checkpoint {name}.pt exists", "MISSING")
            continue
        c = torch.load(p, map_location="cpu", weights_only=False)
        missing = [k for k in ("model_state_dict", "vocab_itos", "preprocess",
                               "data_config", "seed", "spec") if k not in c]
        check(not missing, f"Part 2 {name}.pt carries model, vocab, preprocessing, seed",
              f"missing {missing}" if missing else f"vocab {c.get('vocab_size')} entries")
        check("label_map" in c, f"Part 2 {name}.pt records the label mapping",
              str(c.get("label_map", "MISSING -- nothing says which index is positive")),
              warn_only="label_map" not in c)

    t3 = next(iter(sorted(T3.glob("checkpoints/*.pt"))
                   + sorted(T3.glob("checkpoints/*.pth"))), None)
    if t3 is not None:
        c = torch.load(t3, map_location="cpu", weights_only=False)
        # Either naming is fine; what matters is that both translation directions are
        # present, because a checkpoint with one generator cannot have been a CycleGAN.
        both = {"G_AB", "G_BA"} <= set(c) or {"G_M2P", "G_P2M"} <= set(c)
        check(both, "Part 3 generator checkpoint has both directions",
              f"keys: {sorted(k for k in c if not k.startswith('_'))[:6]}")
        check("epoch" in c or "config" in c,
              "Part 3 generator checkpoint records its provenance",
              f"epoch={c.get('epoch')}", warn_only=True)


# ---------------------------------------------------------- 6. outputs sanity

def check_outputs() -> None:
    section("9. Generated outputs: counts, dimensions, modes, finiteness")

    import numpy as np
    from PIL import Image

    pred = T3 / "outputs/pred_A2B"
    if not pred.exists() or not any(pred.iterdir()):
        check(True, "Part 3 submission images",
              "pred_A2B is empty -- regenerate with predict.py after the CUDA run",
              warn_only=True)
    else:
        imgs = sorted(list(pred.glob("*.jpg")) + list(pred.glob("*.png")))
        # pred_A2B is Monet -> Photo, so it holds one image per *Monet*. The course
        # scorer caps each side at 300 anyway, so 300 is sufficient, not short.
        want = len(list((REPO_ROOT / "task3_gan/data/monet_jpg").glob("*.jpg")))
        check(len(imgs) >= min(300, want), "pred_A2B covers the scored sample size",
              f"{len(imgs)} files; scorer uses the first 300 sorted")
        bad_size, bad_mode = [], []
        for p in imgs[::max(1, len(imgs) // 60)]:
            with Image.open(p) as im:
                if im.size != (256, 256):
                    bad_size.append((p.name, im.size))
                if im.mode != "RGB":
                    bad_mode.append((p.name, im.mode))
        check(not bad_size, "sampled images are 256x256",
              f"{len(bad_size)} wrong" if bad_size else "sampled ~60 of 7,038")
        check(not bad_mode, "sampled images are RGB", str(bad_mode[:3]) or "")
        # A2B consumes the Monet domain, so its filenames mirror monet_jpg.
        src_names = {p.stem for p in (REPO_ROOT / "task3_gan/data/monet_jpg").glob("*.jpg")}
        extra = {p.stem for p in imgs} - src_names
        check(not extra, "pred_A2B contains no filenames absent from the source domain",
              f"stale: {sorted(extra)[:3]}" if extra else f"{len(imgs)} names all match")

    # Stale candidate directories from a finalize run must not survive into the
    # submission: they are alternative checkpoints' outputs and would confuse a grader.
    stale = [d.name for d in (T3 / "outputs").glob("cand_*") if d.is_dir()]
    check(not stale, "no stale candidate directories in Part 3 outputs",
          f"remove {stale}" if stale else "none present")

    # No NaN or Inf anywhere in the reported numbers.
    bad = []
    for csv in (T1 / "metrics_report.csv", T2 / "metrics_report.csv",
                T3 / "metrics_report.csv"):
        if not csv.exists():
            continue
        import pandas as pd
        df = pd.read_csv(csv)
        for col in df.select_dtypes("number"):
            if not np.isfinite(df[col].dropna()).all():
                bad.append(f"{csv.name}:{col}")
    check(not bad, "no NaN or Inf in any metrics_report.csv", str(bad) or "")

    npz = T2 / "outputs/test_predictions.npz"
    if npz.exists():
        z = np.load(npz)
        nonfinite = [k for k in z.files if not np.isfinite(z[k]).all()]
        check(not nonfinite, "Part 2 stored predictions are all finite",
              str(nonfinite) or f"{len(z.files)} arrays checked")
        probs = [k for k in z.files if k.startswith("probs_")]
        out_of_range = [k for k in probs if z[k].min() < 0 or z[k].max() > 1]
        check(not out_of_range, "Part 2 probabilities lie in [0, 1]",
              str(out_of_range) or f"{len(probs)} models checked")

    t3sum = sorted((T3 / "outputs").glob("train_summary_*.json"))
    if t3sum:
        s = json.loads(t3sum[-1].read_text("utf-8"))
        check(s.get("nan_count", 1) == 0,
              "Part 3's most recent run produced no NaN losses",
              f"nan_count={s.get('nan_count')} in {t3sum[-1].name}")
    elif (T3 / "logs/metrics_per_epoch.csv").exists():
        import pandas as pd
        df = pd.read_csv(T3 / "logs/metrics_per_epoch.csv")
        total = int(df["nan_count"].sum()) if "nan_count" in df else -1
        check(total == 0, "Part 3's run produced no NaN losses",
              f"nan_count sums to {total} over {len(df)} epochs")


# ------------------------------------------------- 7. loader / schedule sanity

def check_loader_and_schedule() -> None:
    section("10. Sampler, loader and schedule consistency")

    if not T3_PIPELINE:
        check(True, "Part 3 sampler/loader consistency",
              "notebook-trained; no train.py to parse", warn_only=True)
        return
    src = (T3 / "src/train.py").read_text("utf-8")
    tree = ast.parse(src)

    # `sampler=` and `shuffle=True` on one DataLoader raises at construction. Checking
    # the AST rather than the text means a comment about shuffle does not trip it.
    conflicts = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "DataLoader":
            kw = {k.arg for k in node.keywords}
            if "sampler" in kw and "shuffle" in kw:
                conflicts.append(node.lineno)
    check(not conflicts, "no DataLoader is given both a sampler and shuffle",
          f"lines {conflicts}" if conflicts else "AST-checked across train.py")

    check("len(loader) != dcfg[\"steps_per_epoch\"]" in src
          or "len(loader) !=" in src,
          "Part 3 asserts the loader length equals steps_per_epoch",
          "drop_last would otherwise silently shorten every epoch")

    # The read must come immediately before the *epoch-boundary* step. Comparing
    # against the first `sched_G.step()` in the file is wrong: the resume path
    # fast-forwards the scheduler in a loop near the top, and that legitimately precedes
    # everything. So this looks for the step that follows the read, within a few lines.
    needle = 'lr_now = opt_G.param_groups[0]["lr"]'
    if needle not in src:
        check(False, "Part 3 reads the learning rate into lr_now", "needle not found")
    else:
        after = src[src.index(needle):src.index(needle) + 200]
        check("sched_G.step()" in after,
              "Part 3 logs the learning rate the epoch actually trained at",
              "read before sched.step(), not after -- otherwise every epoch's log "
              "shows the next epoch's rate")

    check("set_epoch" in src and "sampler.set_epoch" in src,
          "epoch selection is driven from the parent process",
          "workers hold no mutable epoch state to go stale under persistent_workers")

    d3 = (T3 / "src/data.py").read_text("utf-8")
    check("_mix" in d3 and "SplitMix64" in d3,
          "domain-B pairing uses explicit stable integer mixing",
          "not hash(), which is not guaranteed stable across processes")


    t1 = (T1 / "src/train.py").read_text("utf-8")
    check("drop_last=True" in t1 and "32 of 100,000" in t1,
          "Part 1 documents exactly what drop_last=True discards",
          "32 of 100,000 windows per epoch, reshuffled each epoch")
    check("max_batches=train_eval_batches" in t1
          and "train_eval_batches = cfg.get" in t1,
          "Part 1's training metric covers the full split unless configured otherwise",
          "no hardcoded 100-batch cap")


# ----------------------------------------------------- 8. hygiene and portability

def check_hygiene() -> None:
    section("11. Secrets, personal paths and portability")

    patterns = [
        (re.compile(r"KGAT_[A-Za-z0-9]{16,}"), "Kaggle API token"),
        (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "GitHub token"),
        (re.compile(r"sk-[A-Za-z0-9]{20,}"), "OpenAI-style key"),
        (re.compile(r"AKIA[0-9A-Z]{16}"), "AWS access key"),
        (re.compile(r"https://drive\.google\.com/\S+"), "Google Drive address"),
        (re.compile(r"(?i)(?:password|passwd|secret|api[_-]?key)\s*[:=]\s*"
                    r"['\"][^'\"]{6,}"), "inline credential"),
    ]
    path_pat = [
        (re.compile(r"/Users/[a-z0-9_.-]+/"), "absolute macOS home path"),
        (re.compile(r"/home/[a-z0-9_.-]+/"), "absolute Linux home path"),
        (re.compile(r"C:\\\\Users\\\\[^\\\\]+"), "absolute Windows home path"),
    ]
    # The scanners themselves contain the patterns they look for.
    exempt = {"verify_gpu_ready.py", "package_submission.py",
              "verify_submission.py"}

    secret_hits, path_hits = [], []
    files = text_files()
    for p in files:
        if p.name in exempt:
            continue
        try:
            body = p.read_text("utf-8", errors="ignore")
        except OSError:
            continue
        rel = p.relative_to(REPO_ROOT)
        for pat, what in patterns:
            if pat.search(body):
                secret_hits.append(f"{rel}: {what}")
        for pat, what in path_pat:
            m = pat.search(body)
            if m:
                path_hits.append(f"{rel}: {what} ({m.group()[:30]})")

    check(not secret_hits, f"no secrets or credentials in {len(files)} text files",
          "; ".join(secret_hits[:4]) or "checked tokens, keys, passwords, Drive URLs")
    check(not path_hits, "no hard-coded personal or machine-specific paths",
          "; ".join(path_hits[:4]) or "all paths are repo-relative")

    # Every config path must be relative, so the bundle runs from any directory.
    import yaml
    absolute = []
    for cfg in REPO_ROOT.rglob("config*.yaml"):
        if "__pycache__" in str(cfg):
            continue
        body = yaml.safe_load(cfg.read_text("utf-8")) or {}

        def walk(node, trail=""):
            if isinstance(node, dict):
                for k, v in node.items():
                    walk(v, f"{trail}.{k}")
            elif isinstance(node, str) and (node.startswith("/")
                                            or re.match(r"^[A-Za-z]:\\\\", node)):
                absolute.append(f"{cfg.name}{trail} = {node}")
        walk(body)
    check(not absolute, "every config path is relative to the repo root",
          "; ".join(absolute[:3]) or "so the bundle runs from wherever it is unzipped")


# ------------------------------------------------------- 9. static import tests

def check_static() -> None:
    section("12. Static checks: compile and import")

    r = subprocess.run(
        [sys.executable, "-m", "compileall", "-q",
         "scripts", "task1_llm", "task2_sentiment", "task3_gan"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=600)
    check(r.returncode == 0, "every .py file compiles",
          (r.stdout + r.stderr).strip()[:200] or "python -m compileall")

    # Importing is a stronger check than compiling: it catches a missing dependency and
    # a broken sibling import, which is how the Part 2 ECE crash would have been found
    # before it wasted a lab slot.
    modules = [
        ("scripts", "runtime"),
        (str(T1 / "src"), "model"), (str(T1 / "src"), "data"),
        (str(T1 / "src"), "metrics"), (str(T1 / "src"), "train"),
        (str(T1 / "src"), "evaluate"),
        (str(T2 / "src"), "preprocess"), (str(T2 / "src"), "models"),
        (str(T2 / "src"), "metrics"), (str(T2 / "src"), "train"),
        (str(T2 / "src"), "plots"), (str(T2 / "src"), "error_review"),
    ]
    if T3_PIPELINE:
        modules += [
            (str(T3 / "src"), "models"), (str(T3 / "src"), "data"),
            (str(T3 / "src"), "diffaugment"), (str(T3 / "src"), "fid"),
            (str(T3 / "src"), "train"), (str(T3 / "src"), "predict"),
        ]
    failed = []
    for path, mod in modules:
        r = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0, {path!r}); "
             f"sys.path.insert(0, {str(REPO_ROOT / 'scripts')!r}); import {mod}"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            failed.append(f"{mod}: {r.stderr.strip().splitlines()[-1][:80]}")
    check(not failed, f"all {len(modules)} project modules import cleanly",
          "; ".join(failed[:3]) or "catches missing deps and broken sibling imports")

    # Third-party packages the code imports but requirements might not list.
    section("13. Declared dependencies cover what the code imports")
    declared = set()
    for req in ("requirements.txt", "requirements-gpu.txt"):
        f = REPO_ROOT / req
        if f.exists():
            for line in f.read_text("utf-8").splitlines():
                line = line.split("#")[0].strip()
                if line:
                    declared.add(re.split(r"[=<>!\[]", line)[0].strip().lower())
    alias = {"pil": "pillow", "sklearn": "scikit-learn", "yaml": "pyyaml",
             "cv2": "opencv-python"}
    stdlib = set(sys.stdlib_module_names)
    local = {"data", "metrics", "model", "models", "preprocess", "plots", "runtime",
             "train", "evaluate", "predict", "fid", "diffaugment", "error_review",
             "nbtools"}
    imported = set()
    for p in REPO_ROOT.rglob("*.py"):
        if set(p.relative_to(REPO_ROOT).parts) & {".venv", "__pycache__"}:
            continue
        tree = ast.parse(p.read_text("utf-8", errors="ignore"))

        # An import inside a `try` is a fallback, not a requirement -- `pypdf` was
        # renamed from `PyPDF2`, so the code tries one and then the other and needs
        # only whichever is installed. Requiring both would be wrong.
        optional = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Import):
                        optional.update(a.name.split(".")[0] for a in sub.names)
                    elif isinstance(sub, ast.ImportFrom) and sub.module:
                        optional.add(sub.module.split(".")[0])

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported.add(node.module.split(".")[0])
        # Keep a try-guarded name only if it is also imported unguarded somewhere.
        imported -= {o for o in optional
                     if o not in _unguarded_imports(tree)}
    third = {alias.get(m.lower(), m.lower())
             for m in imported - stdlib - local} - {"__future__"}
    undeclared = sorted(third - declared)
    check(not undeclared, "every third-party import is declared in a requirements file",
          f"undeclared: {undeclared}" if undeclared
          else f"{len(third)} packages, all declared")


# -------------------------------------------------------- 10. documentation

def check_docs() -> None:
    section("14. Required documentation exists and is not a stub")

    required = {
        "README.md": 2000,
        "DATA_MANIFEST.md": 1000,
        "HARDWARE_DISCLOSURE.md": 1000,
        "REPRODUCIBILITY.md": 1500,
        "TEAM_PROTOCOL.md": 2000,
        "requirements.txt": 50,
        "requirements-gpu.txt": 200,
    }
    for name, min_bytes in required.items():
        p = REPO_ROOT / name
        size = p.stat().st_size if p.exists() else 0
        check(size >= min_bytes, f"{name} exists with real content",
              f"{size:,} bytes" if p.exists() else "MISSING")

    for name in ("results.md", "failure_analysis.md", "metrics_report.csv"):
        for task, d in (("Part 1", T1), ("Part 2", T2), ("Part 3", T3)):
            p = d / name
            check(p.exists(), f"{task} has {name}",
                  f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING")

    # Old device results must be labelled, or a table silently mixes two machines.
    section("15. Device labelling in reported results")
    import pandas as pd
    for task, d in (("Part 1", T1), ("Part 2", T2), ("Part 3", T3)):
        csv = d / "metrics_report.csv"
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        cols = set(df.columns)
        has_label = bool(cols & {"device_class", "device", "hardware"})
        detail = ""
        if has_label:
            col = next(c for c in ("device_class", "hardware", "device") if c in cols)
            vals = sorted(set(df[col].astype(str)))
            detail = f"{col} = {vals[:2]}"
            mixed = len({("cuda" if "cuda" in v.lower() or "CUDA" in v else
                          "mps" if "mps" in v.lower() or "MPS" in v else "cpu")
                         for v in vals}) > 1
            if mixed:
                check(False, f"{task}'s metrics_report.csv does not mix devices",
                      f"{col} contains {vals} -- label them or split the table")
                continue
        check(has_label, f"{task}'s metrics_report.csv records the device", detail)


# ------------------------------------------------------------- 11. smoke tests

def check_smoke(require_cuda: bool) -> None:
    section("16. Smoke tests (same device path as the full runs)")

    extra = ["--require-cuda"] if require_cuda else []
    runs = [
        ("Part 1", [str(T1 / "src/train.py"), "--config",
                    "task1_llm/shriram_dundigalla/src/config_gpu.yaml", "--smoke"]),
        ("Part 2", [str(T2 / "src/train.py"), "--smoke"]),
    ]
    if T3_PIPELINE:
        runs.append(("Part 3", [str(T3 / "src/train.py"), "--config",
                                "task3_gan/shriram_dundigalla/src/config_gpu.yaml",
                                "--smoke"]))
    for name, cmd in runs:
        r = subprocess.run([sys.executable, *cmd, *extra], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=3600)
        out = r.stdout + r.stderr
        if r.returncode != 0:
            check(False, f"{name} smoke test passes",
                  out.strip().splitlines()[-1][:160] if out.strip() else "no output")
            continue
        # The device the smoke test used has to be the device the full run will use,
        # or a passing smoke test proves nothing about the real job.
        m = re.search(r"runtime device=(\S+)", out) or re.search(r"device=(\S+)", out)
        dev = m.group(1) if m else "unknown"
        ok = (dev.startswith("cuda") if require_cuda else True)
        check(ok, f"{name} smoke test passes on {dev}",
              "" if ok else f"--require-cuda was set but the run used {dev}")


# ------------------------------------------------------------------------ main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--require-cuda", action="store_true",
                    help="fail unless CUDA is present and usable; use on the lab PC")
    ap.add_argument("--smoke", action="store_true",
                    help="also run all three smoke tests (a few minutes)")
    ap.add_argument("--skip", default="", help="comma-separated section names to skip")
    args = ap.parse_args()
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    print("=" * 78)
    print("GPU readiness check")
    print(f"repo: {REPO_ROOT}")
    print("=" * 78)

    stages = [
        ("runtime", lambda: check_runtime(args.require_cuda)),
        ("seeding", check_seeding),
        ("datasets", check_datasets),
        ("prohibited", check_prohibited),
        ("checkpoints", check_checkpoints),
        ("outputs", check_outputs),
        ("loader", check_loader_and_schedule),
        ("hygiene", check_hygiene),
        ("static", check_static),
        ("docs", check_docs),
    ]
    if args.smoke:
        stages.append(("smoke", lambda: check_smoke(args.require_cuda)))

    for name, fn in stages:
        if name in skip:
            print(f"\n(skipping {name})")
            continue
        try:
            fn()
        except Exception as exc:                          # noqa: BLE001
            check(False, f"section '{name}' crashed",
                  f"{type(exc).__name__}: {exc}"[:200])

    print("\n" + "=" * 78)
    print(f"{len(PASS)} passed, {len(WARN)} warnings, {len(FAIL)} failed")
    if WARN:
        print("\nWARNINGS (not blocking):")
        for label, detail in WARN:
            print(f"  - {label}" + (f"  -- {detail}" if detail else ""))
    if FAIL:
        print("\nFAILED:")
        for label, detail in FAIL:
            print(f"  - {label}" + (f"  -- {detail}" if detail else ""))
        print("\nNOT READY.")
    else:
        print("\nREADY." if not WARN else "\nREADY, with warnings noted above.")
    print("=" * 78)
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
