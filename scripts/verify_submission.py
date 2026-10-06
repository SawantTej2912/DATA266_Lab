#!/usr/bin/env python3
"""Check the submission's own claims, per task and across tasks.

    python scripts/verify_submission.py
    python scripts/verify_submission.py --task 1

Every check here exists because something could silently be false: a metric quoted in
`results.md` that no longer matches its CSV, a checkpoint that cannot be loaded without
a file that is not beside it, a "no prebuilt attention" claim that a grader would want
to confirm, a training metric computed on a fraction of the split. The point is that
none of it rests on a claim in prose.

Two design choices worth stating:

**Prohibited-module checks parse the AST, not the text.** `model.py`'s own docstring
names `nn.MultiheadAttention` in order to say it is not used, so a grep for that string
flags a clean file. This walks attribute accesses and imports instead, which is what
actually determines whether a module is used.

**Cross-artifact checks compare against the CSV, not against a hardcoded number.** The
CSV is the machine-written record; prose is transcribed by hand and is what drifts. A
metric quoted in `results.md` therefore has to be findable, at the rounding the prose
uses, in the corresponding CSV column.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
T1 = REPO_ROOT / "task1_llm/shriram_dundigalla"
T2 = REPO_ROOT / "task2_sentiment/shriram_dundigalla"
T3 = REPO_ROOT / "task3_gan/shriram_dundigalla"

RESULTS: list[tuple[bool, str, str]] = []


def check(ok: bool, label: str, detail: str = "") -> bool:
    RESULTS.append((ok, label, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f"  — {detail}" if detail else ""))
    return ok


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# --------------------------------------------------------------------------- helpers

def used_attributes(path: Path) -> set[str]:
    """Dotted names actually *used* in code: attribute accesses and imported names.

    Comments and docstrings are invisible to the AST, which is the entire point.
    """
    tree = ast.parse(path.read_text("utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts, cur = [node.attr], node.value
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr); cur = cur.value
            if isinstance(cur, ast.Name):
                parts.append(cur.id)
                names.add(".".join(reversed(parts)))
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            names.add(mod)
            names.update(f"{mod}.{a.name}" for a in node.names)
    return names


def banned_in(src_dir: Path, banned: dict[str, str]) -> list[str]:
    """Which banned names are used anywhere under `src_dir`, and where."""
    hits = []
    for f in sorted(src_dir.rglob("*.py")):
        if "__pycache__" in f.parts:
            continue
        used = used_attributes(f)
        for name, why in banned.items():
            if any(u == name or u.endswith("." + name.split(".")[-1]) and name in u
                   for u in used):
                hits.append(f"{f.relative_to(REPO_ROOT)}: {name} ({why})")
    return hits


def notebook_text(path: Path) -> str:
    """All markdown and all rendered cell output of a notebook, as one string.

    Cell *source* is excluded on purpose: a cell that contains `max_batches=100` in a
    comment explaining why the cap was removed should not read as a stale value. What a
    grader sees is the output, and that is what has to agree with the CSV.
    """
    nb = json.loads(path.read_text("utf-8"))
    parts: list[str] = []
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            parts.extend(cell["source"])
        for out in cell.get("outputs", []):
            parts.extend(out.get("text", []))
            for mime, val in (out.get("data") or {}).items():
                if mime.startswith("text/"):
                    parts.extend(val if isinstance(val, list) else [val])
    return "".join(parts)


def csv_row(path: Path, **where) -> dict | None:
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if all(r.get(k) == v for k, v in where.items()):
            return r
    return None


def quoted_matches_csv(prose: str, row: dict, column: str, decimals: int,
                       label: str) -> bool:
    """Is the CSV value, at the prose's rounding, actually present in the prose?"""
    raw = row.get(column)
    if raw in (None, ""):
        return check(False, label, f"column '{column}' missing from the CSV")
    want = f"{float(raw):.{decimals}f}"
    # Tolerate thin spaces and thousands separators used in the tables.
    found = want in prose.replace(",", "").replace("\u2009", "")
    return check(found, label, f"CSV {column}={want}"
                 + ("" if found else " — not quoted in the prose document"))


# ------------------------------------------------------------------------- task 1

def verify_task1() -> None:
    section("TASK 1 — GPT from scratch")
    report = csv_row(T1 / "metrics_report.csv", task="task1_llm")
    # Pick the summary that belongs to the *reported* run, by matching the tag the CSV
    # names. Globbing and taking the first match silently compared against a smoke run
    # once, which is the sort of thing this whole script exists to catch.
    # The CSV's `checkpoint` column is the result-to-weights mapping, so deriving the
    # run tag from it checks that mapping rather than assuming it.
    ckpt_rel = Path(report["checkpoint"])
    ckpt_path = REPO_ROOT / ckpt_rel if not ckpt_rel.is_absolute() else ckpt_rel
    check(ckpt_path.exists(), "the checkpoint named in metrics_report.csv exists",
          str(ckpt_rel))
    tag = ckpt_path.stem
    summary_path = T1 / "outputs" / f"train_summary_{tag}.json"
    check(summary_path.exists(), "the reported run's training summary is present",
          summary_path.name)
    summary = json.loads(summary_path.read_text())
    prose = (T1 / "results.md").read_text("utf-8")

    banned = {
        "nn.Transformer": "prebuilt transformer",
        "nn.TransformerEncoder": "prebuilt transformer",
        "nn.TransformerEncoderLayer": "prebuilt transformer",
        "nn.TransformerDecoderLayer": "prebuilt transformer",
        "nn.MultiheadAttention": "prebuilt attention",
        "F.scaled_dot_product_attention": "fused attention kernel",
        "transformers": "pretrained model library",
    }
    hits = banned_in(T1 / "src", banned)
    check(not hits, "no prebuilt Transformer or attention module is used",
          "; ".join(hits) if hits else "AST walk of every module in src/")

    # Window counts and epoch count, from the run's own record.
    hist = summary["history"]
    check(int(report["train_eval_windows"]) == 99968,
          "training metrics cover the full training split",
          f"{report['train_eval_windows']} of 100,000 windows "
          f"(32 held back by drop_last=True)")
    cfg = json.loads((T1 / "src/config.yaml").read_text("utf-8")
                     ) if False else None    # config is YAML; read below
    import yaml
    cfg = yaml.safe_load((T1 / "src/config.yaml").read_text("utf-8"))
    check(cfg["data"]["n_train_windows"] == 100000 and cfg["data"]["n_val_windows"] == 10000,
          "100K training / 10K validation windows configured",
          f"{cfg['data']['n_train_windows']} / {cfg['data']['n_val_windows']}")
    check(len(hist["epoch"]) == 12 and hist["epoch"] == list(range(1, 13)),
          "12 epochs actually ran", f"epochs {hist['epoch'][0]}-{hist['epoch'][-1]}")
    check(cfg["data"]["val_story_frac"] > 0,
          "validation is split by story, not by window",
          f"val_story_frac={cfg['data']['val_story_frac']} — windows are cut inside a "
          f"story, so a window-level split would leak text across the boundary")

    # The checkpoint has to be usable on its own.
    import torch
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    check("tokenizer" in ck and len(ck["tokenizer"]["idx_to_char"]) == ck["vocab_size"],
          "checkpoint contains the tokenizer mapping",
          f"{ck.get('vocab_size')} characters, embedding rows "
          f"{ck['model_state_dict']['tok_emb.weight'].shape[0]}")

    # Causal masking, tested numerically rather than by reading the mask.
    sys.path.insert(0, str(T1 / "src"))
    from model import GPT, GPTConfig   # noqa: E402
    torch.manual_seed(0)
    m = GPT(GPTConfig(**ck["model_config"])); m.load_state_dict(ck["model_state_dict"]); m.eval()
    v, L = ck["vocab_size"], 16
    x = torch.randint(0, v, (1, L))
    with torch.no_grad():
        a, _ = m(x)
        x2 = x.clone(); x2[0, -1] = (x2[0, -1] + 1) % v      # perturb only the last token
        b, _ = m(x2)
    early = (a[0, :-1] - b[0, :-1]).abs().max().item()
    late = (a[0, -1] - b[0, -1]).abs().max().item()
    check(early < 1e-5 < late,
          "attention is causal: a future token cannot change an earlier prediction",
          f"max change at positions 0..{L - 2} is {early:.2e}; at the perturbed "
          f"position it is {late:.2e}")

    # Prose must agree with the machine-written record.
    quoted_matches_csv(prose, report, "train_cross_entropy", 4,
                       "results.md train cross-entropy matches metrics_report.csv")
    quoted_matches_csv(prose, report, "val_cross_entropy", 4,
                       "results.md validation cross-entropy matches the CSV")
    quoted_matches_csv(prose, report, "generalization_gap", 4,
                       "results.md generalization gap matches the CSV")
    quoted_matches_csv(prose, report, "parameter_count", 0,
                       "results.md parameter count matches the CSV")
    for k in ("train_cross_entropy", "val_cross_entropy", "generalization_gap"):
        check(abs(float(report[k]) - float(summary[k])) < 1e-12,
              f"{k} identical in metrics_report.csv and train_summary JSON")

    # The notebook is executed output committed to the repo, so it is an artifact that
    # has to agree with the CSV like any other. It recomputed the training metrics with
    # max_batches=100 long after train.py had stopped doing so.
    nb = notebook_text(T1 / "src/task1_gpt_from_scratch.ipynb")
    check(bool(nb), "task1 notebook has executed output committed",
          f"{len(nb):,} characters of markdown and output")
    for col, dp in (("train_cross_entropy", 4), ("val_cross_entropy", 4),
                    ("generalization_gap", 4)):
        quoted_matches_csv(nb, report, col, dp,
                           f"notebook {col} matches metrics_report.csv")
    # The original run's raw log legitimately shows the pre-correction figures. It is
    # evidence and is not edited -- but the notebook must then explain the difference,
    # or a reader sees two numbers and no reason for them.
    check("0.7110" not in nb or "12.8% of the split" in nb,
          "notebook explains why the raw log's train CE differs from the reported one")

    # Artifacts the brief asks to be committed.
    for pattern, what in (("loss_curves_*.png", "loss and validation curves"),
                          ("generations_*.txt", "generated samples"),
                          ("generation_sweep_*.csv", "multi-draw generation statistics"),
                          ("decoding_comparison_*.csv", "decoding-strategy comparison")):
        check(any((T1 / "outputs").glob(pattern)), f"committed: {what}")
    fa = (T1 / "failure_analysis.md").read_text("utf-8")
    cases = re.findall(r"^#+\s*Case\s+\d+", fa, re.M)
    check(len(cases) >= 3, "failure_analysis.md documents at least three cases",
          f"{len(cases)} numbered cases at any heading level")
    check("MPS" in prose and "no CUDA device was used" in prose,
          "hardware line names MPS and explicitly disclaims CUDA",
          prose.split("Hardware:")[1].split("\n")[0].strip()[:80] + "...")


# ------------------------------------------------------------------------- task 2

def verify_task2() -> None:
    section("TASK 2 — Yelp sentiment")
    import numpy as np
    import torch
    prose = (T2 / "results.md").read_text("utf-8")
    rows = list(csv.DictReader((T2 / "metrics_report.csv").open(encoding="utf-8")))

    check(len(rows) == 3 and {r["model"] for r in rows} ==
          {"baseline_bow", "exp1_bilstm", "exp2_textcnn"},
          "all three models present in metrics_report.csv",
          ", ".join(r["model"] for r in rows))

    z = np.load(T2 / "outputs/test_predictions.npz", allow_pickle=True)
    check(len(z["y_true"]) == 38000, "full official 38,000-review test set",
          f"{len(z['y_true'])} rows, {int(z['y_true'].sum())} positive / "
          f"{int((1 - z['y_true']).sum())} negative")

    banned = {
        "transformers": "pretrained language model library",
        "torchtext.vocab.GloVe": "pretrained embeddings",
        "gensim": "pretrained embeddings",
        "sentence_transformers": "pretrained encoder",
        "fasttext": "pretrained embeddings",
    }
    hits = banned_in(T2 / "src", banned)
    check(not hits, "no pretrained embeddings or language models are used",
          "; ".join(hits) if hits else "AST walk of every module in src/")

    # Both ECE definitions, named.
    for col in ("ece_predicted_class_confidence", "ece_positive_class"):
        check(all(r.get(col) for r in rows), f"metrics_report.csv reports {col}")
    check("predicted" in prose.lower() and "positive-class ECE" in prose,
          "results.md distinguishes the two ECE definitions by name")

    # Required content.
    for pattern, what in ((r"mcnemar", "McNemar tests"),
                          (r"confidence interval", "confidence intervals"),
                          (r"\bslice\b", "slice metrics"),
                          (r"hardware", "hardware disclosure"),
                          (r"no CUDA device was used", "an explicit CUDA disclaimer"),
                          (r"limitation", "limitations"),
                          (r"what i would do next|future work|further work",
                           "future work")):
        check(bool(re.search(pattern, prose, re.I)), f"results.md covers {what}")
    check(all(r.get("accuracy_ci95_low") and r.get("macro_f1_ci95_low") for r in rows),
          "bootstrap CIs present for every model in the CSV")
    check((T2 / "outputs/mcnemar_tests.csv").exists()
          and (T2 / "outputs/slice_metrics.csv").exists(),
          "McNemar and slice metrics have their own CSVs")

    # Checkpoints must be self-contained.
    for f in sorted((T2 / "checkpoints").glob("*.pt")):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        ok = ("vocab_itos" in ck and len(ck["vocab_itos"]) == ck["vocab_size"]
              and "preprocess" in ck and "data_config" in ck)
        check(ok, f"{f.name} carries its vocabulary and preprocessing config",
              f"{ck.get('vocab_size')} tokens, stemmer="
              f"{ck.get('preprocess', {}).get('stemmer')}, "
              f"max_len={ck.get('data_config', {}).get('max_len')}")

    # Error review composition.
    errs = list(csv.DictReader(
        (T2 / "outputs/error_review_exp2_textcnn.csv").open(encoding="utf-8")))
    from collections import Counter
    counts = Counter(r["bucket"] for r in errs)
    check(len(errs) == 20 and set(counts.values()) == {5} and len(counts) == 4,
          "20 reviewed errors, exactly 5 in each of 4 buckets",
          "; ".join(f"{k}={v}" for k, v in counts.items()))
    missing = [r["test_index"] for r in errs
               if not r.get("suggested_error_type") or not r.get("proposed_testable_fix")]
    check(not missing, "every reviewed error has an error type and a testable fix",
          f"missing for indices {missing}" if missing else f"{len(errs)}/20 complete")

    nb = notebook_text(T2 / "src/task2_yelp_sentiment.ipynb")
    check(bool(nb), "task2 notebook has executed output committed",
          f"{len(nb):,} characters")
    for r in rows:
        for col in ("accuracy", "f1_macro"):
            quoted_matches_csv(nb, r, col, 4,
                               f"notebook {r['model']} {col} matches the CSV")

    # Every model, not just the best one: the headline table quotes all three, and a
    # rounding slip in any row is the same defect. Checked at 4 d.p. because that is
    # the precision the table uses.
    for r in rows:
        for col in ("accuracy", "f1_macro", "roc_auc",
                    "ece_predicted_class_confidence", "ece_positive_class"):
            quoted_matches_csv(prose, r, col, 4,
                               f"results.md {r['model']} {col} matches the CSV")


# ------------------------------------------------------------------------- task 3

def verify_task3() -> None:
    section("TASK 3 — CycleGAN")
    from PIL import Image
    prose = (T3 / "results.md").read_text("utf-8")
    full = csv_row(T3 / "full_metrics_report.csv")

    # The Task 3 tree in the brief, transcribed literally. It is stricter than the
    # generic per-member tree in section 5: it adds evaluate_local.py, submission.csv,
    # full_metrics_report.csv, the two prediction directories, and a second copy of the
    # full metrics file inside src/.
    for rel in ("src", "checkpoints", "outputs", "outputs/pred_A2B", "outputs/pred_B2A"):
        check((T3 / rel).is_dir(), f"brief requires the directory {rel}/")
    for rel in ("evaluate_local.py", "submission.csv", "full_metrics_report.csv",
                "src/full_metrics_report.csv", "metrics_report.csv",
                "failure_analysis.md", "results.md"):
        check((T3 / rel).is_file(), f"brief requires the file {rel}")
    check(bool(list(T3.glob("src/*.ipynb"))), "brief requires a code ipynb in src/",
          ", ".join(p.name for p in T3.glob("src/*.ipynb")))
    # Two copies of one table is what the brief asks for; two copies that disagree is
    # the failure that asking for two copies invites.
    check((T3 / "full_metrics_report.csv").read_bytes()
          == (T3 / "src/full_metrics_report.csv").read_bytes(),
          "the src/ and root copies of full_metrics_report.csv are identical")

    sub = (T3 / "submission.csv").read_text("utf-8").strip().splitlines()
    check(sub[0].replace(" ", "") == "ID,FID,MiFID",
          "submission.csv has the required header", sub[0])
    sid, sfid, smifid = sub[1].split(",")
    check(sid.isdigit(), "submission ID is numeric", f"ID={sid}")

    # The submitted value is the mean of the two directions, because that is what the
    # course evaluation script writes. Recomputing it here rather than trusting the
    # stored average is the whole point: it catches a report that quotes one direction.
    a2b, b2a = float(full["FID_A2B_monet2photo"]), float(full["FID_B2A_photo2monet"])
    check(abs(float(sfid) - (a2b + b2a) / 2) < 0.01,
          "submission FID is the mean of both directions",
          f"{sfid} vs ({a2b} + {b2a})/2 = {(a2b + b2a) / 2:.3f}")
    ma2b, mb2a = float(full["MiFID_A2B"]), float(full["MiFID_B2A"])
    check(abs(float(smifid) - (ma2b + mb2a) / 2) < 1e-3,
          "submission MiFID is the mean of both directions",
          f"{smifid} vs {(ma2b + mb2a) / 2:.4f}")

    # Which FID this is, is not a detail. The shipped real_stats.npz was built by
    # scaling to [-1,1]; the graded script uses ImageNet normalization and recomputes
    # its reference from the image folders. The two disagree by tens of points, so a
    # silent revert to the other convention would look like a large score improvement.
    ev = (T3 / "evaluate_local.py").read_text("utf-8")
    check("0.485, 0.456, 0.406" in ev,
          "evaluate_local.py uses the graded ImageNet normalization")
    # Checked against the code, not the whole file: the docstring names the .npz files
    # precisely in order to explain why they are not used.
    code = "\n".join(l for l in ev.splitlines() if not l.lstrip().startswith("#"))
    code = code.split('"""')[-1]
    check("np.load" not in code and ".npz" not in code,
          "evaluate_local.py recomputes reference stats rather than loading a cache")
    check("argparse" in ev and "/content/drive" not in ev,
          "evaluate_local.py takes paths as arguments and hard-codes none")

    # evaluate_local.py is a transcription of a course-provided notebook, so the risk is
    # that the two drift. Keep the original alongside and compare the decisions that
    # change the number: the normalization, the crop, and averaging the two directions.
    ta = REPO_ROOT / "task3_gan/Part3_Evaluation_Script.ipynb"
    if check(ta.exists(), "the course evaluation notebook is kept as provenance"):
        src = "".join("".join(c["source"]) for c in
                      json.loads(ta.read_text("utf-8"))["cells"])
        signatures = ("0.485, 0.456, 0.406", "T.Resize(299)", "T.CenterCrop(299)",
                      "transform_input=False", "cosine(")
        missing = [s for s in signatures if s in src and s not in ev]
        check(not missing, "evaluate_local.py keeps every scoring decision the "
              "course notebook makes", f"diverged on: {missing}" if missing
              else f"all {len(signatures)} signatures match")
        check(("fid_A2B + fid_B2A) / 2" in ev.replace("fid_a2b", "fid_A2B")
               .replace("fid_b2a", "fid_B2A")),
              "evaluate_local.py averages the two directions as the course script does")

    for dirname, label in (("pred_A2B", "Monet->Photo"), ("pred_B2A", "Photo->Monet")):
        gen = T3 / "outputs" / dirname
        files = sorted(f for f in gen.iterdir() if f.suffix.lower() in {".jpg", ".jpeg"})
        if not files:
            check(False, f"{dirname} is populated ({label})",
                  "empty — still in Colab; see DRIVE_ARTIFACTS_TODO.md")
            continue
        check(len(files) >= 300, f"{dirname} has at least the scored 300 images",
              f"{len(files)} files")
        bad = [f.name for f in files[:150] + files[-150:]
               if Image.open(f).size != (256, 256) or Image.open(f).mode != "RGB"]
        check(not bad, f"{dirname} images are 256x256 RGB", f"bad: {bad[:3]}")

    nb = notebook_text(T3 / "src/task3_cyclegan_unet.ipynb")
    check(bool(nb), "task3 notebook has executed output committed",
          f"{len(nb):,} characters")
    quoted_matches_csv(prose, full, "FID_averaged", 2,
                       "results.md quotes the submitted FID")
    quoted_matches_csv(prose, full, "params_per_generator_M", 2,
                       "results.md quotes the generator parameter count")

    # A missing metric has to stay visibly missing. The failure mode this guards
    # against is a placeholder zero that reads like a measurement.
    audit = [v for k, v in full.items() if k.startswith("human_audit")]
    check(bool(audit), "human-audit columns exist in full_metrics_report.csv")
    done = [v for v in audit if "pending" not in str(v).lower()]
    check(not done or len(done) == len(audit),
          "human-audit columns are either all pending or all filled",
          f"{len(done)}/{len(audit)} filled")
    if not done:
        check("pending human audit" in prose.lower()
              or "not yet run" in prose.lower(),
              "results.md states the human audit is outstanding")

    check("official evaluation script" in prose.lower(),
          "results.md names the scoring protocol it used")
    check("kaggle" in prose.lower() and str(full["kaggle_rank"]) in prose,
          "results.md records the leaderboard rank", f"rank {full['kaggle_rank']}")


# ----------------------------------------------------------------------- team-wide

def verify_team() -> None:
    section("TEAM-WIDE")
    for name, d in (("task1", T1), ("task2", T2), ("task3", T3)):
        for f in ("results.md", "failure_analysis.md", "metrics_report.csv"):
            check((d / f).exists(), f"{name}: {f} present")
        check(any((REPO_ROOT / "reproducibility/raw_logs").glob(f"{name.replace('task','task')}*")),
              f"{name}: raw logs committed")
    # One per-member manifest covers all three parts; it has to name each of them and
    # quote the same numbers the CSVs hold, since it is transcribed by hand.
    manifest = (REPO_ROOT / "reproducibility/manifests/"
                "shriram_dundigalla_manifest.md").read_text("utf-8")
    for name in ("task1", "task2", "task3"):
        check(name in manifest, f"{name}: named in the reproducibility manifest")
    t1 = csv_row(T1 / "metrics_report.csv", task="task1_llm")
    for col, dp in (("train_cross_entropy", 4), ("val_cross_entropy", 4),
                    ("generalization_gap", 4)):
        quoted_matches_csv(manifest, t1, col, dp,
                           f"manifest {col} matches metrics_report.csv")
    t3 = csv_row(T3 / "full_metrics_report.csv")
    quoted_matches_csv(manifest, t3, "FID_averaged", 2,
                       "manifest task-3 FID matches the CSV")

    # The report is the artifact furthest from the code, so the check that matters is
    # not "does it exist" but "was it built after the numbers it quotes".
    report = REPO_ROOT / "report/DATA266_Lab1_Report_Team_33.pdf"
    if check(report.exists(), "combined Report.pdf is present", report.name):
        built = report.stat().st_mtime
        stale = [c.relative_to(REPO_ROOT) for c in
                 (T1 / "metrics_report.csv", T2 / "metrics_report.csv",
                  T3 / "metrics_report.csv")
                 if c.stat().st_mtime > built]
        check(not stale, "Report.pdf was built after every CSV it quotes",
              f"rebuild it: {', '.join(map(str, stale))}" if stale
              else "no CSV is newer than the report")
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader
        text = "".join(pg.extract_text() for pg in PdfReader(str(report)).pages)
        check("REPOSITORY URL NOT YET SET" not in text and "github" in text.lower(),
              "Report.pdf carries the GitHub repository link Canvas requires",
              "rebuild with --repo-url https://github.com/<user>/<repo>"
              if "REPOSITORY URL NOT YET SET" in text else "link present")
        # Either member's generator may have produced the official report, and they word
        # these sections differently, so accept any phrasing that means the same thing.
        for needles, what in ((("ownership statement", "who did what", "contribution"),
                               "the ownership statement"),
                              (("comparison across team members", "comparison"),
                               "per-task comparison tables"),
                              (("strengths, weaknesses, limitations", "weakness", "limitation"),
                               "the joint analysis"),
                              (("evidence index", "evidence", "artifact index"),
                               "the evidence index"),
                              (("references", "bibliography", "arxiv"),
                               "paper citations")):
            low = text.lower()
            check(any(n in low for n in needles), f"Report.pdf contains {what}")

    readme = (REPO_ROOT / "README.md").read_text("utf-8")
    check("--smoke" in readme, "README documents a smoke-test command")
    check("MPS" in readme and "CUDA" in readme,
          "README discloses MPS and CUDA hardware separately")

    # Secrets and personal paths, delegated to the packager's scanner. Match on what the
    # scanner *reports*, not on the pattern it detected: it prints a label ("contains a
    # Kaggle API token") and deliberately never echoes the matched text, so grepping this
    # output for the pattern itself would call every real leak clean.
    r = subprocess.run([sys.executable, "scripts/package_submission.py", "--check"],
                       cwd=REPO_ROOT, capture_output=True, text=True)
    leaked = re.findall(r"^\s+- (\S+:\d+ contains a .+)$", r.stdout, re.M)
    check(not leaked, "no secrets or personal paths reported by the scanner",
          "; ".join(leaked)[:200] if leaked else "scanner clean")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", choices=["1", "2", "3", "team", "all"], default="all")
    args = ap.parse_args()

    runners = {"1": verify_task1, "2": verify_task2, "3": verify_task3,
               "team": verify_team}
    for key in (runners if args.task == "all" else [args.task]):
        try:
            runners[key]()
        except Exception as e:      # a check that cannot run is a failed check
            check(False, f"task {key}: verification crashed", f"{type(e).__name__}: {e}")

    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{'=' * 78}")
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("\nFAILED:")
        for _, label, detail in failed:
            print(f"  - {label}" + (f"  — {detail}" if detail else ""))
    print("=" * 78)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
