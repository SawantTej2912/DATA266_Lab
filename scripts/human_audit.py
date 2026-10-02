"""Blinded human audit of translated images, with inter-rater agreement.

The rubric asks for "a blinded human audit of 30 fixed samples (style, content,
artifacts) with 2 raters, and report inter-rater agreement". Two subcommands:

    # once, after both members have generated their outputs
    python scripts/human_audit.py build \
        --source shriram=task3_gan/shriram_dundigalla/outputs/pred_A2B \
        --source tejas=task3_gan/tejas_sawant/outputs/pred_A2B

    # after both raters have filled in their sheet
    python scripts/human_audit.py score

What "blinded" buys us. With two members' models in the pool and provenance hidden
behind shuffled sample IDs, a rater cannot favour their own model, which is the bias
that would otherwise make the whole exercise worthless -- each of us is rating our own
work against our teammate's. The mapping from sample ID back to model is written to a
separate key file that raters are not meant to open, and scoring joins on it afterwards.

Real Monet paintings are mixed in as unlabelled controls. They are not part of the 30
scored samples; they are there so that a rater who assigns the same score to everything
is visible in the results rather than invisible. If the controls do not score clearly
higher on style than the generated images, the ratings carry no signal and the
agreement number is not worth reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "task3_gan/human_audit"
EXT = {".jpg", ".jpeg", ".png"}

DIMENSIONS = ["style", "content", "artifacts"]
SCALE = {
    "style": "1 = not Monet-like at all, 5 = convincingly Monet-like",
    "content": "1 = original scene unrecognisable, 5 = scene fully preserved",
    "artifacts": "1 = severe artifacts (blobs, banding, colour bleed), 5 = clean",
}
N_SAMPLES = 30
N_CONTROLS = 5


def build(sources: dict[str, Path], seed: int, monet_dir: Path) -> None:
    pools = {}
    for name, d in sources.items():
        files = sorted(q for q in d.iterdir() if q.suffix.lower() in EXT)
        if not files:
            raise SystemExit(f"source {name!r} has no images in {d}")
        pools[name] = files
        print(f"  {name}: {len(files)} images")

    rng = random.Random(seed)

    # Split the 30 evenly across models, and pick the *same* underlying photos for each
    # model where possible, so a rater is comparing two translations of one scene
    # rather than two unrelated scenes. Comparing different scenes would confound model
    # quality with how hard the individual photo is.
    per_model = N_SAMPLES // len(pools)
    common = sorted(set.intersection(*[{f.name for f in v} for v in pools.values()]))
    chosen_names = rng.sample(common, per_model) if len(common) >= per_model else None

    entries = []
    for name, files in pools.items():
        if chosen_names:
            picks = [f for f in files if f.name in set(chosen_names)]
        else:
            picks = rng.sample(files, per_model)
        for p in picks:
            entries.append({"model": name, "source_image": p.name, "path": p})

    controls = []
    monets = sorted(q for q in monet_dir.iterdir() if q.suffix.lower() in EXT)
    for p in rng.sample(monets, min(N_CONTROLS, len(monets))):
        controls.append({"model": "__control_real_monet", "source_image": p.name, "path": p})

    everything = entries + controls
    rng.shuffle(everything)

    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)
    images = AUDIT_DIR / "images"
    images.mkdir(parents=True)

    key = []
    for i, e in enumerate(everything, start=1):
        sid = f"S{i:03d}"
        shutil.copy2(e["path"], images / f"{sid}{e['path'].suffix.lower()}")
        key.append({"sample_id": sid, "model": e["model"],
                    "source_image": e["source_image"],
                    "is_control": e["model"].startswith("__control")})

    (AUDIT_DIR / "KEY_do_not_open_until_scored.json").write_text(
        json.dumps({"seed": seed, "n_scored": len(entries),
                    "n_controls": len(controls), "key": key}, indent=2), "utf-8")

    for rater in ("rater1", "rater2"):
        with (AUDIT_DIR / f"ratings_{rater}.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["sample_id"] + DIMENSIONS)
            for row in key:
                w.writerow([row["sample_id"]] + [""] * len(DIMENSIONS))

    (AUDIT_DIR / "INSTRUCTIONS.md").write_text(
        "# Blinded human audit — instructions\n\n"
        f"There are {len(everything)} images in `images/`. Score every one on three\n"
        "dimensions, 1-5 integers, in your own `ratings_<rater>.csv`.\n\n"
        + "".join(f"- **{d}** — {SCALE[d]}\n" for d in DIMENSIONS) +
        "\nRules that make the result usable:\n\n"
        "- Rate independently. Do not look at the other rater's sheet, and do not\n"
        "  discuss scores until both sheets are finished — agreement between two people\n"
        "  who conferred measures the conversation, not the images.\n"
        "- Do not open `KEY_do_not_open_until_scored.json`. It maps samples to models,\n"
        "  including which are yours.\n"
        "- Score every row. Blanks are dropped, and dropping rows non-randomly biases\n"
        "  the agreement estimate.\n"
        "- Use the full scale. If everything gets a 4, kappa is undefined no matter how\n"
        "  carefully you looked.\n\n"
        "Then run `python scripts/human_audit.py score`.\n", "utf-8")

    print(f"\n{len(entries)} scored samples + {len(controls)} hidden controls "
          f"-> {AUDIT_DIR.relative_to(REPO_ROOT)}")
    print("give each rater their own ratings_*.csv; do not share the key file")


def cohens_kappa(a: list[int], b: list[int], weighted: bool = True) -> float:
    """Cohen's kappa, quadratic-weighted by default.

    Unweighted kappa treats a 1-vs-5 disagreement exactly like 4-vs-5, which is wrong
    for an ordinal 1-5 scale: on a scale where the categories are ordered, how far apart
    two raters are is the whole question. Quadratic weights are the standard choice for
    Likert data and are what is reported here; the unweighted value is reported next to
    it because that is what "Cohen's kappa" usually means unqualified.

        kappa = 1 - sum(w*O) / sum(w*E)

    with O the observed joint distribution and E the outer product of the marginals.
    """
    cats = sorted(set(a) | set(b))
    idx = {c: i for i, c in enumerate(cats)}
    k, n = len(cats), len(a)
    if k == 1:
        return float("nan")     # no variance: kappa is undefined, not 1.0

    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1 / n

    ra = [sum(obs[i]) for i in range(k)]
    cb = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((cats[i] - cats[j]) / (cats[-1] - cats[0])) ** 2 if weighted else float(i != j)
            num += w * obs[i][j]
            den += w * ra[i] * cb[j]
    return float("nan") if den == 0 else 1 - num / den


def score() -> None:
    key_path = AUDIT_DIR / "KEY_do_not_open_until_scored.json"
    if not key_path.exists():
        raise SystemExit(f"no audit found at {AUDIT_DIR}; run `build` first")
    meta = json.loads(key_path.read_text("utf-8"))
    key = {r["sample_id"]: r for r in meta["key"]}

    ratings = {}
    for rater in ("rater1", "rater2"):
        p = AUDIT_DIR / f"ratings_{rater}.csv"
        if not p.exists():
            raise SystemExit(f"missing {p}")
        with p.open(newline="", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        parsed = {}
        for r in rows:
            vals = {}
            for d in DIMENSIONS:
                v = (r.get(d) or "").strip()
                if v:
                    vals[d] = int(v)
            if len(vals) == len(DIMENSIONS):
                parsed[r["sample_id"]] = vals
        ratings[rater] = parsed
        print(f"{rater}: {len(parsed)}/{len(rows)} rows complete")

    both = sorted(set(ratings["rater1"]) & set(ratings["rater2"]))
    if not both:
        raise SystemExit("no sample scored by both raters")

    report = {"n_rated_by_both": len(both), "dimensions": {}, "by_model": {}, "controls": {}}

    for d in DIMENSIONS:
        a = [ratings["rater1"][s][d] for s in both]
        b = [ratings["rater2"][s][d] for s in both]
        exact = sum(x == y for x, y in zip(a, b)) / len(a)
        within1 = sum(abs(x - y) <= 1 for x, y in zip(a, b)) / len(a)
        report["dimensions"][d] = {
            "mean_rater1": round(sum(a) / len(a), 3),
            "mean_rater2": round(sum(b) / len(b), 3),
            "percent_exact_agreement": round(100 * exact, 1),
            "percent_within_1_point": round(100 * within1, 1),
            "cohens_kappa_unweighted": round(cohens_kappa(a, b, weighted=False), 4),
            "cohens_kappa_quadratic": round(cohens_kappa(a, b, weighted=True), 4),
        }

    scored = [s for s in both if not key[s]["is_control"]]
    controls = [s for s in both if key[s]["is_control"]]
    for model in sorted({key[s]["model"] for s in scored}):
        ids = [s for s in scored if key[s]["model"] == model]
        report["by_model"][model] = {
            "n": len(ids),
            **{d: round(sum(ratings[r][s][d] for s in ids for r in ratings)
                        / (2 * len(ids)), 3) for d in DIMENSIONS},
        }
    if controls:
        report["controls"] = {
            "n": len(controls),
            **{d: round(sum(ratings[r][s][d] for s in controls for r in ratings)
                        / (2 * len(controls)), 3) for d in DIMENSIONS},
        }
        gen_style = sum(v["style"] for k, v in report["by_model"].items()) / max(1, len(report["by_model"]))
        report["controls"]["style_gap_vs_generated"] = round(
            report["controls"]["style"] - gen_style, 3)
        report["controls"]["sanity_check_passed"] = bool(
            report["controls"]["style"] > gen_style)

    out = AUDIT_DIR / "audit_results.json"
    out.write_text(json.dumps(report, indent=2), "utf-8")
    print(json.dumps(report, indent=2))
    print(f"\n-> {out.relative_to(REPO_ROOT)}")
    if controls and not report["controls"]["sanity_check_passed"]:
        print("\nWARNING: the real-Monet controls did not out-score the generated "
              "images on style. Treat the agreement numbers as uninformative and "
              "re-run the audit rather than reporting them.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="assemble the blinded sample set and rating sheets")
    b.add_argument("--source", action="append", required=True, metavar="NAME=DIR",
                   help="a member's pred_A2B directory; repeat for each member")
    b.add_argument("--monet-dir", default="task3_gan/data/monet_jpg")
    b.add_argument("--seed", type=int, default=9015)

    sub.add_parser("score", help="compute agreement once both sheets are filled in")

    args = ap.parse_args()
    if args.cmd == "build":
        sources = {}
        for s in args.source:
            if "=" not in s:
                raise SystemExit(f"--source must look like NAME=DIR, got {s!r}")
            name, d = s.split("=", 1)
            p = Path(d)
            sources[name] = p if p.is_absolute() else REPO_ROOT / p
        md = Path(args.monet_dir)
        build(sources, args.seed, md if md.is_absolute() else REPO_ROOT / md)
    else:
        score()


if __name__ == "__main__":
    main()
