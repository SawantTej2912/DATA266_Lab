"""Build the two zips this lab has to hand in, which have different shapes.

    python scripts/package_submission.py --canvas     # the graded submission zip
    python scripts/package_submission.py --datasets   # the Google Drive data bundle
    python scripts/package_submission.py --check      # verify without writing anything

The Canvas instructions ask for `Part 1/ Part 2/ Part 3/` plus a combined `Report.pdf`,
while the brief's repository diagram uses `task1_llm/ task2_sentiment/ task3_gan/`. Those
are two different layouts for the same work and **both are required** — the repo keeps
the brief's layout and this script projects it into the Canvas layout, so neither has to
be maintained by hand.

Datasets are excluded from the Canvas zip and from git (per the announcement of 28 Sep);
`--datasets` builds the separate Drive bundle for them.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
TEAM = "Team_33"

# repo directory -> Canvas directory
PART_MAP = {
    "task1_llm": "Part 1",
    "task2_sentiment": "Part 2",
    "task3_gan": "Part 3",
}

# Carried into the Canvas zip alongside the three parts.
TOP_LEVEL = ["README.md", "TEAM_PROTOCOL.md", "requirements.txt"]

# Excluded everywhere: caches, datasets, and the oversized Part 3 checkpoints, which are
# 91-340 MB each and belong in the Drive bundle.
EXCLUDE_DIRS = {"__pycache__", ".ipynb_checkpoints", "data", "data_processed",
                "monet_jpg", "photo_jpg", "pred_A2B", "pred_B2A"}
EXCLUDE_SUFFIX = {".pyc", ".pyo"}
MAX_FILE_MB = 95.0   # GitHub rejects >100 MB; stay clear of the boundary


def wanted(path: Path) -> bool:
    if any(part in EXCLUDE_DIRS for part in path.parts):
        return False
    if path.suffix in EXCLUDE_SUFFIX or path.name == ".DS_Store":
        return False
    if path.suffix == ".pt" and path.stat().st_size > MAX_FILE_MB * 1024**2:
        return False
    return True


def collect() -> list[tuple[Path, str]]:
    """Return (source path, path inside the zip) for everything to include."""
    items: list[tuple[Path, str]] = []

    for repo_dir, canvas_dir in PART_MAP.items():
        root = REPO_ROOT / repo_dir
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if p.is_file() and wanted(p):
                items.append((p, f"{canvas_dir}/{p.relative_to(root)}"))

    for name in TOP_LEVEL:
        p = REPO_ROOT / name
        if p.exists():
            items.append((p, name))

    for p in sorted((REPO_ROOT / "reproducibility").rglob("*")):
        if p.is_file() and wanted(p):
            items.append((p, f"reproducibility/{p.relative_to(REPO_ROOT / 'reproducibility')}"))

    for p in sorted((REPO_ROOT / "scripts").rglob("*")):
        if p.is_file() and wanted(p):
            items.append((p, f"scripts/{p.relative_to(REPO_ROOT / 'scripts')}"))

    report = REPO_ROOT / f"report/DATA266_Lab1_Report_{TEAM}.pdf"
    if report.exists():
        items.append((report, "Report.pdf"))

    return items


def required_report_check(items: list[tuple[Path, str]]) -> list[str]:
    """Things the graders will look for. Returns a list of problems, empty if clean."""
    inside = {dst for _, dst in items}
    problems = []

    if "Report.pdf" not in inside:
        problems.append("Report.pdf is missing — the combined report is a graded item "
                        "and must sit at the top level of the zip")

    for canvas_dir in PART_MAP.values():
        if not any(d.startswith(canvas_dir + "/") for d in inside):
            problems.append(f"'{canvas_dir}/' is empty")

    # Member folder names are discovered rather than hard-coded. The two of us named
    # ours differently -- shriram_dundigalla and tejas -- and a fixed list silently
    # reported the other member's work as missing when it was simply under another name.
    members = sorted({d.split("/")[1] for d in inside
                      if d.startswith(tuple(c + "/" for c in PART_MAP.values()))
                      and len(d.split("/")) > 2
                      and d.split("/")[1] not in NON_MEMBER_DIRS})
    for repo_dir, canvas_dir in PART_MAP.items():
        for member in members:
            prefix = f"{canvas_dir}/{member}/"
            has_results = any(d.startswith(prefix) and d.endswith("results.md")
                              for d in inside)
            has_code = any(d.startswith(prefix + "src/") for d in inside)
            if not has_code:
                problems.append(f"{prefix} has no src/ — half of each part's marks are "
                                f"team marks and need both members' work present")
            elif not has_results:
                problems.append(f"{prefix} has code but no results.md")

    if not any(d.startswith("reproducibility/raw_logs/") for d in inside):
        problems.append("no raw training logs — Section 5 grades these as the evidence trail")

    problems += scan_for_secrets_and_paths()
    return problems


# Section 5, verbatim: "No hard-coded personal file paths or secrets anywhere in the
# repo; config-driven runs only." Both halves are easy to violate by accident and
# invisible in review -- a personal path once reached a committed notebook here as
# *cell output*, from a print statement, in a notebook whose own prose claimed there
# were none. Raw logs are excluded: they are the untouched evidence trail and Section 5
# separately forbids cleaning them up.
SECRET_PATTERNS = [
    (r"/Users/[a-z0-9_.-]+/", "absolute macOS home path"),
    (r"/home/[a-z0-9_.-]+/", "absolute Linux home path"),
    (r"C:\\\\Users\\\\", "absolute Windows home path"),
    (r"KGAT_[A-Za-z0-9]+", "Kaggle API token"),
    (r"\"key\"\s*:\s*\"[A-Za-z0-9]{20,}\"", "API key in JSON"),
    (r"(?i)(secret|password|passwd)\s*[:=]\s*[\"'][^\"'$]{6,}", "hard-coded credential"),
]


# The three scanners hold these patterns as *detection rules*, so scanning them finds
# the rules rather than a leak. Exempted by filename rather than by pattern, because a
# pattern broad enough to tell a rule from a leak would also excuse a real one.
# Directories that live beside the member folders under a task but are not members.
NON_MEMBER_DIRS = {"data", "human_audit"}

SCANNER_EXEMPT = {"scripts/package_submission.py", "scripts/verify_submission.py",
                  }


def scan_for_secrets_and_paths() -> list[str]:
    import re
    import subprocess

    tracked = subprocess.run(["git", "ls-files"], cwd=REPO_ROOT,
                             capture_output=True, text=True).stdout.split()
    problems = []
    for rel in tracked:
        if rel.startswith("reproducibility/raw_logs/") or rel in SCANNER_EXEMPT:
            continue
        p = REPO_ROOT / rel
        if not p.is_file() or p.stat().st_size > 20_000_000:
            continue
        try:
            text = p.read_text("utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern, label in SECRET_PATTERNS:
            m = re.search(pattern, text)
            if m:
                line = text[:m.start()].count("\n") + 1
                problems.append(f"{rel}:{line} contains a {label} "
                                f"— Section 5 forbids this in any committed file")
                break
    return problems


def build_zip(name: str, items: list[tuple[Path, str]]) -> Path:
    out = REPO_ROOT / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, dst in items:
            z.write(src, dst)
    return out


def build_datasets() -> Path:
    """The Drive bundle the announcement of 28 Sep asks for: datasets only.

    Deliberately excludes the Part 3 checkpoint. It is 1.3 GB on its own — most of that
    Adam state needed only to *resume* training, not to reproduce a result — and it
    would more than triple a bundle whose job is to restore the datasets.
    `--checkpoints` builds that separately, which matters more for Part 3 than for the
    others: a 217.6 MB fp32 generator cannot go in git at all, so the zip is the only
    way the Part 3 weights travel.
    """
    groups = {
        "task1_llm/data": REPO_ROOT / "task1_llm/data",
        "task2_sentiment/data": REPO_ROOT / "task2_sentiment/data",
        "task3_gan/data": REPO_ROOT / "task3_gan/data",
    }
    out = REPO_ROOT / f"Lab1_datasets_{TEAM}.zip"
    n = 0
    # compresslevel=1: JPEGs and .npz are already compressed, so a higher level costs
    # minutes of CPU for a fraction of a percent.
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for prefix, root in groups.items():
            if not root.exists():
                print(f"  (skipping {prefix} — not present)")
                continue
            for p in sorted(root.rglob("*")):
                if p.is_file() and p.name != ".DS_Store":
                    z.write(p, f"{prefix}/{p.relative_to(root)}")
                    n += 1
    print(f"  {n} dataset files")
    return out


def build_checkpoints() -> Path:
    """Only what is needed to reproduce a reported number, not to resume training.

    For Part 3 that means whatever checkpoint the reported score came from, which for
    the notebook-trained U-Net is a single `ckpt_epoch*.pth`.
    """
    out = REPO_ROOT / f"Lab1_checkpoints_{TEAM}.zip"
    n = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        for task in ("task1_llm", "task2_sentiment"):
            for p in sorted((REPO_ROOT / task).rglob("checkpoints/*.pt")):
                z.write(p, f"{task}/{p.name}")
                n += 1
        # Part 3 is written as the two generators rather than as the raw checkpoint.
        # The file on disk is 1.3 GB, roughly two thirds of it Adam moments for four
        # networks, which this bundle exists specifically not to carry. The generators
        # are kept in fp32: casting to fp16 would halve them and stop reproducing the
        # reported FID, which would make the bundle disagree with metrics_report.csv.
        import torch
        ckpt_dir = REPO_ROOT / "task3_gan/shriram_dundigalla/checkpoints"
        for ck_path in sorted(ckpt_dir.glob("*.pt")) + sorted(ckpt_dir.glob("*.pth")):
            ck = torch.load(ck_path, map_location="cpu", weights_only=False)
            stem = ck_path.stem
            for key in ("G_M2P", "G_P2M", "G_AB", "G_BA"):
                if key not in ck:
                    continue
                tmp = REPO_ROOT / f".{stem}_{key}.pt"
                torch.save(ck[key], tmp)
                z.write(tmp, f"task3_gan/{stem}_{key}.pt")
                tmp.unlink()
                n += 1
    print(f"  {n} checkpoints")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canvas", action="store_true", help="build the graded submission zip")
    ap.add_argument("--datasets", action="store_true", help="build the Drive data bundle")
    ap.add_argument("--checkpoints", action="store_true",
                    help="build a separate bundle of the weights behind each result")
    ap.add_argument("--check", action="store_true", help="report problems, write nothing")
    args = ap.parse_args()
    if not (args.canvas or args.datasets or args.checkpoints or args.check):
        ap.error("pass at least one of --canvas, --datasets, --checkpoints, --check")

    items = collect()
    problems = required_report_check(items)

    print(f"{len(items)} files would go into the Canvas zip")
    total = sum(src.stat().st_size for src, _ in items)
    print(f"uncompressed size: {total / 1024**2:.1f} MB\n")

    if problems:
        print("NOT READY TO SUBMIT:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("all structural checks passed")
    print()

    if args.canvas:
        if problems:
            print("building anyway so the structure can be inspected — do not submit this")
        out = build_zip(f"DATA266_Lab1_{TEAM}.zip", items)
        print(f"wrote {out.name} ({out.stat().st_size / 1024**2:.1f} MB)")

    if args.datasets:
        print("building the Drive dataset bundle (this takes a minute)...")
        out = build_datasets()
        print(f"wrote {out.name} ({out.stat().st_size / 1024**2:.1f} MB)")
        print("upload it to Google Drive, enable read access, and put the link in "
              "README.md where <DRIVE_LINK_PENDING> is")

    if args.checkpoints:
        out = build_checkpoints()
        print(f"wrote {out.name} ({out.stat().st_size / 1024**2:.1f} MB)")

    raise SystemExit(1 if problems and args.check else 0)


if __name__ == "__main__":
    main()
