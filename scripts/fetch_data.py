"""Fetch the shared raw datasets for Task 1 (TinyStories) and Task 2 (Yelp Polarity).

Writes into task1_llm/data/ and task2_sentiment/data/, both relative to the repo root,
so the script works from any checkout with no personal paths baked in.
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

REPO_ROOT = Path(__file__).resolve().parents[1]


def fetch_tinystories(n_stories: int) -> None:
    """Stream the first n_stories rows of TinyStories into a JSONL file.

    Streaming avoids pulling the full ~2 GB corpus when the lab only needs a slice
    large enough to build 110K sliding windows.
    """
    out_dir = REPO_ROOT / "task1_llm" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"tinystories_train_first{n_stories}.jsonl"

    stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    n_chars = 0
    with out_path.open("w", encoding="utf-8") as fh:
        for i, row in enumerate(stream):
            if i >= n_stories:
                break
            text = row["text"].strip()
            n_chars += len(text)
            fh.write(json.dumps({"text": text}) + "\n")

    print(f"tinystories -> {out_path.relative_to(REPO_ROOT)}")
    print(f"  stories: {n_stories}  characters: {n_chars}")


def fetch_yelp() -> None:
    """Download Yelp Polarity and write train/test to parquet."""
    out_dir = REPO_ROOT / "task2_sentiment" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("fancyzhx/yelp_polarity")
    for split in ("train", "test"):
        out_path = out_dir / f"yelp_polarity_{split}.parquet"
        ds[split].to_parquet(out_path)
        print(f"yelp {split} -> {out_path.relative_to(REPO_ROOT)}  rows: {len(ds[split])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["tinystories", "yelp", "all"], default="all")
    parser.add_argument("--n-stories", type=int, default=8000)
    args = parser.parse_args()

    if args.task in ("tinystories", "all"):
        fetch_tinystories(args.n_stories)
    if args.task in ("yelp", "all"):
        fetch_yelp()
