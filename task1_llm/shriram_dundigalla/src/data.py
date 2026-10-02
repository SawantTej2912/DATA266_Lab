"""Character-level preprocessing for TinyStories.

The train/validation split is made over *stories*, not over the concatenated character
stream. Splitting a single stream at one offset leaks the text either side of the cut
into both halves' sliding windows; splitting by story means no validation character is
ever part of a training window.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class CharTokenizer:
    """Integer encoding built from the training text only.

    Building the vocabulary from the training split alone is deliberate: if a character
    only ever appears in validation, the model has genuinely never seen it, and folding
    it into the vocabulary would hide that. Unknown characters map to a reserved <unk>.
    """

    UNK = "\ufffd"

    def __init__(self, chars: list[str]):
        vocab = [self.UNK] + [c for c in sorted(set(chars)) if c != self.UNK]
        self.idx_to_char = {i: c for i, c in enumerate(vocab)}
        self.char_to_idx = {c: i for i, c in self.idx_to_char.items()}

    @classmethod
    def from_text(cls, text: str) -> "CharTokenizer":
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.char_to_idx)

    def encode(self, s: str) -> list[int]:
        unk = self.char_to_idx[self.UNK]
        return [self.char_to_idx.get(ch, unk) for ch in s]

    def decode(self, ids) -> str:
        return "".join(self.idx_to_char[int(i)] for i in ids)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"idx_to_char": self.idx_to_char}, ensure_ascii=False), "utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        idx_to_char = json.loads(path.read_text("utf-8"))["idx_to_char"]
        tok = cls.__new__(cls)
        tok.idx_to_char = {int(k): v for k, v in idx_to_char.items()}
        tok.char_to_idx = {v: k for k, v in tok.idx_to_char.items()}
        return tok


class WindowDataset(Dataset):
    """Non-overlapping length-`block_size` windows over a token stream.

    Non-overlapping rather than stride-1: with stride 1 the "100K sequences" of the
    brief would be 100K near-duplicates of ~100K characters of text, so ten epochs
    would really be ~1000 passes over a very small corpus. Disjoint windows make one
    epoch one genuine pass over 100K * block_size characters.

    Item t is (tokens[t : t+block_size], tokens[t+1 : t+block_size+1]) — the target is
    the input shifted left by one, giving block_size next-character problems per row.
    """

    def __init__(self, tokens: np.ndarray, block_size: int, n_windows: int | None = None):
        self.tokens = torch.from_numpy(tokens.astype(np.int64))
        self.block_size = block_size

        available = (len(self.tokens) - 1) // block_size
        if n_windows is None:
            self.n_windows = available
        elif n_windows > available:
            raise ValueError(
                f"asked for {n_windows} windows of {block_size} chars but the stream only "
                f"supports {available}; fetch more stories"
            )
        else:
            self.n_windows = n_windows

    def __len__(self) -> int:
        return self.n_windows

    def __getitem__(self, i: int):
        start = i * self.block_size
        chunk = self.tokens[start : start + self.block_size + 1]
        return chunk[:-1], chunk[1:]


@dataclass
class Task1Data:
    tokenizer: CharTokenizer
    train_ds: WindowDataset
    val_ds: WindowDataset
    train_text: str
    val_text: str
    n_train_stories: int
    n_val_stories: int


def load_stories(jsonl_path: Path) -> list[str]:
    with Path(jsonl_path).open(encoding="utf-8") as fh:
        return [json.loads(line)["text"] for line in fh]


def build(
    jsonl_path: Path,
    block_size: int,
    n_train_windows: int,
    n_val_windows: int,
    val_story_frac: float = 0.1,
    seed: int = 9015,
) -> Task1Data:
    stories = load_stories(jsonl_path)

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(stories))
    n_val_stories = max(1, int(len(stories) * val_story_frac))
    val_idx, train_idx = order[:n_val_stories], order[n_val_stories:]

    # A blank line between stories gives the model an explicit document boundary to
    # learn, instead of one story running straight into the next mid-sentence.
    sep = "\n\n"
    train_text = sep.join(stories[i] for i in train_idx)
    val_text = sep.join(stories[i] for i in val_idx)

    tokenizer = CharTokenizer.from_text(train_text)
    train_tokens = np.array(tokenizer.encode(train_text), dtype=np.int32)
    val_tokens = np.array(tokenizer.encode(val_text), dtype=np.int32)

    return Task1Data(
        tokenizer=tokenizer,
        train_ds=WindowDataset(train_tokens, block_size, n_train_windows),
        val_ds=WindowDataset(val_tokens, block_size, n_val_windows),
        train_text=train_text,
        val_text=val_text,
        n_train_stories=len(train_idx),
        n_val_stories=len(val_idx),
    )
