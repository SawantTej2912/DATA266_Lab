"""Every metric Task 1 asks for, in one place.

This module is deliberately free of training-loop state so the team can agree on one
scoring implementation and each member can call it against their own checkpoint.
"""

from __future__ import annotations

import math
import re
from collections import Counter

import torch


# --------------------------------------------------------------------------- losses

@torch.no_grad()
def evaluate_loss_and_accuracy(model, loader, device, max_batches: int | None = None) -> dict:
    """Mean cross-entropy and top-1 next-character accuracy over a loader.

    Both are averaged over *characters*, not over batches, so a short final batch
    cannot pull the average around.
    """
    model.eval()
    total_loss, total_correct, total_tokens = 0.0, 0, 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        n = y.numel()
        total_loss += loss.item() * n
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_tokens += n

    ce = total_loss / total_tokens
    return {
        "cross_entropy": ce,
        "perplexity": math.exp(ce),
        "bits_per_character": ce / math.log(2),
        "top1_accuracy": total_correct / total_tokens,
        "n_tokens": total_tokens,
    }


def generalization_gap(train_ce: float, val_ce: float) -> float:
    return val_ce - train_ce


# ------------------------------------------------------------------ generation stats

def _ngrams(seq, n: int):
    return [tuple(seq[i : i + n]) for i in range(len(seq) - n + 1)]


def distinct_n(text: str, n: int, unit: str = "word") -> float:
    """Ratio of unique n-grams to total n-grams.

    Reported at word level by default. At character level distinct-1 is bounded by the
    ~90-symbol vocabulary and saturates near 1.0 for any text of reasonable length, so
    it says nothing about whether the model is repeating itself; word level does.
    """
    seq = text.split() if unit == "word" else list(text)
    grams = _ngrams(seq, n)
    if not grams:
        return 0.0
    return len(set(grams)) / len(grams)


def repeated_ngram_rate(text: str, n: int = 4, unit: str = "word") -> float:
    """Fraction of n-gram occurrences that are not the first occurrence of that n-gram.

    0.0 means every n-gram is unique; values climbing past ~0.3 are the signature of a
    model stuck in a loop.
    """
    seq = text.split() if unit == "word" else list(text)
    grams = _ngrams(seq, n)
    if not grams:
        return 0.0
    counts = Counter(grams)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / len(grams)


_WORD_RE = re.compile(r"[a-z']+")


def generation_report(text: str) -> dict:
    return {
        "distinct_1_word": distinct_n(text, 1, "word"),
        "distinct_2_word": distinct_n(text, 2, "word"),
        "distinct_3_word": distinct_n(text, 3, "word"),
        "distinct_1_char": distinct_n(text, 1, "char"),
        "distinct_2_char": distinct_n(text, 2, "char"),
        "distinct_3_char": distinct_n(text, 3, "char"),
        "repeated_4gram_rate_word": repeated_ngram_rate(text, 4, "word"),
        "repeated_4gram_rate_char": repeated_ngram_rate(text, 4, "char"),
        "n_chars": len(text),
        "n_words": len(text.split()),
    }


def train_vocab_word_rate(text: str, vocabulary: set[str]) -> float:
    """Share of generated words that occur in the *training corpus* vocabulary.

    A character-level model can emit fluent-looking non-words; this puts a number on how
    often that happens, which the standard diversity metrics do not capture.

    Named for what it measures. This was called `real_word_rate` through the MPS runs,
    which overstated it: the reference set is the 20,000 TinyStories training documents,
    not a dictionary, so a genuine English word absent from that corpus counts against
    the model and a typo present in it counts for it. TinyStories is deliberately
    written in simple vocabulary, so the two are close in practice -- but "real word" is
    a claim about English and this is a claim about the training data. Pre-rename CSVs
    carry the old column name and are labelled as historical MPS results.
    """
    words = _WORD_RE.findall(text.lower())
    if not words:
        return 0.0
    return sum(w in vocabulary for w in words) / len(words)


# --------------------------------------------------------------------- stability

def stability_report(step_losses: list[float], grad_norms: list[float], spike_factor: float = 1.5) -> dict:
    """Loss spikes, NaN/inf count and gradient-norm summary.

    A 'spike' is a step whose loss exceeds `spike_factor` times the trailing mean of
    the previous 50 steps, which catches sudden divergence without flagging ordinary
    batch-to-batch noise early in training.
    """
    n_nan = sum(1 for v in step_losses + grad_norms if not math.isfinite(v))
    finite_losses = [v for v in step_losses if math.isfinite(v)]

    spikes = 0
    for i in range(50, len(finite_losses)):
        window = finite_losses[i - 50 : i]
        if finite_losses[i] > spike_factor * (sum(window) / len(window)):
            spikes += 1

    finite_norms = [v for v in grad_norms if math.isfinite(v)]
    return {
        "nan_or_inf_count": n_nan,
        "loss_spike_count": spikes,
        "grad_norm_mean": sum(finite_norms) / len(finite_norms) if finite_norms else float("nan"),
        "grad_norm_max": max(finite_norms) if finite_norms else float("nan"),
        "grad_norm_final": finite_norms[-1] if finite_norms else float("nan"),
    }


# ------------------------------------------------------------------------ hardware

def peak_memory_mb(device: torch.device) -> float:
    """Peak device memory in MB, using whichever accounting the backend offers."""
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 1024**2
    if device.type == "mps":
        # MPS has no peak counter, only a live one; the training loop samples this
        # each step and keeps the maximum.
        return torch.mps.current_allocated_memory() / 1024**2
    import resource

    # macOS reports ru_maxrss in bytes, Linux in kilobytes.
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return rss / 1024**2 if sys.platform == "darwin" else rss / 1024
