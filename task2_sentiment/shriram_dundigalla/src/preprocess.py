"""Task 2 preprocessing: clean, tokenise, build a from-scratch vocabulary, tensorise.

No pretrained embeddings and no pretrained language model are used anywhere. The
vocabulary is built from this run's training split only, and every embedding is a
randomly initialised `nn.Embedding` learned during training.

Slice features for the robustness metrics are computed from the RAW review text before
cleaning, because the cleaning step destroys exactly the signals the slices are about
(capitalisation, punctuation).
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

PAD, UNK = "<pad>", "<unk>"
PAD_ID, UNK_ID = 0, 1

# Negations, plus the contraction suffixes that expand into them. Anything that flips
# polarity stays in the text even though NLTK lists most of these as stopwords.
_NEGATIONS = {
    "not", "no", "nor", "never", "none", "nothing", "nowhere", "neither",
    "cannot", "cant", "wont", "dont", "didnt", "doesnt", "isnt", "wasnt",
    "arent", "werent", "hasnt", "havent", "hadnt", "couldnt", "wouldnt",
    "shouldnt", "aint", "without", "hardly", "barely", "scarcely",
}

_CONTRACTION_NT = re.compile(r"n['\u2019]t\b")
_NON_ALPHA = re.compile(r"[^a-z\s]+")
_MULTISPACE = re.compile(r"\s+")


def build_stopwords(keep_negations: bool) -> frozenset[str]:
    sw = {w.replace("'", "") for w in stopwords.words("english")}
    if keep_negations:
        sw -= _NEGATIONS
    return frozenset(sw)


@lru_cache(maxsize=400_000)
def _stem(word: str, stemmer_name: str) -> str:
    """Cached stemming. The Porter stemmer is pure Python and slow per call, but the
    set of distinct words is small relative to the number of tokens, so caching turns
    ~15M calls into ~200K."""
    if stemmer_name == "none":
        return word
    return _PORTER.stem(word)


_PORTER = PorterStemmer()


def clean(text: str, sw: frozenset[str], stemmer_name: str) -> list[str]:
    """Lowercase, expand n't, strip non-letters, drop stopwords, stem.

    The n't expansion happens *before* punctuation stripping: otherwise "didn't"
    becomes "didnt" only by luck of the apostrophe, and "don t" would split the
    negation off into a meaningless single letter.
    """
    t = text.lower()
    t = _CONTRACTION_NT.sub(" not", t)
    t = _NON_ALPHA.sub(" ", t)
    tokens = _MULTISPACE.sub(" ", t).strip().split()
    out = []
    for w in tokens:
        if len(w) < 2 and w not in {"a", "i"}:
            continue
        if w in sw:
            continue
        out.append(_stem(w, stemmer_name))
    return out


# ----------------------------------------------------------------- slice features

_EXCLAIM = re.compile(r"!")
_UPPER_WORD = re.compile(r"\b[A-Z]{3,}\b")


def slice_features(raw_text: str, n_tokens: int) -> dict:
    """Per-review attributes used for the robustness (per-slice) metrics."""
    n_chars = len(raw_text)
    caps_words = len(_UPPER_WORD.findall(raw_text))
    words = max(1, len(raw_text.split()))
    lowered = raw_text.lower()
    lowered = _CONTRACTION_NT.sub(" not", lowered)
    tokens = set(_NON_ALPHA.sub(" ", lowered).split())
    return {
        "n_chars": n_chars,
        "n_tokens": n_tokens,
        "has_negation": bool(tokens & _NEGATIONS),
        "shouty": caps_words / words > 0.05,
        "exclamations": len(_EXCLAIM.findall(raw_text)),
    }


def length_bucket(n_tokens: int, q33: float, q66: float) -> str:
    if n_tokens <= q33:
        return "short"
    if n_tokens <= q66:
        return "medium"
    return "long"


# --------------------------------------------------------------------- vocabulary

class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}

    @classmethod
    def from_docs(cls, docs, vocab_size: int, min_freq: int) -> "Vocab":
        counts = Counter()
        for toks in docs:
            counts.update(toks)
        kept = [w for w, c in counts.most_common() if c >= min_freq][: vocab_size - 2]
        return cls([PAD, UNK] + kept)

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens, max_len: int) -> np.ndarray:
        ids = np.full(max_len, PAD_ID, dtype=np.int32)
        for i, tok in enumerate(tokens[:max_len]):
            ids[i] = self.stoi.get(tok, UNK_ID)
        return ids

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"itos": self.itos}), "utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text("utf-8"))["itos"])


# ------------------------------------------------------------------------ bundle

@dataclass
class Split:
    ids: np.ndarray            # (N, max_len) int32 token ids
    lengths: np.ndarray        # (N,) true token count before padding
    labels: np.ndarray         # (N,) int64
    raw: list[str]             # original review text, for the manual error review
    slices: pd.DataFrame       # per-review slice attributes


@dataclass
class Task2Data:
    vocab: Vocab
    train: Split
    val: Split
    test: Split
    stats: dict


def _make_split(df, vocab, max_len, sw, stemmer_name, q33=None, q66=None):
    docs = [clean(t, sw, stemmer_name) for t in df["text"]]
    lengths = np.array([len(d) for d in docs], dtype=np.int32)

    if q33 is None:
        q33, q66 = np.quantile(lengths, [0.33, 0.66])

    ids = np.stack([vocab.encode(d, max_len) for d in docs]) if vocab else None
    feats = [slice_features(t, int(n)) for t, n in zip(df["text"], lengths)]
    sl = pd.DataFrame(feats)
    sl["length_bucket"] = [length_bucket(int(n), q33, q66) for n in lengths]

    return docs, lengths, ids, sl, (q33, q66)


def build(cfg: dict, repo_root: Path, log=print) -> Task2Data:
    dcfg, pcfg = cfg["data"], cfg["preprocess"]
    rng = np.random.default_rng(cfg["seed"])
    sw = build_stopwords(pcfg["keep_negations"])
    stemmer_name = pcfg["stemmer"]

    train_df = pd.read_parquet(repo_root / dcfg["train_parquet"])
    test_df = pd.read_parquet(repo_root / dcfg["test_parquet"])

    # ------------------------------------------------- malformed / missing entries
    before = len(train_df), len(test_df)
    def scrub(df):
        df = df.dropna(subset=["text", "label"])
        df = df[df["text"].str.strip().str.len() > 0]
        return df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    train_df, test_df = scrub(train_df), scrub(test_df)
    log(f"scrub: train {before[0]} -> {len(train_df)}, test {before[1]} -> {len(test_df)}")

    # ------------------------------------------------------- balanced subsampling
    n_train, n_val = dcfg["n_train"], dcfg["n_val"]
    per_class = (n_train + n_val) // 2
    picks = []
    for label in (0, 1):
        pool = train_df.index[train_df["label"] == label].to_numpy()
        picks.append(rng.choice(pool, size=per_class, replace=False))
    sel = rng.permutation(np.concatenate(picks))
    sub = train_df.loc[sel].reset_index(drop=True)
    tr_df, va_df = sub.iloc[:n_train].reset_index(drop=True), sub.iloc[n_train:].reset_index(drop=True)

    if not dcfg["use_full_test"]:
        test_df = test_df.sample(n=10000, random_state=cfg["seed"]).reset_index(drop=True)

    # --------------------------------------------- vocabulary from TRAINING only
    tr_docs = [clean(t, sw, stemmer_name) for t in tr_df["text"]]
    tr_lengths = np.array([len(d) for d in tr_docs], dtype=np.int32)
    vocab = Vocab.from_docs(tr_docs, dcfg["vocab_size"], dcfg["min_freq"])
    log(f"vocab size {len(vocab)} (cap {dcfg['vocab_size']}, min_freq {dcfg['min_freq']})")

    q33, q66 = np.quantile(tr_lengths, [0.33, 0.66])
    max_len = dcfg["max_len"]

    def finish(df, docs=None, lengths=None):
        if docs is None:
            docs = [clean(t, sw, stemmer_name) for t in df["text"]]
            lengths = np.array([len(d) for d in docs], dtype=np.int32)
        ids = np.stack([vocab.encode(d, max_len) for d in docs])
        sl = pd.DataFrame([slice_features(t, int(n)) for t, n in zip(df["text"], lengths)])
        sl["length_bucket"] = [length_bucket(int(n), q33, q66) for n in lengths]
        return Split(
            ids=ids,
            lengths=np.minimum(lengths, max_len),
            labels=df["label"].to_numpy(dtype=np.int64),
            raw=df["text"].tolist(),
            slices=sl,
        )

    train = finish(tr_df, tr_docs, tr_lengths)
    val = finish(va_df)
    test = finish(test_df)

    oov = float((test.ids == UNK_ID).sum() / max(1, (test.ids != PAD_ID).sum()))
    truncated = float((np.array([len(clean(t, sw, stemmer_name)) for t in test_df["text"][:2000]]) > max_len).mean())

    stats = {
        "n_train": len(train.labels),
        "n_val": len(val.labels),
        "n_test": len(test.labels),
        "vocab_size": len(vocab),
        "max_len": max_len,
        "train_class_counts": {int(k): int(v) for k, v in zip(*np.unique(train.labels, return_counts=True))},
        "test_class_counts": {int(k): int(v) for k, v in zip(*np.unique(test.labels, return_counts=True))},
        "token_length_mean": float(tr_lengths.mean()),
        "token_length_median": float(np.median(tr_lengths)),
        "token_length_p90": float(np.quantile(tr_lengths, 0.90)),
        "token_length_max": int(tr_lengths.max()),
        "length_quantiles_33_66": [float(q33), float(q66)],
        "test_oov_token_rate": oov,
        "test_truncation_rate_sample": truncated,
        "stopwords_removed": len(sw),
        "negations_retained": pcfg["keep_negations"],
    }
    log(f"stats={json.dumps(stats)}")

    cache = repo_root / dcfg["cache_dir"]
    cache.mkdir(parents=True, exist_ok=True)
    vocab.save(cache / "vocab.json")
    (cache / "preprocess_stats.json").write_text(json.dumps(stats, indent=2), "utf-8")

    return Task2Data(vocab=vocab, train=train, val=val, test=test, stats=stats)
