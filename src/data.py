"""Shared data pipeline used by every model in the comparison.

Common protocol (identical for MLP, 1D-CNN, BiLSTM and DistilBERT):
  * IMDb Large Movie Review Dataset (Maas et al., 2011), official archive.
  * Official 25,000-review training set -> stratified 20,000 train / 5,000
    validation split with seed 42.
  * Official 25,000-review test set is only used for the final evaluation.
  * The word-level tokenizer is fitted on the 20,000 training reviews only.
"""
from __future__ import annotations

import json
import pickle
import re
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARCHIVE = DATA_DIR / "aclImdb_v1.tar.gz"
CACHE = DATA_DIR / "imdb_reviews.pkl"
URL = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

SEED = 42
VAL_SIZE = 5000
VOCAB_SIZE = 20000  # includes index 0 (padding) and index 1 (<OOV>)
MAX_LEN = 512

_FILTERS = re.compile(r"[!\"#$%&()*+,\-./:;<=>?@\[\\\]^_`{|}~\t\n]")
_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _read_archive(archive: Path) -> dict:
    """Read every review straight from the .tar.gz (no extraction needed)."""
    splits = {"train": ([], []), "test": ([], [])}
    pattern = re.compile(r"aclImdb/(train|test)/(pos|neg)/\d+_\d+\.txt$")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            m = pattern.match(member.name)
            if not m:
                continue
            text = tar.extractfile(member).read().decode("utf-8", errors="ignore")
            texts, labels = splits[m.group(1)]
            texts.append(text)
            labels.append(1 if m.group(2) == "pos" else 0)
    # Sort by (label, text) so the order does not depend on archive layout.
    out = {}
    for name, (texts, labels) in splits.items():
        order = sorted(range(len(texts)), key=lambda i: (labels[i], texts[i]))
        out[name] = ([texts[i] for i in order], np.array([labels[i] for i in order]))
    return out


def load_imdb() -> dict:
    """Return {'train': (texts, labels), 'val': ..., 'test': ...}."""
    if CACHE.exists():
        with open(CACHE, "rb") as f:
            raw = pickle.load(f)
    else:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not ARCHIVE.exists():
            print(f"Downloading {URL} ...")
            urllib.request.urlretrieve(URL, ARCHIVE)
        raw = _read_archive(ARCHIVE)
        with open(CACHE, "wb") as f:
            pickle.dump(raw, f)

    full_texts, full_labels = raw["train"]
    tr_x, va_x, tr_y, va_y = train_test_split(
        full_texts, full_labels, test_size=VAL_SIZE,
        random_state=SEED, stratify=full_labels)
    return {"train": (list(tr_x), np.asarray(tr_y)),
            "val": (list(va_x), np.asarray(va_y)),
            "test": (raw["test"][0], np.asarray(raw["test"][1]))}


def clean(text: str) -> str:
    """Remove HTML line breaks, lowercase, and strip punctuation."""
    return _FILTERS.sub(" ", _BREAK.sub(" ", text).lower())


class WordTokenizer:
    """Frequency-ranked word index, equivalent to the Keras Tokenizer used in
    Phase 2 (num_words=20000, oov_token='<OOV>', post padding/truncation)."""

    def __init__(self, num_words: int = VOCAB_SIZE):
        self.num_words = num_words
        self.word_index: dict[str, int] = {}

    def fit(self, texts):
        counts = Counter(w for t in texts for w in clean(t).split())
        ranked = [w for w, _ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))]
        # 0 = padding, 1 = <OOV>, real words start at 2
        self.word_index = {w: i + 2 for i, w in enumerate(ranked[: self.num_words - 2])}
        return self

    def encode(self, texts, max_len: int = MAX_LEN) -> np.ndarray:
        out = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, t in enumerate(texts):
            ids = [self.word_index.get(w, 1) for w in clean(t).split()][:max_len]
            out[i, : len(ids)] = ids
        return out

    def save(self, path: Path):
        path.write_text(json.dumps({"num_words": self.num_words,
                                    "word_index": self.word_index}))


def encoded_splits(max_len: int = MAX_LEN):
    """Tokenizer fitted on train only; returns arrays for all three splits."""
    data = load_imdb()
    tok = WordTokenizer().fit(data["train"][0])
    return {k: (tok.encode(x, max_len), y) for k, (x, y) in data.items()}, data, tok


if __name__ == "__main__":
    splits, data, tok = encoded_splits()
    for k, (x, y) in splits.items():
        print(f"{k:5s} X={x.shape} positives={int(y.sum())}")
    lengths = np.array([len(clean(t).split()) for t in data["train"][0]])
    print(f"median tokens={np.median(lengths):.0f}  <= {MAX_LEN}: {np.mean(lengths <= MAX_LEN):.1%}")
