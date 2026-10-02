"""Minimal BiLSTM smoke-test entry point."""

from __future__ import annotations

import argparse

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for BiLSTM training. Install requirements.txt first."
    ) from exc

from models.bilstm import BiLSTMSentimentClassifier


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    model = BiLSTMSentimentClassifier(args.vocab_size)
    example = torch.ones((2, args.sequence_length), dtype=torch.long)
    logits = model(example)
    print({"model": "BiLSTM", "logits_shape": list(logits.shape), "seed": args.seed})


if __name__ == "__main__":
    main()
