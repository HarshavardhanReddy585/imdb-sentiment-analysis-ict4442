"""Train and evaluate the BiLSTM on the official IMDb folder structure."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

try:
    import torch
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for BiLSTM training. Install requirements.txt first."
    ) from exc

from models.bilstm import BiLSTMSentimentClassifier

TOKEN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def read_reviews(root: Path, limit: int | None):
    rows = []
    for label, name in [(0, "neg"), (1, "pos")]:
        files = sorted((root / name).glob("*.txt"))
        if limit:
            files = files[: limit // 2]
        for path in files:
            rows.append((path.read_text(encoding="utf-8", errors="ignore"), label))
    return rows


def tokens(text: str):
    return TOKEN.findall(text.lower())


def encode(rows, vocab, max_len):
    x = torch.zeros((len(rows), max_len), dtype=torch.long)
    y = torch.tensor([label for _, label in rows], dtype=torch.long)
    for i, (text, _) in enumerate(rows):
        ids = [vocab.get(word, 1) for word in tokens(text)[:max_len]]
        if ids:
            x[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
    return x, y


def scores(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": float((y_true == y_pred).mean()), "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": [[tn, fp], [fn, tp]]}


def evaluate(model, x, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).argmax(dim=1).cpu().numpy()
    return scores(y.numpy(), pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/aclImdb"))
    parser.add_argument("--max-train", type=int, default=1000)
    parser.add_argument("--max-test", type=int, default=200)
    parser.add_argument("--sequence-length", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rows = read_reviews(args.data_root / "train", args.max_train)
    test = read_reviews(args.data_root / "test", args.max_test)
    random.shuffle(rows)
    split = int(len(rows) * 0.8)
    train, valid = rows[:split], rows[split:]
    counts = {}
    for text, _ in train:
        for word in set(tokens(text)):
            counts[word] = counts.get(word, 0) + 1
    vocab = {"<pad>": 0, "<unk>": 1}
    vocab.update({word: i + 2 for i, word in enumerate(sorted(counts, key=counts.get, reverse=True)[:12000])})
    x_train, y_train = encode(train, vocab, args.sequence_length)
    x_valid, y_valid = encode(valid, vocab, args.sequence_length)
    x_test, y_test = encode(test, vocab, args.sequence_length)
    device = torch.device("cpu")
    model = BiLSTMSentimentClassifier(len(vocab)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    model.train()
    for _ in range(args.epochs):
        order = torch.randperm(len(x_train))
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            logits = model(x_train[batch].to(device))
            loss = loss_fn(logits, y_train[batch].to(device))
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    result = {"model": "BiLSTM", "seed": args.seed,
              "dataset": {"train": len(train), "validation": len(valid), "test": len(test),
                          "vocab": len(vocab), "sequence_length": args.sequence_length},
              "validation": evaluate(model, x_valid, y_valid, device),
              "test": evaluate(model, x_test, y_test, device)}
    out = Path("results/bilstm_metrics.json")
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
