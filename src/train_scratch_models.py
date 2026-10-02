"""Common 512-token training entry point for the MLP and 1D-CNN models."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

from models.cnn1d import CNN1DSentimentClassifier
from models.mlp import MLPSentimentClassifier

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


def words(text):
    return TOKEN.findall(text.lower())


def encode(rows, vocab, max_len):
    x = torch.zeros((len(rows), max_len), dtype=torch.long)
    y = torch.tensor([label for _, label in rows], dtype=torch.long)
    for i, (text, _) in enumerate(rows):
        ids = [vocab.get(word, 1) for word in words(text)[:max_len]]
        if ids:
            x[i, :len(ids)] = torch.tensor(ids)
    return x, y


def metrics(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum()); tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum()); fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"accuracy": float((y_true == y_pred).mean()), "precision": precision,
            "recall": recall, "f1": f1, "confusion_matrix": [[tn, fp], [fn, tp]]}


def evaluate(model, loader, device):
    model.eval(); ys=[]; ps=[]
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu()); ys.append(y)
    return metrics(torch.cat(ys).numpy(), torch.cat(ps).numpy())


def main():
    p = argparse.ArgumentParser(); p.add_argument("--model", choices=["mlp", "cnn"], required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/aclImdb")); p.add_argument("--max-train", type=int, default=None)
    p.add_argument("--max-test", type=int, default=None); p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--epochs", type=int, default=5); p.add_argument("--batch-size", type=int, default=64); p.add_argument("--seed", type=int, default=42)
    a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed)
    rows=read_reviews(a.data_root/"train",a.max_train); test=read_reviews(a.data_root/"test",a.max_test); random.shuffle(rows); cut=int(.8*len(rows))
    train, valid=rows[:cut], rows[cut:]; counts={}
    for text,_ in train:
        for word in set(words(text)): counts[word]=counts.get(word,0)+1
    vocab={"<pad>":0,"<unk>":1}; vocab.update({w:i+2 for i,w in enumerate(sorted(counts,key=counts.get,reverse=True)[:20000])})
    xt,yt=encode(train,vocab,a.max_len); xv,yv=encode(valid,vocab,a.max_len); xte,yte=encode(test,vocab,a.max_len)
    train_loader=DataLoader(TensorDataset(xt,yt),batch_size=a.batch_size,shuffle=True); valid_loader=DataLoader(TensorDataset(xv,yv),batch_size=a.batch_size); test_loader=DataLoader(TensorDataset(xte,yte),batch_size=a.batch_size)
    model=(MLPSentimentClassifier if a.model=="mlp" else CNN1DSentimentClassifier)(len(vocab)); opt=torch.optim.Adam(model.parameters(),lr=1e-3); loss_fn=torch.nn.CrossEntropyLoss()
    for _ in range(a.epochs):
        model.train()
        for x,y in train_loader:
            loss=loss_fn(model(x),y); opt.zero_grad(); loss.backward(); opt.step()
    result={"model":a.model,"seed":a.seed,"dataset":{"train":len(train),"validation":len(valid),"test":len(test),"vocab":len(vocab),"max_length":a.max_len},"validation":evaluate(model,valid_loader,"cpu"),"test":evaluate(model,test_loader,"cpu")}
    out=Path(f"results/{a.model}_metrics.json"); out.write_text(json.dumps(result,indent=2),encoding="utf-8"); print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
