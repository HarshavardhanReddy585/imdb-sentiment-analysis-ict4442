"""Fine-tune DistilBERT (distilbert-base-uncased) under the common protocol.

    python src/train_distilbert.py

Same 20,000 / 5,000 / 25,000 split as the scratch models. DistilBERT uses
its own WordPiece tokenizer (256 sub-word tokens) because it must match the
pretrained vocabulary. Configuration selected in Phase 2: AdamW lr 2e-5,
weight decay 0.01, linear warm-up over 10% of steps then linear decay,
gradient clipping 1.0, batch 16, at most 5 epochs, early stopping
(patience 2) on validation loss. Mixed precision (fp16) is used on GPU.
"""
import random
import time

import numpy as np
import torch
from torch import nn
from transformers import (DistilBertForSequenceClassification,
                          DistilBertTokenizerFast,
                          get_linear_schedule_with_warmup)

from data import load_imdb
from evaluate import save_run

SEED = 42
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 256
BATCH_SIZE = 16
LR = 2e-5
MAX_EPOCHS = 5
PATIENCE = 2
WARMUP_FRAC = 0.1


def encode(tokenizer, texts):
    enc = tokenizer([t.replace("<br />", " ") for t in texts], truncation=True,
                    padding="max_length", max_length=MAX_LEN, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"]


@torch.no_grad()
def predict(model, ids, mask, device, batch_size=128):
    model.eval()
    probs = []
    for i in range(0, len(ids), batch_size):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids[i:i + batch_size].to(device),
                           attention_mask=mask[i:i + batch_size].to(device)).logits
        probs.append(torch.sigmoid(logits.float().squeeze(-1)).cpu())
    return torch.cat(probs).numpy()


def bce(probs, y):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = load_imdb()
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    enc = {k: encode(tokenizer, x) for k, (x, _) in data.items()}
    labels = {k: y for k, (_, y) in data.items()}

    model = DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=1).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"DistilBERT: {n_params:,} parameters on {device}")

    tr_ids, tr_mask = enc["train"]
    ytr = torch.tensor(labels["train"], dtype=torch.float32)
    steps_per_epoch = (len(tr_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP_FRAC * steps_per_epoch * MAX_EPOCHS),
                                            steps_per_epoch * MAX_EPOCHS)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    loss_fn = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(SEED)

    hist = {k: [] for k in ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "lr", "epoch_seconds"]}
    best_loss, best_state, bad = float("inf"), None, 0
    t_start = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(tr_ids), generator=gen)
        tot, correct = 0.0, 0
        for step, i in enumerate(range(0, len(order), BATCH_SIZE)):
            idx = order[i:i + BATCH_SIZE]
            yb = ytr[idx].to(device)
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=tr_ids[idx].to(device),
                               attention_mask=tr_mask[idx].to(device)).logits.squeeze(-1)
            loss = loss_fn(logits.float(), yb)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item() * len(idx)
            correct += ((logits > 0).float() == yb).sum().item()
            if step % 250 == 0:
                print(f"  epoch {epoch} step {step}/{steps_per_epoch} loss={loss.item():.4f}", flush=True)

        va_probs = predict(model, *enc["val"], device)
        va_loss = bce(va_probs, labels["val"])
        hist["train_loss"].append(tot / len(tr_ids))
        hist["train_accuracy"].append(correct / len(tr_ids))
        hist["val_loss"].append(va_loss)
        hist["val_accuracy"].append(float(((va_probs >= 0.5) == labels["val"]).mean()))
        hist["lr"].append(sched.get_last_lr()[0])
        hist["epoch_seconds"].append(time.time() - t0)
        print(f"epoch {epoch} train_loss={hist['train_loss'][-1]:.4f} train_acc={hist['train_accuracy'][-1]:.4f} "
              f"val_loss={va_loss:.4f} val_acc={hist['val_accuracy'][-1]:.4f} ({hist['epoch_seconds'][-1]:.0f}s)",
              flush=True)
        if va_loss < best_loss:
            best_loss, bad = va_loss, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                print(f"early stopping after epoch {epoch}")
                break

    train_seconds = time.time() - t_start
    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)
    t0 = time.time()
    te_probs = predict(model, *enc["test"], device)
    infer_seconds = time.time() - t0
    va_probs = predict(model, *enc["val"], device)
    save_run("distilbert", hist, {"val": va_probs, "test": te_probs},
             {"val": labels["val"], "test": labels["test"]},
             {"architecture": "DistilBERT", "parameters": n_params,
              "config": {"lr": LR, "batch_size": BATCH_SIZE, "max_epochs": MAX_EPOCHS,
                         "patience": PATIENCE, "warmup_frac": WARMUP_FRAC, "weight_decay": 0.01},
              "max_len": MAX_LEN, "train_seconds": train_seconds,
              "test_inference_seconds": infer_seconds,
              "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})


if __name__ == "__main__":
    main()
