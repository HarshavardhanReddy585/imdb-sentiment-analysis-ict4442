"""Train and evaluate the MLP, 1D-CNN or BiLSTM under the common protocol.

    python src/train_scratch.py --model mlp
    python src/train_scratch.py --model cnn
    python src/train_scratch.py --model bilstm

Training mirrors the Phase 2 Keras setup: Adam, binary cross-entropy,
L2(1e-4) on the hidden dense layer, ReduceLROnPlateau(factor 0.5,
patience 1) and early stopping on validation loss with the best weights
restored. The test split is evaluated once, after training has finished.
"""
import argparse
import random
import time

import numpy as np
import torch
from torch import nn

from data import MAX_LEN, encoded_splits
from evaluate import RESULTS, save_run
from models import CONFIGS

SEED = 42
L2 = 1e-4


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, X, device, batch_size=512):
    model.eval()
    probs = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(device)
        probs.append(torch.sigmoid(model(xb)).float().cpu())
    return torch.cat(probs).numpy()


def bce(probs, y):
    p = np.clip(probs, 1e-7, 1 - 1e-7)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=CONFIGS, required=True)
    args = ap.parse_args()

    set_seed()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    splits, _, tok = encoded_splits(MAX_LEN)
    (Xtr, ytr), (Xva, yva), (Xte, yte) = splits["train"], splits["val"], splits["test"]
    RESULTS.mkdir(exist_ok=True)
    tok.save(RESULTS / "tokenizer.json")

    cls, cfg = CONFIGS[args.model]
    model = cls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{cls.__name__}: {n_params:,} parameters on {device}; config {cfg}")

    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    # patience=0 here equals Keras ReduceLROnPlateau(patience=1): halve after one epoch without improvement
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=0, min_lr=1e-6)
    loss_fn = nn.BCEWithLogitsLoss()
    hist = {k: [] for k in ["train_loss", "val_loss", "train_accuracy", "val_accuracy", "lr", "epoch_seconds"]}
    best_loss, best_state, bad_epochs = float("inf"), None, 0
    gen = torch.Generator().manual_seed(SEED)
    t_start = time.time()

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t0 = time.time()
        order = torch.randperm(len(Xtr), generator=gen).numpy()
        tot_loss, correct = 0.0, 0
        for i in range(0, len(order), cfg["batch_size"]):
            idx = order[i:i + cfg["batch_size"]]
            xb = torch.from_numpy(Xtr[idx]).to(device)
            yb = torch.from_numpy(ytr[idx]).float().to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            reg = L2 * sum((w ** 2).sum() for w in model.l2_params())
            opt.zero_grad()
            (loss + reg).backward()
            opt.step()
            tot_loss += loss.item() * len(idx)
            correct += ((logits > 0).float() == yb).sum().item()

        va_probs = predict(model, Xva, device)
        va_loss = bce(va_probs, yva)
        va_acc = float(((va_probs >= 0.5) == yva).mean())
        hist["train_loss"].append(tot_loss / len(Xtr))
        hist["train_accuracy"].append(correct / len(Xtr))
        hist["val_loss"].append(va_loss)
        hist["val_accuracy"].append(va_acc)
        hist["lr"].append(opt.param_groups[0]["lr"])
        hist["epoch_seconds"].append(time.time() - t0)
        print(f"epoch {epoch:2d} train_loss={hist['train_loss'][-1]:.4f} "
              f"train_acc={hist['train_accuracy'][-1]:.4f} val_loss={va_loss:.4f} "
              f"val_acc={va_acc:.4f} lr={hist['lr'][-1]:.2e} ({hist['epoch_seconds'][-1]:.0f}s)")

        if va_loss < best_loss:
            best_loss, bad_epochs = va_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping after epoch {epoch}")
                break
        sched.step(va_loss)

    train_seconds = time.time() - t_start
    model.load_state_dict(best_state)
    hist["epochs_run"] = len(hist["train_loss"])
    hist["best_epoch"] = int(np.argmin(hist["val_loss"]) + 1)

    t0 = time.time()
    te_probs = predict(model, Xte, device)
    infer_seconds = time.time() - t0
    va_probs = predict(model, Xva, device)
    save_run(args.model, hist, {"val": va_probs, "test": te_probs},
             {"val": yva, "test": yte},
             {"architecture": cls.__name__, "parameters": n_params, "config": cfg,
              "max_len": MAX_LEN, "train_seconds": train_seconds,
              "test_inference_seconds": infer_seconds,
              "best_epoch": hist["best_epoch"], "epochs_run": hist["epochs_run"],
              "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"})


if __name__ == "__main__":
    main()
