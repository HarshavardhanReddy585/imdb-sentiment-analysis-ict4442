"""Common evaluation protocol: identical metrics and outputs for every model."""
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

RESULTS = Path(__file__).resolve().parents[1] / "results"
THRESHOLD = 0.5


def compute_metrics(labels, probs) -> dict:
    preds = (np.asarray(probs) >= THRESHOLD).astype(int)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds)),
        "recall": float(recall_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
        "confusion_matrix": confusion_matrix(labels, preds).tolist(),
    }


def save_run(name: str, history: dict, splits_probs: dict, labels: dict, extra: dict):
    """Write history.json, metrics.json and per-split probabilities."""
    out = RESULTS / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "history.json").write_text(json.dumps(history, indent=2))
    metrics = {"model": name, **extra}
    for split, probs in splits_probs.items():
        np.save(out / f"{split}_probs.npy", np.asarray(probs, dtype=np.float32))
        metrics[split] = compute_metrics(labels[split], probs)
        fpr, tpr, _ = roc_curve(labels[split], probs)
        idx = np.linspace(0, len(fpr) - 1, min(len(fpr), 400)).astype(int)
        metrics[split]["roc"] = {"fpr": fpr[idx].tolist(), "tpr": tpr[idx].tolist()}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for split in splits_probs:
        m = metrics[split]
        print(f"[{name}] {split:5s} acc={m['accuracy']:.4f} P={m['precision']:.4f} "
              f"R={m['recall']:.4f} F1={m['f1']:.4f} AUC={m['roc_auc']:.4f}")
    return metrics
