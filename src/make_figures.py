"""Build the comparison table, figures and error analysis from saved results.

    python src/make_figures.py

Reads results/<model>/{metrics.json,history.json,test_probs.npy}; never
retrains anything, so every number in the report traces back to a file.
"""
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import clean, load_imdb
from evaluate import RESULTS

MODELS = [("mlp", "MLP"), ("cnn", "1D-CNN"), ("bilstm", "BiLSTM"), ("distilbert", "DistilBERT")]
COLORS = {"mlp": "#4C72B0", "cnn": "#DD8452", "bilstm": "#55A868", "distilbert": "#C44E52"}
FIG = RESULTS / "figures"
NEGATION = re.compile(r"\b(not|no|never|nothing|n't|nor|neither|hardly|barely)\b|n't\b")
CONTRAST = re.compile(r"\b(but|however|although|though|yet|despite)\b")


def chi2_sf(x):
    """Upper tail of the chi-squared distribution with one degree of freedom."""
    return math.erfc(math.sqrt(x / 2))


def load():
    runs = {}
    for key, label in MODELS:
        d = RESULTS / key
        if (d / "metrics.json").exists():
            runs[key] = {"label": label,
                         "metrics": json.loads((d / "metrics.json").read_text()),
                         "history": json.loads((d / "history.json").read_text()),
                         "test_probs": np.load(d / "test_probs.npy")}
    return runs


def comparison_table(runs):
    rows = []
    # median epoch time rather than total wall-clock: the DistilBERT run was suspended
    # mid-epoch, and laptop GPU throttling made some scratch epochs slow
    header = ["Model", "Params", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC",
              "Val acc", "Best epoch", "Epochs run", "Median epoch (s)", "Test inference (s)"]
    for key, r in runs.items():
        m, t = r["metrics"], r["metrics"]["test"]
        rows.append([r["label"], m["parameters"], t["accuracy"], t["precision"], t["recall"],
                     t["f1"], t["roc_auc"], m["val"]["accuracy"], m["best_epoch"], m["epochs_run"],
                     float(np.median(r["history"]["epoch_seconds"])), m["test_inference_seconds"]])
    with open(RESULTS / "comparison_table.csv", "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(round(v, 4)) if isinstance(v, float) else str(v) for v in row) + "\n")
    print("\n" + " | ".join(header))
    for row in rows:
        print(" | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row))


def training_curves(runs):
    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 6.4), squeeze=False)
    for j, (key, r) in enumerate(runs.items()):
        h = r["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        for i, (metric, name) in enumerate([("loss", "BCE loss"), ("accuracy", "Accuracy")]):
            ax = axes[i, j]
            ax.plot(ep, h[f"train_{metric}"], "o-", ms=3, label="Train")
            ax.plot(ep, h[f"val_{metric}"], "o-", ms=3, label="Validation")
            ax.axvline(h["best_epoch"], color="grey", ls="--", lw=1, label="Best epoch")
            ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.grid(alpha=.3)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
            if i == 0:
                ax.set_title(r["label"])
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "training_curves.png", dpi=200)
    plt.close(fig)

    # one loss/accuracy pair per model, for that model's subsection of the report
    for key, r in runs.items():
        h = r["history"]
        ep = np.arange(1, len(h["train_loss"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
        for ax, (metric, name) in zip(axes, [("loss", "BCE loss"), ("accuracy", "Accuracy")]):
            ax.plot(ep, h[f"train_{metric}"], "o-", ms=3, color=COLORS[key], label="Train")
            ax.plot(ep, h[f"val_{metric}"], "s--", ms=3, color="#444444", label="Validation")
            ax.axvline(h["best_epoch"], color="grey", ls=":", lw=1, label="Best epoch")
            ax.set_xlabel("Epoch"); ax.set_ylabel(name); ax.grid(alpha=.3); ax.legend(fontsize=7)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        fig.tight_layout()
        fig.savefig(FIG / f"curves_{key}.png", dpi=200)
        plt.close(fig)


def confusion_matrices(runs):
    fig, axes = plt.subplots(1, len(runs), figsize=(3.3 * len(runs), 3.2))
    for ax, (key, r) in zip(np.atleast_1d(axes), runs.items()):
        cm = np.array(r["metrics"]["test"]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (a, b), v in np.ndenumerate(cm):
            ax.text(b, a, f"{v:,}", ha="center", va="center",
                    color="white" if v > cm.max() / 2 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["Neg", "Pos"]); ax.set_yticks([0, 1], ["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"{r['label']} (acc {r['metrics']['test']['accuracy']:.3f})", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "confusion_matrices_test.png", dpi=200)
    plt.close(fig)


def roc_and_bars(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, r in runs.items():
        roc = r["metrics"]["test"]["roc"]
        a1.plot(roc["fpr"], roc["tpr"], color=COLORS[key],
                label=f"{r['label']} (AUC {r['metrics']['test']['roc_auc']:.3f})")
    a1.plot([0, 1], [0, 1], "k:", lw=1)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate")
    a1.set_title("ROC curves (test set)"); a1.legend(fontsize=8); a1.grid(alpha=.3)

    names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    w = 0.8 / len(runs)
    for i, (key, r) in enumerate(runs.items()):
        vals = [r["metrics"]["test"][n] for n in names]
        a2.bar(np.arange(len(names)) + i * w, vals, w, color=COLORS[key], label=r["label"])
    a2.set_xticks(np.arange(len(names)) + w * (len(runs) - 1) / 2,
                  ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"])
    a2.set_ylim(0.80, 1.0); a2.set_title("Test metrics"); a2.legend(fontsize=8); a2.grid(axis="y", alpha=.3)
    fig.tight_layout()
    fig.savefig(FIG / "roc_and_metrics_test.png", dpi=200)
    plt.close(fig)


def error_analysis(runs):
    data = load_imdb()
    texts, y = data["test"]
    lengths = np.array([len(clean(t).split()) for t in texts])
    neg = np.array([bool(NEGATION.search(t.lower())) for t in texts])
    contrast = np.array([bool(CONTRAST.search(t.lower())) for t in texts])
    correct = {k: ((r["test_probs"] >= 0.5).astype(int) == y) for k, r in runs.items()}

    bins = [0, 100, 200, 300, 512, 10_000]
    bin_names = ["<100", "100-199", "200-299", "300-511", ">=512"]
    which = np.digitize(lengths, bins[1:-1])
    out = {"length_buckets": bin_names,
           "bucket_sizes": [int((which == b).sum()) for b in range(len(bin_names))],
           "accuracy_by_length": {}, "subsets": {}, "agreement": {}}
    for k in runs:
        out["accuracy_by_length"][k] = [float(correct[k][which == b].mean()) for b in range(len(bin_names))]
        out["subsets"][k] = {
            "with_negation": float(correct[k][neg].mean()),
            "without_negation": float(correct[k][~neg].mean()),
            "with_contrast": float(correct[k][contrast].mean()),
            "without_contrast": float(correct[k][~contrast].mean()),
            "false_positives": int(((~correct[k]) & (y == 0)).sum()),
            "false_negatives": int(((~correct[k]) & (y == 1)).sum()),
        }
    all_wrong = np.all([~c for c in correct.values()], axis=0)
    all_right = np.all(list(correct.values()), axis=0)
    out["agreement"] = {"all_correct": int(all_right.sum()), "all_wrong": int(all_wrong.sum()),
                        "n_test": int(len(y)), "negation_share_test": float(neg.mean()),
                        "contrast_share_test": float(contrast.mean())}
    out["agreement"]["all_wrong_true_negative"] = int((all_wrong & (y == 0)).sum())
    # McNemar's test (chi-squared with continuity correction) on every pair of models
    out["mcnemar"] = {}
    keys = list(correct)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            n01 = int((correct[a] & ~correct[b]).sum())
            n10 = int((~correct[a] & correct[b]).sum())
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            out["mcnemar"][f"{a}_vs_{b}"] = {"only_first_correct": n01, "only_second_correct": n10,
                                             "chi2": chi2, "p_value": float(chi2_sf(chi2))}
    if "distilbert" in correct and "mlp" in correct:
        out["agreement"]["distilbert_fixes_mlp"] = int((correct["distilbert"] & ~correct["mlp"]).sum())
        out["agreement"]["mlp_fixes_distilbert"] = int((correct["mlp"] & ~correct["distilbert"]).sum())
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2))

    # accuracy vs review length
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    for k, r in runs.items():
        ax.plot(bin_names, out["accuracy_by_length"][k], "o-", color=COLORS[k], label=r["label"])
    ax.set_xlabel("Review length (words)"); ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy by review length"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "accuracy_by_length.png", dpi=200); plt.close(fig)

    # review length distribution (train split only)
    tr_len = np.array([len(clean(t).split()) for t in data["train"][0]])
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.hist(tr_len, bins=80, range=(0, 1500), color="#4C72B0", edgecolor="white")
    ax.axvline(512, color="red", ls="--", label=f"512 tokens ({np.mean(tr_len <= 512):.1%} of reviews)")
    ax.axvline(np.median(tr_len), color="black", ls=":", label=f"median {np.median(tr_len):.0f}")
    ax.set_xlabel("Words per review"); ax.set_ylabel("Reviews"); ax.legend(fontsize=8)
    ax.set_title("Training review lengths")
    fig.tight_layout(); fig.savefig(FIG / "review_length_distribution.png", dpi=200); plt.close(fig)

    # qualitative examples
    lines = ["# Error analysis examples (test set)\n"]

    def show(title, mask, n=5):
        lines.append(f"\n## {title} ({int(mask.sum())} reviews)\n")
        for i in np.where(mask)[0][:n]:
            probs = ", ".join(f"{runs[k]['label']}={runs[k]['test_probs'][i]:.2f}" for k in runs)
            snippet = re.sub(r"\s+", " ", texts[i].replace("<br />", " "))[:600]
            lines.append(f"- **True: {'positive' if y[i] else 'negative'}**, P(positive): {probs}\n  > {snippet}\n")

    show("Misclassified by every model", all_wrong)
    if "distilbert" in correct and "mlp" in correct:
        show("DistilBERT correct, MLP wrong", correct["distilbert"] & ~correct["mlp"])
    if "bilstm" in correct and "cnn" in correct:
        show("BiLSTM wrong, 1D-CNN correct", ~correct["bilstm"] & correct["cnn"])
    (RESULTS / "error_examples.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(out, indent=2))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    runs = load()
    print("models found:", list(runs))
    comparison_table(runs)
    training_curves(runs)
    confusion_matrices(runs)
    roc_and_bars(runs)
    error_analysis(runs)


if __name__ == "__main__":
    main()
