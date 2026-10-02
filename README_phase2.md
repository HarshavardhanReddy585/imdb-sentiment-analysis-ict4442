# Phase 2 interim package

This folder contains the reproducible interim pipeline for the approved IMDb sentiment project.

## Quick start

1. Download `aclImdb_v1.tar.gz` from the official Stanford source listed in the report, or place the archive at `data/aclImdb_v1.tar.gz`.
2. Run:

```powershell
python phase2/run_experiments.py --archive data/aclImdb_v1.tar.gz --max-train 4000 --max-test 1000
```

The script extracts the official train/test folders, creates a deterministic 80/20 train/validation split, builds a vocabulary from the training partition only, trains an MLP and a 1D-CNN-style n-gram model, and writes JSON/CSV metrics under `phase2/results/`.

The implementation is intentionally CPU-friendly for the overdue interim milestone. Final work should rerun the same pipeline at full scale, add BiLSTM and DistilBERT, and report the common held-out test metrics.

## Integrity note

The report must include the real repository URL, actual Git history, and signatures. Do not submit the placeholders in the report without replacing them with the team's evidence.
