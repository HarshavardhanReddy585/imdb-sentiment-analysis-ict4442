# Comparative Sentiment Analysis of IMDb Movie Reviews

ICT 4442 Deep Learning mini project.

This repository compares four approved model families on the IMDb binary sentiment task:

- MLP with averaged learned embeddings
- 1D-CNN with max-pooled n-gram features
- BiLSTM
- DistilBERT

## Repository layout

`src/` contains reusable preprocessing, training, and evaluation code. `data/` is reserved for locally downloaded data and is ignored by Git. `results/` contains generated metrics and figures. `report/` contains the interim and final report sources. `docs/` contains the supplied project guidelines and synopsis.

## Current status

The Phase 2 report and CPU-friendly MLP/CNN experiment script are being completed. BiLSTM and DistilBERT remain final-phase work. Results must be generated from the official IMDb dataset; no fabricated metrics are acceptable.

## Reproducibility

```powershell
python src/run_experiments.py --archive data/aclImdb_v1.tar.gz --max-train 4000 --max-test 1000
```

The script writes metrics to `results/metrics.json` and `results/metrics.csv`.

## Team and repository policy

This repository is the canonical submission repository. Teammate repositories may be connected as Git remotes for review, but code is merged only after checking ownership, licensing, dependencies, and reproducibility. Dataset archives, generated caches, secrets, and temporary files are never committed.
