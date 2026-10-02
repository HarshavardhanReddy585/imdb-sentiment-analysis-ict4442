# Comparative Sentiment Analysis of IMDb Movie Reviews

ICT 4442 Deep Learning mini project.

This repository compares four approved model families on the IMDb binary sentiment task:

- MLP with averaged learned embeddings
- 1D-CNN with max-pooled n-gram features
- BiLSTM
- DistilBERT

## Repository layout

`src/` contains reusable preprocessing, model, training, and evaluation code. `data/` is reserved for locally downloaded data and is ignored by Git. `results/` contains generated metrics and figures. `report/` contains the interim and final report sources.

## Current status

The Phase 2 report, PyTorch MLP/CNN/BiLSTM implementations, and reproducible smoke-run evidence are present. DistilBERT remains final-phase work. Results must be generated from the official IMDb dataset; no fabricated metrics are acceptable.

## Reproducibility

```powershell
python src/train_scratch_models.py --model mlp --data-root data/aclImdb --max-train 1000 --max-test 200 --max-len 512 --epochs 2
python src/train_scratch_models.py --model cnn --data-root data/aclImdb --max-train 1000 --max-test 200 --max-len 512 --epochs 2
python src/train_bilstm.py --data-root data/aclImdb --max-train 1000 --max-test 200 --sequence-length 40 --epochs 2
```

The scripts write model-specific metrics to `results/*_metrics.json`. The original preprocessing smoke runner writes `results/metrics.json` and `results/metrics.csv`.

## Team and repository policy

This repository is the canonical submission repository. Each model owner must push their implementation and evidence through a named branch or pull request. See `MODEL_CONTRIBUTION_CONTRACT.md`. Dataset archives, generated caches, secrets, and temporary files are never committed.
