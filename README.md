# ICT-4442 Deep Learning Project

## Comparative Analysis of Deep Learning Architectures for Sentiment Analysis of Movie Reviews

This repository contains the model notebook contributed by Harshavardhan Reddy for the ICT-4442 Deep Learning project. The project uses the IMDb Large Movie Review Dataset to compare deep-learning architectures for binary sentiment classification.

## Repository structure

<!-- BEGIN AUTO-GENERATED REPOSITORY STRUCTURE -->
```text
imdb-sentiment-analysis-ict4442/
├── .github/
│   └── workflows/
│       └── update-readme-structure.yml
├── models/
│   ├── Bilstm.ipynb
│   ├── CNN.ipynb
│   └── MLP.ipynb
├── results/
│   ├── bilstm/
│   │   ├── history.json
│   │   ├── metrics.json
│   │   ├── test_probs.npy
│   │   └── val_probs.npy
│   ├── logs/
│   │   └── bilstm.log
│   └── tokenizer.json
├── scripts/
│   └── update_readme_structure.py
├── src/
│   ├── data.py
│   ├── evaluate.py
│   ├── models.py
│   └── train_scratch.py
├── .gitignore
├── README.md
└── requirements.txt
```
<!-- END AUTO-GENERATED REPOSITORY STRUCTURE -->

The structure shown above is generated automatically from the repository files. Whenever a new file is pushed to `main`, GitHub Actions updates this section.

The remaining model notebooks will be added by their respective contributors from their own GitHub accounts.

## Harshavardhan's contribution

### BiLSTM sentiment model

`models/Bilstm.ipynb` contains the complete BiLSTM experiment, including:

- train-only tokenisation and the common 20,000/5,000 train-validation split;
- GPU-compatible bidirectional LSTM training;
- early stopping and validation-loss checkpoint selection;
- validation metrics, classification report, confusion matrix, and ROC curve.

The fresh Phase 2 validation run reported:

| Metric | Value |
|---|---:|
| Accuracy | 0.8790 |
| Precision | 0.9010 |
| Recall | 0.8516 |
| F1 score | 0.8756 |
| ROC-AUC | 0.9490 |

The official IMDb test split was not used for interim model selection or evaluation.

## Running the notebook

Open `models/Bilstm.ipynb` in Google Colab, select a GPU runtime, and run the cells from top to bottom. The notebook downloads the IMDb dataset, creates the reproducible split with seed 42, trains the model, and displays the saved validation results.

## Dataset

IMDb Large Movie Review Dataset: <https://ai.stanford.edu/~amaas/data/sentiment/>
