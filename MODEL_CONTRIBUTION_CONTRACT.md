# Model contribution contract

Each member must build and push their assigned model. A repository URL alone is not a contribution.

## Required model ownership

| Owner | Model | Required location |
|---|---|---|
| Meesha Gupta | DistilBERT | `src/models/distilbert.py` |
| Ratul Jha | 1D-CNN | `src/models/cnn1d.py` |
| Harshavardhan Reddy | BiLSTM | `src/models/bilstm.py` |
| Kshitij Singh | MLP | `src/models/mlp.py` |

## Required push contents

Each model contribution must include:

- model implementation and a runnable training/evaluation entry point;
- exact preprocessing and hyperparameters;
- validation and held-out test metrics: accuracy, precision, recall, F1, and confusion matrix;
- reproducible command and dependency changes, if any;
- at least one meaningful commit authored by the model owner;
- disclosure of external code, tutorials, or LLM assistance.

Use a branch such as `model/<name>` and open a pull request into `main`, or push directly only when the owner is explicitly authorized. Do not upload the dataset, supplied guidelines, report templates, TODO lists, secrets, or temporary files.

The model is not considered complete until it runs on the common split and its evidence is reviewed.
