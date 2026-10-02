# Team contribution handoff area

This folder is a structured drop zone for the four model owners from the approved synopsis. Each member has a separate folder so their code, notes, and evidence can be reviewed independently before anything is merged into `src/`.

## Assignment map

| Folder | Member | Assignment | Expected handoff |
|---|---|---|---|
| `meesha_distilbert/` | Meesha Gupta - 230953090 | DistilBERT | Model code, tokenizer/training settings, metrics, commit evidence |
| `ratul_cnn/` | Ratul Jha - 230953338 | 1D-CNN | Model code, preprocessing assumptions, metrics, commit evidence |
| `harshavardhan_bilstm/` | Harshavardhan Reddy - 230953396 | BiLSTM | Model code, training settings, metrics, error notes |
| `kshitij_mlp/` | Kshitij Singh - 230953460 | MLP | Model code, representation details, metrics, commit evidence |

## Integration rule

Incoming work stays in its member folder until it passes the common contract:

1. Same IMDb train/validation/test split.
2. Same label mapping and preprocessing documentation.
3. Same metrics: accuracy, precision, recall, F1, and confusion matrix.
4. Reproducible command and dependency list.
5. No dataset archive, secrets, copied tutorial code, or fabricated result.
6. Member attribution and commit history preserved.

After review, the working implementation can be copied or merged into `src/models/` and recorded in the report.
