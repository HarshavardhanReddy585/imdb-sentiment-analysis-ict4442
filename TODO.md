# Project TODO

## Phase 2 urgent work

- [x] Create a clean canonical project directory.
- [x] Add supplied guidelines and synopsis under `docs/`.
- [x] Add reproducible MLP/CNN experiment script.
- [x] Add interim report draft.
- [x] Initialize local Git repository on `main`.
- [ ] Create or select the GitHub repository and add `origin`.
- [ ] Download the official IMDb archive into `data/`.
- [ ] Run the smoke experiment and save real metrics under `results/`.
- [ ] Update the interim report with measured metrics.
- [ ] Add actual GitHub URL to the report.
- [ ] Record the solo contribution accurately; do not invent teammate signatures.
- [ ] Request faculty approval for late submission.
- [ ] Submit Part B through MS Forms.

## Final phase

- [ ] Implement BiLSTM.
- [ ] Implement DistilBERT.
- [ ] Evaluate all models on the same held-out test set.
- [ ] Add confusion matrices, learning curves, and error analysis.
- [ ] Expand references to at least 12.
- [ ] Prepare Part C report and signed contribution statement.
- [ ] Prepare presentation and viva notes.
- [ ] Push final code and submit the repository link.

## Repository integration policy

Use this repository as the canonical submission repository. Other repositories can be connected as named remotes for inspection:

```powershell
git remote add teammate-meesha <url>
git fetch teammate-meesha
git log --oneline teammate-meesha/main
```

Merge only selected, reviewed commits. Never commit datasets, secrets, temporary files, or unverified results.
