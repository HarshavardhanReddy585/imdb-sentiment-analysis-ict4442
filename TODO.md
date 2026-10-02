# Project TODO

## Phase 2 urgent work

- [x] Create a clean canonical project directory.
- [x] Add supplied guidelines and synopsis under `docs/`.
- [x] Add reproducible MLP/CNN experiment script.
- [x] Add interim report draft and actual smoke-run metrics.
- [x] Initialize local Git repository on `main`.
- [x] Create the private GitHub repository and add `origin`.
- [x] Download and extract the official IMDb archive into `data/`.
- [x] Run the smoke experiment and save real metrics under `results/`.
- [x] Update the interim report with measured metrics.
- [x] Add the actual GitHub URL to the report.
- [x] Create separate teammate handoff folders under `team_contributions/`.
- [ ] Record the solo contribution accurately; do not invent teammate signatures.
- [ ] Request faculty approval for late submission.
- [ ] Submit Part B through MS Forms.

## Final phase

- [x] Implement BiLSTM model module and training entry point; record preliminary validation/test metrics.
- [ ] Implement DistilBERT.
- [ ] Evaluate all models on the same held-out test set.
- [ ] Add confusion matrices, learning curves, and error analysis.
- [ ] Expand references to at least 12.
- [ ] Prepare Part C report and signed contribution statement.
- [ ] Prepare presentation and viva notes.
- [ ] Push final code and submit the repository link.

## Team handoff status

- [ ] Meesha: DistilBERT repository/code/evidence received.
- [ ] Ratul: 1D-CNN repository/code/evidence received.
- [ ] Harshavardhan: BiLSTM implementation completed.
- [ ] Kshitij: MLP repository/code/evidence received.
- [ ] Review all incoming code against the common split and metrics.

## Repository integration policy

Use this repository as the canonical submission repository. Other repositories can be connected as named remotes for inspection:

```powershell
git remote add teammate-meesha <url>
git fetch teammate-meesha
git log --oneline teammate-meesha/main
```

Merge only selected, reviewed commits. Never commit datasets, secrets, temporary files, or unverified results.
