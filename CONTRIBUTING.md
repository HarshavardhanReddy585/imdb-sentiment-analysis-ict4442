# Contribution workflow

Use one focused commit per meaningful contribution, for example:

```text
feat: add IMDb preprocessing pipeline
feat: implement MLP baseline
feat: implement BiLSTM model
docs: add Phase 2 literature review
```

Before merging code from another repository:

1. Add the repository as a named remote for inspection.
2. Review its commit history and identify the contributing member.
3. Check that the code runs with this repository's fixed split and metrics.
4. Copy or merge only the needed files into the canonical structure.
5. Preserve attribution in the report and commit message.
6. Run the smoke experiment and record the result.

Do not commit datasets, credentials, notebook checkpoints, or unverified results.
