# Harshavardhan Reddy - BiLSTM handoff

Owner: Harshavardhan Reddy (230953396)  
Assignment: Bidirectional LSTM sentiment classifier

Planned implementation:

- Shared tokenizer and fixed split from `src/run_experiments.py`
- Embedding layer followed by a bidirectional LSTM
- Dropout and sigmoid/softmax classification head
- Common metrics and saved confusion matrix

Implementation is in `src/models/bilstm.py` and `src/train_bilstm.py`. PyTorch forward smoke test passes with logits shape `[2, 2]`. A two-epoch smoke run on 800 train / 200 validation / 200 test reviews produced validation accuracy 46.5%, F1 0.389 and test accuracy 51.0%, F1 0.372. Full-scale tuning remains pending.
