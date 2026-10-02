# Harshavardhan Reddy - BiLSTM handoff

Owner: Harshavardhan Reddy (230953396)  
Assignment: Bidirectional LSTM sentiment classifier

Planned implementation:

- Shared tokenizer and fixed split from `src/run_experiments.py`
- Embedding layer followed by a bidirectional LSTM
- Dropout and sigmoid/softmax classification head
- Common metrics and saved confusion matrix

Implementation started in `src/models/bilstm.py`. PyTorch forward smoke test passes with logits shape `[2, 2]`; training results are still pending. Record final hyperparameters, runtime, and failure cases here after training.
