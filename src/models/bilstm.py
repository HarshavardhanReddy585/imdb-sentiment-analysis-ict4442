"""Bidirectional LSTM classifier for IMDb sentiment.

Input token IDs use zero for padding and one for unknown tokens.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


if nn is not None:

    class BiLSTMSentimentClassifier(nn.Module):
        """Embedding -> BiLSTM -> final bidirectional state -> classifier."""

        def __init__(self, vocab_size: int, embedding_dim: int = 128,
                     hidden_dim: int = 128, num_layers: int = 1,
                     dropout: float = 0.3, num_classes: int = 2) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
            recurrent_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                embedding_dim, hidden_dim, num_layers=num_layers,
                batch_first=True, bidirectional=True, dropout=recurrent_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_dim * 2, num_classes)

        def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
            lengths = token_ids.ne(0).sum(dim=1).clamp_min(1).cpu()
            embedded = self.embedding(token_ids)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths, batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
            document_state = torch.cat((hidden[-2], hidden[-1]), dim=1)
            return self.classifier(self.dropout(document_state))

else:

    class BiLSTMSentimentClassifier:
        """Clear error when the optional final-model dependency is absent."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "BiLSTM training requires PyTorch. Install requirements.txt first."
            )
