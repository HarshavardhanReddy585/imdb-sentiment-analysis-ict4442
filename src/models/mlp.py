"""Order-insensitive MLP sentiment baseline."""

from __future__ import annotations

import torch
from torch import nn


class MLPSentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 64, dropout: float = 0.4,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_embedding = nn.Dropout2d(0.2)
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)
        mask = token_ids.ne(0).unsqueeze(-1)
        pooled = (embedded * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
        return self.classifier(pooled)
