"""Multi-kernel 1D-CNN sentiment classifier."""

from __future__ import annotations

import torch
from torch import nn


class CNN1DSentimentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 filters: int = 64, kernel_sizes=(3, 4, 5),
                 hidden_dim: int = 64, dropout: float = 0.4,
                 num_classes: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout_embedding = nn.Dropout(0.3)
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, filters, kernel_size) for kernel_size in kernel_sizes
        ])
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(filters * len(kernel_sizes), hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout_embedding(self.embedding(token_ids)).transpose(1, 2)
        features = [torch.relu(conv(embedded)).amax(dim=2) for conv in self.convs]
        return self.classifier(torch.cat(features, dim=1))
