"""The three word-level models trained from scratch.

All three share the same input (20,000-word vocabulary, 512 tokens, 128-d
learned embedding) so that differences in performance come from how each
architecture combines the word vectors, not from the input representation.
Hyperparameters are the ones selected during Phase 2.
"""
import torch
from torch import nn

VOCAB_SIZE = 20000
EMBED_DIM = 128


class SpatialDropout1D(nn.Module):
    """Drops whole embedding channels for a review (Keras SpatialDropout1D)."""

    def __init__(self, p: float):
        super().__init__()
        self.p = p

    def forward(self, x):  # x: (batch, seq, dim)
        if not self.training or self.p == 0:
            return x
        mask = x.new_empty(x.size(0), 1, x.size(2)).bernoulli_(1 - self.p) / (1 - self.p)
        return x * mask


def _embedding():
    emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM, padding_idx=0)
    nn.init.uniform_(emb.weight, -0.05, 0.05)
    with torch.no_grad():
        emb.weight[0].zero_()
    return emb


class MLP(nn.Module):
    """Bag-of-embeddings baseline: word order is discarded by averaging.

    Embedding -> SpatialDropout(0.2) -> masked mean pooling ->
    Dense(64, ReLU) -> Dropout(0.4) -> Dense(1)
    """

    def __init__(self):
        super().__init__()
        self.embedding = _embedding()
        self.emb_drop = SpatialDropout1D(0.2)
        self.hidden = nn.Linear(EMBED_DIM, 64)
        self.drop = nn.Dropout(0.4)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        mask = (x != 0).unsqueeze(-1).float()
        e = self.emb_drop(self.embedding(x))
        pooled = (e * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return self.out(self.drop(torch.relu(self.hidden(pooled)))).squeeze(-1)

    def l2_params(self):
        return [self.hidden.weight]


class CNN1D(nn.Module):
    """Kim (2014)-style multi-width convolution over word embeddings.

    Embedding -> SpatialDropout(0.3) -> 3 parallel Conv1D(64 filters,
    widths 3/4/5, ReLU) -> global max pooling -> concat(192) ->
    Dropout(0.4) -> Dense(64, ReLU) -> Dropout(0.3) -> Dense(1)
    """

    def __init__(self, widths=(3, 4, 5), filters=64):
        super().__init__()
        self.embedding = _embedding()
        self.emb_drop = SpatialDropout1D(0.3)
        self.convs = nn.ModuleList(nn.Conv1d(EMBED_DIM, filters, k) for k in widths)
        self.drop1 = nn.Dropout(0.4)
        self.hidden = nn.Linear(filters * len(widths), 64)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        e = self.emb_drop(self.embedding(x)).transpose(1, 2)  # (B, D, T)
        pooled = [torch.relu(conv(e)).amax(dim=2) for conv in self.convs]
        h = torch.relu(self.hidden(self.drop1(torch.cat(pooled, dim=1))))
        return self.out(self.drop2(h)).squeeze(-1)

    def l2_params(self):
        return [self.hidden.weight]


class BiLSTM(nn.Module):
    """Bidirectional LSTM over the (unpadded) review.

    Embedding -> SpatialDropout(0.3) -> Dropout(0.3) on LSTM inputs ->
    BiLSTM(64 units per direction) -> concat final states (128) ->
    Dense(64, ReLU) -> Dropout(0.4) -> Dense(1)
    Padding is excluded with packed sequences, so the forward state is
    taken at the last real word and the backward state at the first word.
    """

    def __init__(self, units=64):
        super().__init__()
        self.embedding = _embedding()
        self.emb_drop = SpatialDropout1D(0.3)
        self.in_drop = nn.Dropout(0.3)
        self.lstm = nn.LSTM(EMBED_DIM, units, batch_first=True, bidirectional=True)
        self.hidden = nn.Linear(2 * units, 64)
        self.drop = nn.Dropout(0.4)
        self.out = nn.Linear(64, 1)

    def forward(self, x):
        lengths = (x != 0).sum(1).clamp(min=1).cpu()
        e = self.in_drop(self.emb_drop(self.embedding(x)))
        packed = nn.utils.rnn.pack_padded_sequence(
            e, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h = torch.cat([h_n[0], h_n[1]], dim=1)  # forward, backward
        return self.out(self.drop(torch.relu(self.hidden(h)))).squeeze(-1)

    def l2_params(self):
        return [self.hidden.weight]


# name -> (class, training configuration selected in Phase 2)
CONFIGS = {
    "mlp": (MLP, dict(lr=1e-3, batch_size=64, max_epochs=20, patience=3)),
    "cnn": (CNN1D, dict(lr=1e-3, batch_size=512, max_epochs=12, patience=2)),
    "bilstm": (BiLSTM, dict(lr=5e-4, batch_size=64, max_epochs=20, patience=3)),
}
