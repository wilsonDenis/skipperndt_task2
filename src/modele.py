import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMWidth(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2, meta_size=3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3,
            bidirectional=True,
        )
        self.regresseur = nn.Sequential(
            nn.Linear(hidden_size * 2 + meta_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequences, longueurs, meta):
        packed = pack_padded_sequence(
            sequences, longueurs.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        h        = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([h, meta], dim=1)
        return self.regresseur(combined).squeeze(-1)
