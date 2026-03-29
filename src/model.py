"""
model.py
--------
Bidirectional LSTM architecture for magnetic pipe width regression.

The model combines:
- A bidirectional LSTM that processes the centred 1D intensity profile;
- A MLP regressor that fuses the LSTM hidden state with metadata
  (active pixel count, image height, image width).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMWidth(nn.Module):
    """Bidirectional LSTM for pipe width regression.

    Parameters
    ----------
    hidden_size : int
        Hidden state size per LSTM direction.
    num_layers : int
        Number of stacked LSTM layers.
    meta_size : int
        Dimension of the metadata vector (active_pixels, height, width).
    """

    def __init__(
        self,
        hidden_size : int = 64,
        num_layers  : int = 2,
        meta_size   : int = 3,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = 1,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = 0.3,
            bidirectional = True,
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_size * 2 + meta_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, sequences: Tensor, lengths: Tensor, meta: Tensor) -> Tensor:
        """Predict the normalised pipe width from a sequence and metadata.

        Parameters
        ----------
        sequences : Tensor
            Padded sequences of shape (batch, L_max, 1).
        lengths : Tensor
            Actual length of each sequence, shape (batch,).
        meta : Tensor
            Normalised metadata of shape (batch, meta_size).

        Returns
        -------
        Tensor
            Normalised predicted width, shape (batch,).
        """
        packed = pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)

        h        = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([h, meta], dim=1)
        return self.regressor(combined).squeeze(-1)
