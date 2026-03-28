"""
modele.py
---------
Architecture du modèle LSTM bidirectionnel pour l'estimation de la largeur
de la zone d'influence magnétique d'un pipe.

Le modèle combine :
- un LSTM bidirectionnel qui traite le profil d'intensité 1D centré sur le pipe ;
- un MLP de régression qui fusionne l'état caché du LSTM avec des métadonnées
  (nombre de pixels actifs, hauteur et largeur de l'image).
"""

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMWidth(nn.Module):
    """LSTM bidirectionnel pour la régression de la largeur d'un pipe.

    Parameters
    ----------
    hidden_size : int
        Taille de l'état caché par direction du LSTM.
    num_layers : int
        Nombre de couches LSTM empilées.
    meta_size : int
        Dimension du vecteur de métadonnées (nb_pixels, hauteur, largeur).
    """

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 2,
        meta_size: int = 3,
    ) -> None:
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

    def forward(self, sequences: Tensor, longueurs: Tensor, meta: Tensor) -> Tensor:
        """Calcule la largeur prédite à partir d'une séquence et de métadonnées.

        Parameters
        ----------
        sequences : Tensor
            Séquences paddées de forme (batch, L_max, 1).
        longueurs : Tensor
            Longueur réelle de chaque séquence, de forme (batch,).
        meta : Tensor
            Métadonnées normalisées de forme (batch, meta_size).

        Returns
        -------
        Tensor
            Largeur prédite normalisée, de forme (batch,).
        """
        packed = pack_padded_sequence(
            sequences, longueurs.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        # Concaténation des états cachés avant et arrière de la dernière couche
        h        = torch.cat([hidden[-2], hidden[-1]], dim=1)
        combined = torch.cat([h, meta], dim=1)
        return self.regresseur(combined).squeeze(-1)
