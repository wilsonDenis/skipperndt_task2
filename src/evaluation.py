"""
evaluation.py
-------------
Fonctions d'évaluation et de visualisation des résultats du modèle LSTMWidth.

Métriques calculées :
- MAE  (Mean Absolute Error) en mètres.
- RMSE (Root Mean Square Error) en mètres.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import APPAREIL, DOSSIER_RESULTATS


@torch.no_grad()
def evaluer(
    modele: nn.Module,
    chargeur: DataLoader,
    width_mean: float,
    width_std: float,
    largeurs_vraies: np.ndarray,
    nom: str = '',
) -> Tuple[float, float]:
    """Évalue le modèle sur un jeu de données et affiche les métriques.

    Les prédictions normalisées sont dénormalisées avant le calcul des métriques
    afin d'obtenir des erreurs en mètres.

    Parameters
    ----------
    modele : nn.Module
        Modèle entraîné à évaluer.
    chargeur : DataLoader
        DataLoader du jeu à évaluer (val ou test).
    width_mean : float
        Moyenne des largeurs brutes (pour dénormalisation).
    width_std : float
        Écart-type des largeurs brutes (pour dénormalisation).
    largeurs_vraies : np.ndarray
        Valeurs réelles de largeur en mètres, dans le même ordre que le chargeur.
    nom : str
        Nom du jeu affiché dans les logs (ex. 'Test Reel').

    Returns
    -------
    mae : float
        Erreur absolue moyenne en mètres.
    rmse : float
        Racine de l'erreur quadratique moyenne en mètres.
    """
    modele.eval()
    preds_all: List[float] = []

    for seqs, largs, metas, longs in chargeur:
        seqs  = seqs.to(APPAREIL)
        metas = metas.to(APPAREIL)
        preds = modele(seqs, longs, metas)
        preds_all.extend(preds.cpu().numpy())

    preds_m = np.array(preds_all) * width_std + width_mean
    mae     = float(np.mean(np.abs(preds_m - largeurs_vraies)))
    rmse    = float(np.sqrt(np.mean((preds_m - largeurs_vraies) ** 2)))

    print(f'comme resultat du [{nom}] MAE: {mae:.2f}m | RMSE: {rmse:.2f}m')
    for i in range(min(10, len(preds_m))):
        print(
            f'    Predit: {preds_m[i]:6.1f}m | '
            f'Vrai: {largeurs_vraies[i]:6.1f}m | '
            f'Err: {abs(preds_m[i] - largeurs_vraies[i]):5.1f}m'
        )

    return mae, rmse


def afficher_courbes(historique: Dict[str, List[float]]) -> None:
    """Génère et sauvegarde la courbe d'apprentissage (train vs val).

    Parameters
    ----------
    historique : dict
        Dictionnaire avec les clés 'perte_train' et 'perte_val'.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(historique['perte_train'], label='Train')
    plt.plot(historique['perte_val'],   label='Val')
    plt.title('LSTM Width - Perte')
    plt.xlabel('Epoque')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(DOSSIER_RESULTATS, 'courbes_lstm_width.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def afficher_comparaison(mae_val: float, mae_reel: float) -> None:
    """Affiche un tableau comparatif des approches.

    Parameters
    ----------
    mae_val : float
        MAE du modèle LSTM sur le jeu de validation.
    mae_reel : float
        MAE du modèle LSTM sur le jeu de test réel.
    """
    print()
    print('  COMPARAISON')
    print()
    print(f'  CNN Regression   : MAE = 14.91 m')
    print(f'  Mesure Physique  : MAE =  2.40 m')
    print(f'  LSTM (ce modele) : MAE = {mae_reel:.2f} m')
