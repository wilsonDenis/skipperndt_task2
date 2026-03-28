"""
entrainement.py
---------------
Boucle d'entraînement du modèle LSTMWidth avec :
- optimiseur Adam et scheduler ReduceLROnPlateau ;
- gradient clipping (norme max = 1.0) pour la stabilité ;
- early stopping basé sur la perte de validation ;
- sauvegarde automatique du meilleur état du modèle.
"""

import os
import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    APPAREIL, NOMBRE_EPOQUES, TAUX_APPRENTISSAGE, PATIENCE, DOSSIER_RESULTATS,
)


def entrainer(
    modele: nn.Module,
    chargeur_train: DataLoader,
    chargeur_val: DataLoader,
) -> Dict[str, List[float]]:
    """Entraîne le modèle et retourne l'historique des pertes.

    À chaque époque, la perte MSE est calculée sur le train et la validation.
    Le meilleur modèle (perte val minimale) est sauvegardé dans le dossier
    résultats. L'entraînement s'arrête par early stopping si la perte de
    validation ne s'améliore pas pendant `PATIENCE` époques consécutives.

    Parameters
    ----------
    modele : nn.Module
        Le modèle à entraîner (LSTMWidth).
    chargeur_train : DataLoader
        DataLoader du jeu d'entraînement.
    chargeur_val : DataLoader
        DataLoader du jeu de validation.

    Returns
    -------
    historique : dict
        Dictionnaire avec les clés 'perte_train' et 'perte_val',
        chacune contenant une liste de valeurs par époque.
    """
    critere       = nn.MSELoss()
    optimiseur    = torch.optim.Adam(modele.parameters(), lr=TAUX_APPRENTISSAGE)
    planificateur = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiseur, factor=0.5, patience=4
    )

    meilleure_perte   = float('inf')
    compteur_patience = 0
    meilleur_etat     = None
    historique: Dict[str, List[float]] = {'perte_train': [], 'perte_val': []}

    print()
    print(f'---Entrainement ({NOMBRE_EPOQUES} epoques max)')
    print()

    for epoque in range(NOMBRE_EPOQUES):
        # --- Phase entraînement ---
        modele.train()
        total_perte, total = 0.0, 0

        for seqs, largs, metas, longs in chargeur_train:
            seqs  = seqs.to(APPAREIL)
            largs = largs.to(APPAREIL)
            metas = metas.to(APPAREIL)
            optimiseur.zero_grad()
            preds = modele(seqs, longs, metas)
            perte = critere(preds, largs)
            perte.backward()
            torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
            optimiseur.step()
            total_perte += perte.item() * seqs.size(0)
            total       += seqs.size(0)

        perte_train = total_perte / total

        # --- Phase validation ---
        modele.eval()
        total_pv, total_v = 0.0, 0

        with torch.no_grad():
            for seqs, largs, metas, longs in chargeur_val:
                seqs  = seqs.to(APPAREIL)
                largs = largs.to(APPAREIL)
                metas = metas.to(APPAREIL)
                preds = modele(seqs, longs, metas)
                total_pv += critere(preds, largs).item() * seqs.size(0)
                total_v  += seqs.size(0)

        perte_val = total_pv / total_v

        historique['perte_train'].append(perte_train)
        historique['perte_val'].append(perte_val)
        planificateur.step(perte_val)

        print(f'Epoque [{epoque+1:2d}/{NOMBRE_EPOQUES}] | Perte: {perte_train:.6f}/{perte_val:.6f}')

        # --- Early stopping ---
        if perte_val < meilleure_perte:
            meilleure_perte   = perte_val
            compteur_patience = 0
            meilleur_etat     = copy.deepcopy(modele.state_dict())
            print(f'--Meilleur modele (perte val: {perte_val:.6f})')
        else:
            compteur_patience += 1
            if compteur_patience >= PATIENCE:
                print(f'-- Early stopping a l\'epoque {epoque + 1}')
                break

    if meilleur_etat:
        modele.load_state_dict(meilleur_etat)
        torch.save(meilleur_etat, os.path.join(DOSSIER_RESULTATS, 'modele_lstm_width.pth'))

    return historique
