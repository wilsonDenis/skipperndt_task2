"""
donnees.py
----------
Chargement, extraction et mise en forme des données pour le modèle LSTM.

Pipeline de traitement :
1. Lecture d'un fichier .npz et calcul de la norme des canaux magnétiques.
2. Localisation du centre de masse du signal pour extraire un profil 1D centré.
3. Construction du Dataset PyTorch et des DataLoaders (train / val / test).

Répartition des données :
- Synthétiques : 85 % train, 15 % val.
- Réelles      : 20 % train, 20 % val, 60 % test.
"""

import os
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from scipy.ndimage import center_of_mass

from src.config import (
    FICHIER_CSV, DOSSIER_DONNEES, DOSSIER_DONNEES_REELLES,
    MAX_SEQ_LEN, TAILLE_LOT,
)


def extraire_sequence(
    chemin_fichier: str,
) -> Tuple[np.ndarray, List[float]]:
    """Extrait un profil d'intensité 1D centré sur le pipe depuis un fichier .npz.

    Le profil est calculé en faisant la moyenne de 5 lignes autour du centre
    de masse du signal magnétique normalisé. Cette approche préserve l'information
    d'intensité nulle (zone du pipe) contrairement à un tri par intensité.

    Parameters
    ----------
    chemin_fichier : str
        Chemin vers le fichier .npz contenant les données magnétiques.

    Returns
    -------
    profil : np.ndarray
        Tableau 1D de forme (L,) avec L <= MAX_SEQ_LEN.
    meta : list[float]
        Métadonnées [nb_pixels_actifs, hauteur, largeur].
    """
    data    = np.load(chemin_fichier)
    key     = 'data' if 'data' in data.files else data.files[0]
    mat     = data[key].astype(np.float64)
    mat     = np.nan_to_num(mat, nan=0.0)

    norme   = np.abs(mat) if mat.ndim == 2 else np.sqrt(np.sum(mat ** 2, axis=2))
    h, w    = norme.shape

    if norme.max() == 0:
        return np.zeros(10, dtype=np.float32), [0.0, float(h), float(w)]

    norme_n = (norme / norme.max()).astype(np.float32)

    # Profil centré : 5 lignes autour du centre de masse du signal
    centre_y, _ = center_of_mass(norme_n)
    centre_y    = int(np.clip(centre_y, 2, h - 3))
    valeurs     = norme_n[centre_y - 2 : centre_y + 3, :].mean(axis=0)

    if len(valeurs) > MAX_SEQ_LEN:
        indices = np.linspace(0, len(valeurs) - 1, MAX_SEQ_LEN, dtype=int)
        valeurs = valeurs[indices]

    nb_pix = float((norme_n > 0).sum())
    return valeurs.astype(np.float32), [nb_pix, float(h), float(w)]


def precomputer_donnees() -> Tuple[
    List[Tensor], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Charge et prépare l'ensemble des données (synthétiques + réelles).

    Lit le fichier CSV de labels, extrait les séquences depuis les fichiers .npz
    et normalise les largeurs et métadonnées.

    Returns
    -------
    sequences : list[Tensor]
        Liste de tenseurs de forme (L, 1) pour chaque exemple.
    larg_norm : np.ndarray
        Largeurs normalisées (z-score).
    largeurs : np.ndarray
        Largeurs brutes en mètres.
    metas : np.ndarray
        Métadonnées normalisées de forme (N, 3).
    est_reel : np.ndarray
        Masque booléen True si la donnée est réelle.
    width_mean : float
        Moyenne des largeurs brutes.
    width_std : float
        Écart-type des largeurs brutes.
    """
    print(f'Lecture du CSV : {FICHIER_CSV}')
    df      = pd.read_csv(FICHIER_CSV, sep=';', keep_default_na=False)
    df_pipe = df[df['label'] == 1].copy()

    sequences: List[Tensor] = []
    largeurs:  List[float]  = []
    metas:     List[List[float]] = []
    est_reel:  List[bool]   = []

    print(f'  Extraction des sequences ({len(df_pipe)} fichiers)...')
    compteur = 0

    for _, row in df_pipe.iterrows():
        nom = row['field_file']
        try:
            width_val = float(row['width_m'])
        except (ValueError, TypeError):
            continue

        chemin = (
            os.path.join(DOSSIER_DONNEES_REELLES, nom)
            if nom.startswith('real_data')
            else os.path.join(DOSSIER_DONNEES, 'avec_fourreau', nom)
        )
        if not os.path.exists(chemin):
            continue

        seq, meta = extraire_sequence(chemin)
        sequences.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(-1))
        largeurs.append(width_val)
        metas.append(meta)
        est_reel.append(nom.startswith('real_data'))

        compteur += 1
        if compteur % 200 == 0:
            print(f'    {compteur} fichiers traites...')

    largeurs_arr = np.array(largeurs, dtype=np.float32)
    metas_arr    = np.array(metas,    dtype=np.float32)
    est_reel_arr = np.array(est_reel)

    meta_mean = metas_arr.mean(axis=0)
    meta_std  = metas_arr.std(axis=0) + 1e-8
    metas_arr = (metas_arr - meta_mean) / meta_std

    width_mean = float(largeurs_arr.mean())
    width_std  = float(largeurs_arr.std())
    larg_norm  = (largeurs_arr - width_mean) / width_std

    print(f'  {compteur} sequences extraites.')
    print(f'  Largeur : mean={width_mean:.2f}m  std={width_std:.2f}m')
    print(f'  Synthetiques: {(~est_reel_arr).sum()}  |  Reels: {est_reel_arr.sum()}')

    return sequences, larg_norm, largeurs_arr, metas_arr, est_reel_arr, width_mean, width_std


class DatasetLSTM(Dataset):
    """Dataset PyTorch pour les séquences LSTM.

    Parameters
    ----------
    sequences : list[Tensor]
        Séquences d'entrée de forme (L, 1).
    largeurs : np.ndarray
        Largeurs normalisées cibles.
    metas : np.ndarray
        Métadonnées normalisées de forme (N, 3).
    """

    def __init__(
        self,
        sequences: List[Tensor],
        largeurs: np.ndarray,
        metas: np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.largeurs  = largeurs
        self.metas     = metas

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        return (
            self.sequences[idx],
            torch.tensor(self.largeurs[idx], dtype=torch.float32),
            torch.tensor(self.metas[idx],    dtype=torch.float32),
            len(self.sequences[idx]),
        )


def collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fonction de collation : padde les séquences à la longueur maximale du lot.

    Parameters
    ----------
    batch : list
        Liste de tuples (séquence, largeur, meta, longueur).

    Returns
    -------
    Tuple contenant séquences paddées, largeurs, metas et longueurs.
    """
    sequences, largeurs, metas, longueurs = zip(*batch)
    longueurs_t      = torch.tensor(longueurs, dtype=torch.long)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.stack(largeurs), torch.stack(metas), longueurs_t


def creer_dataloaders(
    sequences: List[Tensor],
    larg_norm: np.ndarray,
    larg_brut: np.ndarray,
    metas: np.ndarray,
    est_reel: np.ndarray,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Crée les DataLoaders train / val / test avec mélange réel + synthétique.

    Répartition :
    - Synthétiques : 85 % train, 15 % val.
    - Réelles      : 20 % train, 20 % val, 60 % test.

    Parameters
    ----------
    sequences : list[Tensor]
        Toutes les séquences.
    larg_norm : np.ndarray
        Largeurs normalisées.
    larg_brut : np.ndarray
        Largeurs brutes (non utilisées ici, passées par cohérence d'interface).
    metas : np.ndarray
        Métadonnées normalisées.
    est_reel : np.ndarray
        Masque booléen indiquant les données réelles.

    Returns
    -------
    dl_train, dl_val, dl_reel : DataLoader
        Chargeurs de données pour l'entraînement, la validation et le test.
    idx_val : np.ndarray
        Indices (dans le tableau global) des exemples de validation.
    idx_reel_test : np.ndarray
        Indices des exemples réels réservés au test.
    """
    idx_synth = np.where(~est_reel)[0]
    idx_reel  = np.where(est_reel)[0]

    # Split synthétiques : 85 % train, 15 % val
    idx_train_s, idx_val_s = train_test_split(idx_synth, test_size=0.15, random_state=42)

    # Split réels : 60 % test, 20 % train, 20 % val
    idx_reel_test, idx_reel_tv   = train_test_split(idx_reel, test_size=0.40, random_state=42)
    idx_reel_train, idx_reel_val = train_test_split(idx_reel_tv, test_size=0.50, random_state=42)

    idx_train = np.concatenate([idx_train_s, idx_reel_train])
    idx_val   = np.concatenate([idx_val_s,   idx_reel_val])

    ds_train = DatasetLSTM(
        [sequences[i] for i in idx_train], larg_norm[idx_train], metas[idx_train])
    ds_val   = DatasetLSTM(
        [sequences[i] for i in idx_val],   larg_norm[idx_val],   metas[idx_val])
    ds_reel  = DatasetLSTM(
        [sequences[i] for i in idx_reel_test], larg_norm[idx_reel_test], metas[idx_reel_test])

    dl_train = DataLoader(ds_train, batch_size=TAILLE_LOT, shuffle=True,  collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val,   batch_size=TAILLE_LOT, shuffle=False, collate_fn=collate_fn)
    dl_reel  = DataLoader(ds_reel,  batch_size=TAILLE_LOT, shuffle=False, collate_fn=collate_fn)

    print(f'\n  Train: {len(ds_train)} (dont {len(idx_reel_train)} reels)'
          f'  |  Val: {len(ds_val)} (dont {len(idx_reel_val)} reels)'
          f'  |  Test reel: {len(ds_reel)}')

    return dl_train, dl_val, dl_reel, idx_val, idx_reel_test
