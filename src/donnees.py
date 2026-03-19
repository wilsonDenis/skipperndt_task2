import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

from src.config import (
    FICHIER_CSV, DOSSIER_DONNEES, DOSSIER_DONNEES_REELLES,
    MAX_SEQ_LEN, TAILLE_LOT,
)


def extraire_sequence(chemin_fichier):
    data = np.load(chemin_fichier)
    key  = 'data' if 'data' in data.files else data.files[0]
    mat  = data[key].astype(np.float64)
    mat  = np.nan_to_num(mat, nan=0.0)

    if mat.ndim == 2:
        norme = np.abs(mat)
    else:
        norme = np.sqrt(np.sum(mat ** 2, axis=2))

    h, w = norme.shape

    if norme.max() == 0:
        return np.zeros(10, dtype=np.float32), [0.0, float(h), float(w)]

    norme_n = (norme / norme.max()).astype(np.float32)

    # Profil moyen par colonne : capture la variation spatiale sur toute la largeur
    valeurs = norme_n.mean(axis=0)

    if len(valeurs) > MAX_SEQ_LEN:
        indices = np.linspace(0, len(valeurs) - 1, MAX_SEQ_LEN, dtype=int)
        valeurs = valeurs[indices]

    nb_pix = float(norme_n.size)
    return valeurs, [nb_pix, float(h), float(w)]


def precomputer_donnees():
    print(f'Lecture du CSV : {FICHIER_CSV}')
    df      = pd.read_csv(FICHIER_CSV, sep=';', keep_default_na=False)
    df_pipe = df[df['label'] == 1].copy()

    sequences = []
    largeurs  = []
    metas     = []
    est_reel  = []

    print(f'  Extraction des sequences ({len(df_pipe)} fichiers)...')
    compteur = 0

    for _, row in df_pipe.iterrows():
        nom = row['field_file']
        try:
            width_val = float(row['width_m'])
        except (ValueError, TypeError):
            continue

        if nom.startswith('real_data'):
            chemin = os.path.join(DOSSIER_DONNEES_REELLES, nom)
        else:
            chemin = os.path.join(DOSSIER_DONNEES, 'avec_fourreau', nom)

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

    largeurs = np.array(largeurs, dtype=np.float32)
    metas    = np.array(metas,    dtype=np.float32)
    est_reel = np.array(est_reel)

    meta_mean = metas.mean(axis=0)
    meta_std  = metas.std(axis=0) + 1e-8
    metas     = (metas - meta_mean) / meta_std

    width_mean  = largeurs.mean()
    width_std   = largeurs.std()
    larg_norm   = (largeurs - width_mean) / width_std

    print(f'  {compteur} sequences extraites.')
    print(f'  Largeur : mean={width_mean:.2f}m  std={width_std:.2f}m')
    print(f'  Synthetiques: {(~est_reel).sum()}  |  Reels: {est_reel.sum()}')

    return sequences, larg_norm, largeurs, metas, est_reel, width_mean, width_std


class DatasetLSTM(Dataset):
    def __init__(self, sequences, largeurs, metas):
        self.sequences = sequences
        self.largeurs  = largeurs
        self.metas     = metas

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            torch.tensor(self.largeurs[idx], dtype=torch.float32),
            torch.tensor(self.metas[idx],    dtype=torch.float32),
            len(self.sequences[idx]),
        )


def collate_fn(batch):
    sequences, largeurs, metas, longueurs = zip(*batch)
    longueurs        = torch.tensor(longueurs, dtype=torch.long)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.stack(largeurs), torch.stack(metas), longueurs


def creer_dataloaders(sequences, larg_norm, larg_brut, metas, est_reel):
    idx_synth = np.where(~est_reel)[0]
    idx_reel  = np.where(est_reel)[0]

    # Split synthétiques : 85% train, 15% val
    idx_train_s, idx_val_s = train_test_split(idx_synth, test_size=0.15, random_state=42)

    # Split réels : 60% test, 20% train, 20% val
    idx_reel_test, idx_reel_tv = train_test_split(idx_reel, test_size=0.40, random_state=42)
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
