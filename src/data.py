"""
data.py
-------
Data loading, feature extraction and DataLoader creation for the LSTM model.

Processing pipeline:
1. Load a .npz file and compute the norm of the magnetic channels.
2. Locate the center of mass (barycentre) of the signal to extract a centred 1D profile.
3. Build PyTorch Datasets and DataLoaders (train / val / test).

Data splits:
- Synthetic : 85 % train, 15 % val.
- Real      : 20 % train, 20 % val, 60 % test.
"""

import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from scipy.ndimage import center_of_mass

from src.config import (
    CSV_FILE, DATA_DIR, REAL_DATA_DIR,
    MAX_SEQ_LEN, BATCH_SIZE,
)


def extract_sequence(file_path: str) -> Tuple[np.ndarray, List[float]]:
    """Extract a centred 1D intensity profile from a .npz magnetic field file.

    The profile is computed by averaging 5 rows around the center of mass of
    the normalised magnetic signal. This approach preserves near-zero intensity
    values (at the pipe location) unlike threshold-based masking.

    Parameters
    ----------
    file_path : str
        Path to the .npz file containing the magnetic field data.

    Returns
    -------
    profile : np.ndarray
        1D array of shape (L,) with L <= MAX_SEQ_LEN.
    meta : list[float]
        Metadata [active_pixels, height, width].
    """
    data    = np.load(file_path)
    key     = 'data' if 'data' in data.files else data.files[0]
    mat     = data[key].astype(np.float64)
    mat     = np.nan_to_num(mat, nan=0.0)

    norm    = np.abs(mat) if mat.ndim == 2 else np.sqrt(np.sum(mat ** 2, axis=2))
    h, w    = norm.shape

    if norm.max() == 0:
        return np.zeros(10, dtype=np.float32), [0.0, float(h), float(w)]

    norm_n  = (norm / norm.max()).astype(np.float32)

    # Centred profile: 5 rows around the center of mass of the signal
    cy, _   = center_of_mass(norm_n)
    cy      = int(np.clip(cy, 2, h - 3))
    profile = norm_n[cy - 2 : cy + 3, :].mean(axis=0)

    if len(profile) > MAX_SEQ_LEN:
        indices = np.linspace(0, len(profile) - 1, MAX_SEQ_LEN, dtype=int)
        profile = profile[indices]

    active_pixels = float((norm_n > 0).sum())
    return profile.astype(np.float32), [active_pixels, float(h), float(w)]


def precompute_data() -> Tuple[
    List[Tensor], np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float
]:
    """Load and prepare all data (synthetic + real).

    Reads the label CSV, extracts sequences from .npz files,
    and normalises widths and metadata.

    Returns
    -------
    sequences : list[Tensor]
        List of tensors of shape (L, 1) for each sample.
    widths_norm : np.ndarray
        Z-score normalised widths.
    widths : np.ndarray
        Raw widths in metres.
    metas : np.ndarray
        Normalised metadata of shape (N, 3).
    is_real : np.ndarray
        Boolean mask, True for real acquisition data.
    width_mean : float
        Mean of raw widths.
    width_std : float
        Standard deviation of raw widths.
    """
    print(f'Reading CSV: {CSV_FILE}')
    df      = pd.read_csv(CSV_FILE, sep=';', keep_default_na=False)
    df_pipe = df[df['label'] == 1].copy()

    sequences : List[Tensor]       = []
    widths    : List[float]        = []
    metas     : List[List[float]]  = []
    is_real   : List[bool]         = []

    print(f'  Extracting sequences ({len(df_pipe)} files)...')
    count = 0

    for _, row in df_pipe.iterrows():
        filename = row['field_file']
        try:
            width_val = float(row['width_m'])
        except (ValueError, TypeError):
            continue

        path = (
            os.path.join(REAL_DATA_DIR, filename)
            if filename.startswith('real_data')
            else os.path.join(DATA_DIR, 'avec_fourreau', filename)
        )
        if not os.path.exists(path):
            continue

        seq, meta = extract_sequence(path)
        sequences.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(-1))
        widths.append(width_val)
        metas.append(meta)
        is_real.append(filename.startswith('real_data'))

        count += 1
        if count % 200 == 0:
            print(f'    {count} files processed...')

    widths_arr   = np.array(widths,  dtype=np.float32)
    metas_arr    = np.array(metas,   dtype=np.float32)
    is_real_arr  = np.array(is_real)

    meta_mean    = metas_arr.mean(axis=0)
    meta_std     = metas_arr.std(axis=0) + 1e-8
    metas_arr    = (metas_arr - meta_mean) / meta_std

    width_mean   = float(widths_arr.mean())
    width_std    = float(widths_arr.std())
    widths_norm  = (widths_arr - width_mean) / width_std

    print(f'  {count} sequences extracted.')
    print(f'  Width: mean={width_mean:.2f}m  std={width_std:.2f}m')
    print(f'  Synthetic: {(~is_real_arr).sum()}  |  Real: {is_real_arr.sum()}')

    return sequences, widths_norm, widths_arr, metas_arr, is_real_arr, width_mean, width_std


class LSTMDataset(Dataset):
    """PyTorch Dataset for LSTM sequences.

    Parameters
    ----------
    sequences : list[Tensor]
        Input sequences of shape (L, 1).
    widths : np.ndarray
        Normalised target widths.
    metas : np.ndarray
        Normalised metadata of shape (N, 3).
    """

    def __init__(
        self,
        sequences : List[Tensor],
        widths    : np.ndarray,
        metas     : np.ndarray,
    ) -> None:
        self.sequences = sequences
        self.widths    = widths
        self.metas     = metas

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor, int]:
        return (
            self.sequences[idx],
            torch.tensor(self.widths[idx], dtype=torch.float32),
            torch.tensor(self.metas[idx],  dtype=torch.float32),
            len(self.sequences[idx]),
        )


def collate_fn(
    batch: List[Tuple[Tensor, Tensor, Tensor, int]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate function: pad sequences to the maximum length in the batch.

    Parameters
    ----------
    batch : list
        List of tuples (sequence, width, meta, length).

    Returns
    -------
    Tuple of padded sequences, widths, metas and lengths.
    """
    sequences, widths, metas, lengths = zip(*batch)
    lengths_t        = torch.tensor(lengths, dtype=torch.long)
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
    return sequences_padded, torch.stack(widths), torch.stack(metas), lengths_t


def create_dataloaders(
    sequences   : List[Tensor],
    widths_norm : np.ndarray,
    widths_raw  : np.ndarray,
    metas       : np.ndarray,
    is_real     : np.ndarray,
) -> Tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray]:
    """Create train / val / test DataLoaders mixing real and synthetic data.

    Splits:
    - Synthetic : 85 % train, 15 % val.
    - Real      : 20 % train, 20 % val, 60 % test.

    Parameters
    ----------
    sequences : list[Tensor]
        All sequences.
    widths_norm : np.ndarray
        Normalised widths.
    widths_raw : np.ndarray
        Raw widths (unused inside, kept for interface consistency).
    metas : np.ndarray
        Normalised metadata.
    is_real : np.ndarray
        Boolean mask indicating real data samples.

    Returns
    -------
    dl_train, dl_val, dl_test : DataLoader
        DataLoaders for training, validation and test.
    idx_val : np.ndarray
        Indices (in the global array) of validation samples.
    idx_real_test : np.ndarray
        Indices of real samples reserved for testing.
    """
    idx_synth = np.where(~is_real)[0]
    idx_real  = np.where(is_real)[0]

    # Synthetic split: 85 % train, 15 % val
    idx_train_s, idx_val_s     = train_test_split(idx_synth, test_size=0.15, random_state=42)

    # Real split: 60 % test, 20 % train, 20 % val
    idx_real_test, idx_real_tv = train_test_split(idx_real, test_size=0.40, random_state=42)
    idx_real_train, idx_real_val = train_test_split(idx_real_tv, test_size=0.50, random_state=42)

    idx_train = np.concatenate([idx_train_s, idx_real_train])
    idx_val   = np.concatenate([idx_val_s,   idx_real_val])

    ds_train = LSTMDataset(
        [sequences[i] for i in idx_train], widths_norm[idx_train], metas[idx_train])
    ds_val   = LSTMDataset(
        [sequences[i] for i in idx_val],   widths_norm[idx_val],   metas[idx_val])
    ds_test  = LSTMDataset(
        [sequences[i] for i in idx_real_test], widths_norm[idx_real_test], metas[idx_real_test])

    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    dl_val   = DataLoader(ds_val,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    dl_test  = DataLoader(ds_test,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    print(f'\n  Train: {len(ds_train)} ({len(idx_real_train)} real)'
          f'  |  Val: {len(ds_val)} ({len(idx_real_val)} real)'
          f'  |  Test (real): {len(ds_test)}')

    return dl_train, dl_val, dl_test, idx_val, idx_real_test
