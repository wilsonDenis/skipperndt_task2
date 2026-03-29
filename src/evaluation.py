"""
evaluation.py
-------------
Evaluation and visualisation functions for the LSTMWidth model.

Metrics computed:
- MAE  (Mean Absolute Error) in metres.
- RMSE (Root Mean Square Error) in metres.
"""

import os
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import DEVICE, RESULTS_DIR


@torch.no_grad()
def evaluate(
    model        : nn.Module,
    dataloader   : DataLoader,
    width_mean   : float,
    width_std    : float,
    true_widths  : np.ndarray,
    name         : str = '',
) -> Tuple[float, float]:
    """Evaluate the model on a dataset and print metrics.

    Normalised predictions are denormalised before computing metrics,
    so that errors are expressed in metres.

    Parameters
    ----------
    model : nn.Module
        Trained model to evaluate.
    dataloader : DataLoader
        DataLoader of the dataset to evaluate (val or test).
    width_mean : float
        Mean of raw widths (for denormalisation).
    width_std : float
        Standard deviation of raw widths (for denormalisation).
    true_widths : np.ndarray
        Ground-truth widths in metres, in the same order as the dataloader.
    name : str
        Dataset name displayed in logs (e.g. 'Real Test').

    Returns
    -------
    mae : float
        Mean Absolute Error in metres.
    rmse : float
        Root Mean Square Error in metres.
    """
    model.eval()
    all_preds: List[float] = []

    for seqs, _, metas, lengths in dataloader:
        seqs  = seqs.to(DEVICE)
        metas = metas.to(DEVICE)
        preds = model(seqs, lengths, metas)
        all_preds.extend(preds.cpu().numpy())

    preds_m = np.array(all_preds) * width_std + width_mean
    mae     = float(np.mean(np.abs(preds_m - true_widths)))
    rmse    = float(np.sqrt(np.mean((preds_m - true_widths) ** 2)))

    print(f'[{name}] MAE: {mae:.2f}m | RMSE: {rmse:.2f}m')
    for i in range(min(10, len(preds_m))):
        print(
            f'    Predicted: {preds_m[i]:6.1f}m | '
            f'True: {true_widths[i]:6.1f}m | '
            f'Error: {abs(preds_m[i] - true_widths[i]):5.1f}m'
        )

    return mae, rmse


def plot_curves(history: Dict[str, List[float]]) -> None:
    """Generate and save the learning curve (train vs val loss).

    Parameters
    ----------
    history : dict
        Dictionary with keys 'train_loss' and 'val_loss'.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'],   label='Val')
    plt.title('LSTM Width — Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(
        os.path.join(RESULTS_DIR, 'lstm_width_curves.png'),
        dpi=150, bbox_inches='tight',
    )
    plt.close()


def print_comparison(mae_real: float) -> None:
    """Print a comparison table of the different approaches.

    Parameters
    ----------
    mae_real : float
        LSTM model MAE on the real test set.
    """
    print()
    print('  COMPARISON')
    print()
    print(f'  CNN Regression    : MAE = 14.91 m')
    print(f'  Physical Measure  : MAE =  2.40 m')
    print(f'  LSTM (this model) : MAE = {mae_real:.2f} m')
