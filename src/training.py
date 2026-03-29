"""
training.py
-----------
Training loop for the LSTMWidth model with:
- Adam optimiser and ReduceLROnPlateau scheduler;
- gradient clipping (max norm = 1.0) for stability;
- early stopping based on validation loss;
- automatic saving of the best model weights.
"""

import os
import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    DEVICE, NUM_EPOCHS, LEARNING_RATE, PATIENCE, RESULTS_DIR,
)


def train(
    model      : nn.Module,
    dl_train   : DataLoader,
    dl_val     : DataLoader,
) -> Dict[str, List[float]]:
    """Train the model and return the loss history.

    At each epoch, the MSE loss is computed on the training and validation sets.
    The best model (minimum validation loss) is saved to the results directory.
    Training stops early if the validation loss does not improve for `PATIENCE`
    consecutive epochs.

    Parameters
    ----------
    model : nn.Module
        The model to train (LSTMWidth).
    dl_train : DataLoader
        Training set DataLoader.
    dl_val : DataLoader
        Validation set DataLoader.

    Returns
    -------
    history : dict
        Dictionary with keys 'train_loss' and 'val_loss',
        each containing a list of values per epoch.
    """
    criterion   = nn.MSELoss()
    optimizer   = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler   = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=4
    )

    best_loss     = float('inf')
    patience_count = 0
    best_state    = None
    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': []}

    print()
    print(f'--- Training ({NUM_EPOCHS} epochs max)')
    print()

    for epoch in range(NUM_EPOCHS):
        # --- Training phase ---
        model.train()
        total_loss, total = 0.0, 0

        for seqs, widths, metas, lengths in dl_train:
            seqs   = seqs.to(DEVICE)
            widths = widths.to(DEVICE)
            metas  = metas.to(DEVICE)
            optimizer.zero_grad()
            preds = model(seqs, lengths, metas)
            loss  = criterion(preds, widths)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * seqs.size(0)
            total      += seqs.size(0)

        train_loss = total_loss / total

        # --- Validation phase ---
        model.eval()
        total_val_loss, total_val = 0.0, 0

        with torch.no_grad():
            for seqs, widths, metas, lengths in dl_val:
                seqs   = seqs.to(DEVICE)
                widths = widths.to(DEVICE)
                metas  = metas.to(DEVICE)
                preds  = model(seqs, lengths, metas)
                total_val_loss += criterion(preds, widths).item() * seqs.size(0)
                total_val      += seqs.size(0)

        val_loss = total_val_loss / total_val

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1:2d}/{NUM_EPOCHS}] | Loss: {train_loss:.6f}/{val_loss:.6f}')

        # --- Early stopping ---
        if val_loss < best_loss:
            best_loss      = val_loss
            patience_count = 0
            best_state     = copy.deepcopy(model.state_dict())
            print(f'-- Best model (val loss: {val_loss:.6f})')
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f'-- Early stopping at epoch {epoch + 1}')
                break

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(RESULTS_DIR, 'lstm_width_model.pth'))

    return history
