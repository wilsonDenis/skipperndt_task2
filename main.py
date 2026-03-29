"""
main.py
-------
Entry point for Task 2 — Pipe width estimation using a bidirectional LSTM.

Usage:
    python main.py

Steps:
1. Load and preprocess data (synthetic + real).
2. Create DataLoaders (train / val / test).
3. Instantiate and train the LSTMWidth model.
4. Evaluate and display final metrics.
"""

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config     import DEVICE
from src.data       import precompute_data, create_dataloaders
from src.model      import LSTMWidth
from src.training   import train
from src.evaluation import evaluate, plot_curves, print_comparison


def main() -> None:
    """Run the full pipeline: data loading, training and evaluation."""
    print('  TASK 2: PIPE WIDTH ESTIMATION — BIDIRECTIONAL LSTM')

    sequences, widths_norm, widths_raw, metas, is_real, w_mean, w_std = \
        precompute_data()

    dl_train, dl_val, dl_test, idx_val, idx_test = \
        create_dataloaders(sequences, widths_norm, widths_raw, metas, is_real)

    model = LSTMWidth(hidden_size=64, num_layers=2, meta_size=3).to(DEVICE)
    print(f'  Model: {sum(p.numel() for p in model.parameters()):,} parameters')

    history = train(model, dl_train, dl_val)
    plot_curves(history)

    print()
    print('  EVALUATION')
    print()

    evaluate(model, dl_val,  w_mean, w_std, widths_raw[idx_val],  'Val (Synth+Real)')
    mae_real, _ = evaluate(model, dl_test, w_mean, w_std, widths_raw[idx_test], 'Real Test')

    print_comparison(mae_real)


if __name__ == '__main__':
    main()
