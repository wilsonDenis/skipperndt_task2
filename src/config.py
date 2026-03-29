"""
config.py
---------
Training hyperparameters and paths to data and results directories.
The compute device (CUDA, MPS or CPU) is detected automatically.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Compute device
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT     = os.path.join(os.path.dirname(PROJECT_ROOT), 'skipperndt')
DATA_DIR      = os.path.join(DATA_ROOT, 'data', 'nettoye')
REAL_DATA_DIR = os.path.join(DATA_ROOT, 'real_data')
CSV_FILE      = os.path.join(REAL_DATA_DIR, 'pipe_presence_width_detection_label.csv')
RESULTS_DIR   = os.path.join(PROJECT_ROOT, 'resultats')

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR,    exist_ok=True)

# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------
NUM_EPOCHS    : int   = 50      # Maximum number of training epochs
BATCH_SIZE    : int   = 32      # Mini-batch size
LEARNING_RATE : float = 0.001   # Initial learning rate (Adam)
PATIENCE      : int   = 10      # Early stopping patience (epochs)
MAX_SEQ_LEN   : int   = 3000    # Maximum profile length (columns)
