"""
config.py
---------
Hyperparamètres d'entraînement et chemins vers les données et résultats.
L'appareil de calcul (CUDA, MPS ou CPU) est détecté automatiquement.
"""

import os
import torch

# ---------------------------------------------------------------------------
# Appareil de calcul
# ---------------------------------------------------------------------------
if torch.cuda.is_available():
    APPAREIL = torch.device('cuda')
elif torch.backends.mps.is_available():
    APPAREIL = torch.device('mps')
else:
    APPAREIL = torch.device('cpu')

# ---------------------------------------------------------------------------
# Chemins
# ---------------------------------------------------------------------------
RACINE_PROJET           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACINE_DONNEES          = os.path.join(os.path.dirname(RACINE_PROJET), 'skipperndt')
DOSSIER_DONNEES         = os.path.join(RACINE_DONNEES, 'data', 'nettoye')
DOSSIER_DONNEES_REELLES = os.path.join(RACINE_DONNEES, 'real_data')
FICHIER_CSV             = os.path.join(DOSSIER_DONNEES_REELLES, 'pipe_presence_width_detection_label.csv')
DOSSIER_RESULTATS       = os.path.join(RACINE_PROJET, 'resultats')

os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
os.makedirs(DOSSIER_DONNEES,   exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparamètres d'entraînement
# ---------------------------------------------------------------------------
NOMBRE_EPOQUES     : int   = 50      # Nombre maximum d'époques
TAILLE_LOT         : int   = 32      # Taille des mini-lots (batch size)
TAUX_APPRENTISSAGE : float = 0.001   # Learning rate initial (Adam)
PATIENCE           : int   = 10      # Patience pour l'early stopping
MAX_SEQ_LEN        : int   = 3000    # Longueur maximale de séquence (colonnes)
