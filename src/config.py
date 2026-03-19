import torch
import os

if torch.cuda.is_available():
    APPAREIL = torch.device('cuda')
elif torch.backends.mps.is_available():
    APPAREIL = torch.device('mps')
else:
    APPAREIL = torch.device('cpu')

RACINE_PROJET           = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RACINE_DONNEES          = os.path.join(os.path.dirname(RACINE_PROJET), 'skipperndt')
DOSSIER_DONNEES         = os.path.join(RACINE_DONNEES, 'data', 'nettoye')
DOSSIER_DONNEES_REELLES = os.path.join(RACINE_DONNEES, 'real_data')
FICHIER_CSV             = os.path.join(DOSSIER_DONNEES_REELLES, 'pipe_presence_width_detection_label.csv')
DOSSIER_RESULTATS       = os.path.join(RACINE_PROJET, 'resultats')

os.makedirs(DOSSIER_RESULTATS, exist_ok=True)
os.makedirs(DOSSIER_DONNEES,   exist_ok=True)

NOMBRE_EPOQUES     = 50
TAILLE_LOT         = 32
TAUX_APPRENTISSAGE = 0.001
PATIENCE           = 10
MAX_SEQ_LEN        = 3000
