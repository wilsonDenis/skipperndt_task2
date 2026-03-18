# Tache 2 - Estimation de la Largeur de Zone Magnetique (LSTM)

Estimation de la largeur effective de la zone d'influence magnetique a partir
de cartes de champ magnetique 4 canaux, en utilisant un **LSTM bidirectionnel**.

## Probleme

**Type** : Regression
**Entree** : images TIF/NPZ multicanaux (4 canaux : Bx, By, Bz, Norme)
**Sortie** : largeur en metres (plage : 5 a 80 m)
**Objectif** : MAE < 1 m

## Approche LSTM

Contrairement au CNN qui redimensionne l'image a 224x224 (perte d'echelle),
le LSTM travaille sur une **sequence 1D** extraite de la carte magnetique :

1. Calculer la norme des 4 canaux
2. Normaliser entre 0 et 1
3. Masquer les pixels sous 10% d'intensite (peu informatifs)
4. Trier par intensite decroissante -> sequence de longueur variable
5. Enrichir avec 3 meta-features : nb pixels actifs, hauteur, largeur de l'image

Le LSTM traite cette sequence et predit la largeur. Avantage : l'information
d'echelle physique (1 pixel = 20 cm) est preservee.

## Architecture

```
Sequence (L, 1)  +  Meta (3,)
        |
   LSTM bidirectionnel
   hidden_size=64, num_layers=2, dropout=0.3
        |
   Concatenation etat cache [avant | arriere] + meta
   Taille : 64*2 + 3 = 131
        |
   MLP : Linear(131, 64) -> ReLU -> Dropout(0.3)
      -> Linear(64, 32)  -> ReLU
      -> Linear(32, 1)
        |
   Largeur predite (metres)
```

## Structure du projet

```
tache2_lstm/
├── main.py                  # Point d'entree : charge, entraine, evalue
├── src/
│   ├── config.py            # Hyperparametres et chemins
│   ├── donnees.py           # Extraction sequences + Dataset + DataLoaders
│   ├── modele.py            # Architecture LSTMWidth
│   ├── entrainement.py      # Boucle d'entrainement + early stopping
│   └── evaluation.py        # Metriques MAE/RMSE + graphiques
├── data/
│   └── nettoye/
│       └── avec_fourreau/   # Fichiers .npz synthetiques (a placer ici)
├── real_data/               # Fichiers .npz reels + CSV de labels (a placer ici)
└── resultats/               # Genere automatiquement
    ├── modele_lstm_width.pth
    └── courbes_lstm_width.png
```

## Donnees requises

Les donnees ne sont pas incluses. Placer dans les dossiers :

```
data/nettoye/avec_fourreau/   <- fichiers .npz synthetiques avec fourreau
real_data/                    <- fichiers .npz reels + pipe_presence_width_detection_label.csv
```

Le CSV doit avoir les colonnes : `field_file`, `label`, `width_m`

## Installation

```bash
pip install torch torchvision numpy pandas matplotlib scikit-learn
```

## Utilisation

```bash
python main.py
```

## Hyperparametres

| Parametre        | Valeur | Description                              |
|------------------|:------:|------------------------------------------|
| NOMBRE_EPOQUES   | 50     | Maximum d'epoques                        |
| TAILLE_LOT       | 32     | Batch size                               |
| TAUX_APPRENTISSAGE | 0.001 | Learning rate (Adam)                    |
| PATIENCE         | 10     | Early stopping                           |
| SEUIL_MASK       | 0.10   | Seuil d'intensite pour garder un pixel   |
| MAX_SEQ_LEN      | 3000   | Longueur max de sequence                 |
| hidden_size      | 64     | Taille etat cache LSTM (par direction)   |
| num_layers       | 2      | Nombre de couches LSTM empilees          |

## Resultats

Evaluation sur 1751 fichiers (1700 synthetiques, 51 reels).

| Jeu de donnees    | MAE     | RMSE    |
|-------------------|:-------:|:-------:|
| Val Synthetique   | 10.86 m | 19.89 m |
| Test Reel         |  4.49 m |  6.37 m |

### Comparaison des approches

| Modele          | MAE (Test Reel) |
|-----------------|:---------------:|
| CNN Regression  | 14.91 m         |
| LSTM (ce modele)|  4.49 m         |
| Mesure Physique |  2.40 m         |

Le LSTM reduit l'erreur du CNN de 70% et se rapproche de la mesure physique directe.
