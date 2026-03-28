# Tâche 2 — Estimation de la Largeur de Zone Magnétique (LSTM)

Estimation de la largeur effective de la zone d'influence magnétique d'un pipe
à partir de cartes de champ magnétique 4 canaux, en utilisant un **LSTM bidirectionnel**.

## Problème

| Propriété | Valeur |
|-----------|--------|
| Type | Régression supervisée |
| Entrée | Fichiers `.npz` multicanaux (4 canaux : Bx, By, Bz, Norme) |
| Sortie | Largeur en mètres (plage : 5 à 80 m) |
| Objectif | MAE < 1 m |

## Approche LSTM

Contrairement au CNN qui redimensionne l'image à 224×224 (perte d'échelle physique),
le LSTM travaille sur un **profil 1D centré** extrait de la carte magnétique :

1. Calcul de la norme des 4 canaux magnétiques
2. Normalisation entre 0 et 1
3. Localisation du centre de masse du signal (scipy `center_of_mass`)
4. Extraction d'une tranche de 5 lignes autour de ce centre
5. Moyenne colonne par colonne → séquence 1D de longueur variable
6. Enrichissement avec 3 métadonnées : nb pixels actifs, hauteur, largeur image

L'information d'échelle physique (1 pixel ≈ 20 cm) est ainsi préservée.

## Architecture

```
Séquence (L, 1)  +  Meta (3,)
        |
   LSTM bidirectionnel
   hidden_size=64, num_layers=2, dropout=0.3
        |
   Concaténation états cachés [avant | arrière] + meta
   Taille : 64×2 + 3 = 131
        |
   MLP : Linear(131, 64) → ReLU → Dropout(0.3)
      → Linear(64, 32)   → ReLU
      → Linear(32, 1)
        |
   Largeur prédite (mètres)
```

## Structure du projet

```
tache2_lstm/
├── main.py                  # Point d'entrée : charge, entraîne, évalue
├── requirements.txt         # Dépendances Python
├── src/
│   ├── config.py            # Hyperparamètres et chemins
│   ├── donnees.py           # Extraction séquences + Dataset + DataLoaders
│   ├── modele.py            # Architecture LSTMWidth
│   ├── entrainement.py      # Boucle d'entraînement + early stopping
│   └── evaluation.py        # Métriques MAE/RMSE + graphiques
└── resultats/               # Généré automatiquement
    ├── modele_lstm_width.pth
    └── courbes_lstm_width.png
```

## Données requises

Les données ne sont pas incluses dans ce dépôt. Le projet attend la structure suivante
**en dehors** du dossier `tache2_lstm/`, dans un dossier frère `skipperndt/` :

```
skipperndt/
├── data/nettoye/avec_fourreau/   ← fichiers .npz synthétiques
└── real_data/
    ├── pipe_presence_width_detection_label.csv
    └── real_data_*.npz           ← fichiers .npz réels
```

Le CSV doit contenir les colonnes : `field_file`, `label`, `width_m`.

## Installation

```bash
pip install -r requirements.txt
```

Ou manuellement :

```bash
pip install torch numpy pandas matplotlib scikit-learn scipy
```

## Utilisation

```bash
python main.py
```

Le script affiche les métriques à chaque époque et sauvegarde dans `resultats/` :
- `modele_lstm_width.pth` — poids du meilleur modèle
- `courbes_lstm_width.png` — courbe d'apprentissage train/val

## Hyperparamètres

| Paramètre          | Valeur | Description                            |
|--------------------|:------:|----------------------------------------|
| NOMBRE_EPOQUES     | 50     | Maximum d'époques                      |
| TAILLE_LOT         | 32     | Batch size                             |
| TAUX_APPRENTISSAGE | 0.001  | Learning rate initial (Adam)           |
| PATIENCE           | 10     | Early stopping                         |
| MAX_SEQ_LEN        | 3000   | Longueur max de séquence (colonnes)    |
| hidden_size        | 64     | Taille état caché LSTM (par direction) |
| num_layers         | 2      | Nombre de couches LSTM empilées        |

## Répartition des données

| Ensemble      | Données synthétiques | Données réelles |
|---------------|:--------------------:|:---------------:|
| Entraînement  | 85 %                 | 20 %            |
| Validation    | 15 %                 | 20 %            |
| Test          | —                    | 60 %            |

## Résultats

| Jeu de données    | MAE     | RMSE    |
|-------------------|:-------:|:-------:|
| Val (Synth+Réel)  | —       | —       |
| Test Réel         | 4.49 m  | 6.37 m  |

### Comparaison des approches

| Modèle           | MAE (Test Réel) |
|------------------|:---------------:|
| CNN Régression   | 14.91 m         |
| **LSTM (ce modèle)** | **4.49 m**  |
| Mesure Physique  | 2.40 m          |

Le LSTM réduit l'erreur du CNN de **70 %** et se rapproche de la mesure physique directe.

## Contributeurs

- **Wilson Denis Bahun** — HETIC, promotion 2025
