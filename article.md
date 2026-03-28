# Estimation de la Largeur d'un Pipe Enfoui par Réseau de Neurones Récurrent

**Wilson Denis Bahun** — HETIC, Mars 2026

---

## Résumé

Ce travail s'inscrit dans le cadre du projet Skipper, qui vise à automatiser l'inspection de pipelines enfouis à partir de données de champ magnétique. Nous nous concentrons sur la **tâche d'estimation de la largeur** de la zone d'influence magnétique, un indicateur clé du diamètre effectif du pipe. Nous démontrons qu'un réseau LSTM bidirectionnel, entraîné sur un profil d'intensité centré sur le signal du pipe, surpasse significativement une approche CNN de référence, en réduisant l'erreur absolue moyenne de 14.91 m à **4.49 m** sur données réelles — soit une amélioration de **70 %** — tout en se rapprochant de la mesure physique directe (MAE = 2.40 m).

---

## 1. Introduction

La détection et la caractérisation de pipelines enfouis est un enjeu industriel et de sécurité majeur. Les opérateurs doivent régulièrement inspecter leurs réseaux pour détecter des anomalies, évaluer l'état des canalisations et planifier des interventions de maintenance. Les méthodes traditionnelles reposent sur des mesures physiques directes, coûteuses en temps et en main d'œuvre.

Les techniques magnétiques offrent une alternative non intrusive : un capteur déplacé en surface enregistre les perturbations du champ magnétique terrestre induites par les structures métalliques enfouies. Ces données, organisées sous forme de cartes 2D multi-canaux (Bx, By, Bz et la norme du vecteur B), contiennent l'information nécessaire pour caractériser le pipe, notamment sa position et sa largeur d'influence.

### Problème traité

Nous cherchons à estimer automatiquement la **largeur de la zone d'influence magnétique** (en mètres) à partir d'une carte magnétique. Il s'agit d'un problème de **régression supervisée** : l'entrée est une image 2D de dimensions variables, et la sortie est un scalaire compris entre 5 et 80 mètres.

### État de l'art

Plusieurs approches ont été explorées pour des problèmes similaires de détection d'objets enfouis :

- **Méthodes physiques directes** : calcul analytique à partir de la réponse magnétique théorique d'un cylindre. Précises mais nécessitent des conditions idéales.
- **Réseaux convolutifs (CNN)** : approche de référence dans ce projet. Le CNN redimensionne les images à 224×224, ce qui détruit l'information d'échelle physique (1 pixel ≈ 20 cm). MAE observée : 14.91 m.
- **Réseaux récurrents (LSTM)** : notre approche. Le LSTM traite une séquence 1D extraite de l'image, préservant l'échelle physique et la structure spatiale du signal.

---

## 2. Matériel et Méthodes

### 2.1 Données

Le jeu de données provient du projet Skipper et comprend deux types de fichiers :

- **Données synthétiques** : générées par simulation physique, couvrant diverses configurations (pipe droit/courbé, avec/sans fourreau, bruyant/propre). Environ **2 884 fichiers** avec labels de largeur.
- **Données réelles** : acquisitions terrain, **51 fichiers** `.npz` accompagnés d'un CSV de labels (`field_file`, `label`, `width_m`).

Chaque fichier `.npz` contient un tableau de forme `(H, W, 4)` en `float16` représentant les 4 canaux magnétiques. Les dimensions H et W sont variables selon les acquisitions (typiquement entre 197 et 3000 colonnes).

**Répartition utilisée :**

| Ensemble      | Synthétiques | Réelles |
|---------------|:------------:|:-------:|
| Entraînement  | 85 %         | 20 %    |
| Validation    | 15 %         | 20 %    |
| Test          | —            | 60 %    |

Le test est exclusivement constitué de données réelles afin d'évaluer la capacité de généralisation du modèle.

### 2.2 Extraction du profil d'intensité

L'approche centrale de notre méthode est l'extraction d'un **profil 1D** depuis la carte magnétique 2D, plutôt que de traiter l'image entière.

**Étapes :**

1. Calcul de la norme du vecteur magnétique : `||B|| = sqrt(Bx² + By² + Bz²)`.
2. Normalisation entre 0 et 1 par rapport au maximum de l'image.
3. Localisation du **centre de masse** du signal normalisé (fonction `scipy.ndimage.center_of_mass`) pour identifier la ligne centrale du pipe.
4. Extraction d'une tranche de **5 lignes** autour de ce centre et calcul de la moyenne colonne par colonne.
5. Si le profil dépasse 3000 colonnes, sous-échantillonnage uniforme.

Cette approche préserve l'information d'intensité nulle aux emplacements du pipe (contrairement à un filtrage par seuil), ce qui est essentiel pour la mesure de largeur.

**Métadonnées complémentaires :** nombre de pixels actifs (intensité > 0), hauteur H et largeur W de l'image, normalisées par z-score. Ces 3 features encodent l'échelle physique de l'acquisition.

### 2.3 Architecture du modèle

Le modèle `LSTMWidth` combine un encodeur récurrent et un régresseur MLP :

```
Séquence (L, 1)         Métadonnées (3,)
      |                        |
LSTM bidirectionnel            |
hidden=64, layers=2            |
dropout=0.3                    |
      |                        |
 état caché final ─────────────┘
  (64×2 = 128 dim)    concat → (131 dim)
      |
 Linear(131→64) → ReLU → Dropout(0.3)
 Linear(64→32)  → ReLU
 Linear(32→1)
      |
 Largeur prédite (normalisée)
```

Le LSTM bidirectionnel permet de traiter le profil dans les deux sens, capturant à la fois les transitions d'entrée et de sortie de la zone magnétique. L'absence d'activation sur la couche de sortie permet une régression non bornée.

**Nombre de paramètres :** ~120 000.

### 2.4 Entraînement

- **Fonction de perte** : MSE (Mean Squared Error) sur les largeurs normalisées.
- **Optimiseur** : Adam, learning rate initial = 0.001.
- **Scheduler** : ReduceLROnPlateau (facteur 0.5, patience 4 époques).
- **Gradient clipping** : norme maximale = 1.0 pour la stabilité.
- **Early stopping** : patience = 10 époques sur la perte de validation.
- **Batch size** : 32, avec padding des séquences de longueurs variables.

---

## 3. Résultats

### 3.1 Convergence

Le modèle converge généralement entre 20 et 40 époques. L'early stopping évite le sur-apprentissage, particulièrement important ici en raison du faible nombre de données réelles.

### 3.2 Métriques finales

| Jeu de données      | MAE (m) | RMSE (m) |
|---------------------|:-------:|:--------:|
| Validation (Synth+Réel) | ~10.86  | ~19.89   |
| **Test Réel**       | **4.49**| **6.37** |

L'écart entre la validation et le test réel s'explique par la distribution différente des données synthétiques et réelles. Le modèle généralise bien aux données réelles.

### 3.3 Comparaison des approches

| Approche           | MAE Test Réel | Amélioration vs CNN |
|--------------------|:-------------:|:-------------------:|
| CNN Régression     | 14.91 m       | référence           |
| **LSTM (ce modèle)** | **4.49 m** | **−70 %**           |
| Mesure Physique    | 2.40 m        | —                   |

Le LSTM surpasse le CNN d'un facteur 3.3 et réduit l'écart avec la mesure physique directe de manière significative. La différence résiduelle de ~2 m avec la mesure physique peut s'expliquer par :
- le faible nombre de données réelles d'entraînement (≈10 fichiers) ;
- la variabilité des conditions d'acquisition terrain non couverte par les données synthétiques.

---

## 4. Conclusion

Nous avons présenté une approche LSTM pour l'estimation de la largeur d'un pipe enfoui à partir de données magnétiques. L'extraction d'un profil 1D centré sur le signal, combinée à un réseau bidirectionnel, permet de préserver l'information d'échelle physique que le CNN détruisait par redimensionnement.

**Limites :** Le faible nombre de données réelles (51 fichiers) constitue le principal facteur limitant. L'entreprise Skipper a fourni un nouveau jeu de 4 715 fichiers synthétiques supplémentaires, mais sans annotation de largeur (`width_m`), rendant ce jeu inutilisable pour la régression en l'état.

**Pistes futures :**
- Obtenir les annotations `width_m` pour le nouveau jeu de données, ce qui multiplierait par ~2 le volume d'entraînement.
- Explorer des architectures Transformer, mieux adaptées aux longues séquences.
- Intégrer des techniques d'augmentation de données (symétries, bruit) pour améliorer la robustesse.

---

## 5. Ressources

- **Code source** : [https://github.com/wilsonDenis/skipperndt_task2](https://github.com/wilsonDenis/skipperndt_task2)
- **Données** : fournies par l'entreprise Skipper NDT (non publiques)

---

## 6. Bibliographie

- Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8), 1735–1780.
- Schuster, M., & Paliwal, K. K. (1997). Bidirectional recurrent neural networks. *IEEE Transactions on Signal Processing*, 45(11), 2673–2681.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR 2016*.
- Paszke, A. et al. (2019). PyTorch: An imperative style, high-performance deep learning library. *NeurIPS 2019*.
