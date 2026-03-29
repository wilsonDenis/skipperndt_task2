# Task 2 — Magnetic Zone Width Estimation (Bidirectional LSTM)

> **Industry partner:** [Skipper NDT](https://skipperndt.com/) · **School:** HETIC — École du Numérique, Paris

Estimate the effective width of a pipe's magnetic influence zone from 4-channel magnetic field maps using a **bidirectional LSTM** with a centre-of-mass profile extraction strategy.

---

## Problem Statement

| Property | Value |
|----------|-------|
| Task | Supervised regression |
| Input | Multi-channel `.npz` files (4 channels: Bx, By, Bz, Norm) |
| Output | Width in metres (range: 5 – 80 m) |
| Target | MAE < 1 m |

---

## Approach

Unlike CNN-based models that resize images to a fixed 224×224 grid (losing physical scale), this LSTM operates on a **variable-length 1D profile** extracted from the magnetic map:

1. Compute the norm across all 4 magnetic channels
2. Normalise values between 0 and 1
3. Locate the **centre of mass** of the signal (`scipy.ndimage.center_of_mass`)
4. Extract a 5-row slice centred on that position
5. Average column-by-column → 1D sequence of variable length
6. Enrich with 3 metadata features: active pixel count, image height, image width

Since 1 pixel ≈ 20 cm, physical scale information is fully preserved.

---

## Model Architecture

```
Sequence (L, 1)  +  Metadata (3,)
        │
  Bidirectional LSTM
  hidden_size=64 · num_layers=2 · dropout=0.3
        │
  Concat [forward | backward] hidden states + metadata
  Size: 64×2 + 3 = 131
        │
  MLP regressor:
    Linear(131 → 64) → ReLU → Dropout(0.3)
    Linear(64  → 32) → ReLU
    Linear(32  →  1)
        │
  Predicted width (metres)
```

---

## Project Structure

```
tache2_lstm/
├── main.py              # Entry point: load data, train, evaluate
├── requirements.txt     # Python dependencies
├── src/
│   ├── config.py        # Hyperparameters and paths
│   ├── data.py          # Sequence extraction, Dataset, DataLoaders
│   ├── model.py         # LSTMWidth architecture
│   ├── training.py      # Training loop + early stopping
│   └── evaluation.py    # MAE / RMSE metrics + learning curves
└── results/             # Auto-created on first run
    ├── lstm_width_model.pth
    └── lstm_width_curves.png
```

---

## Prerequisites

- Python 3.9+
- pip

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/wilsonDenis/skipperndt_task2.git
cd skipperndt_task2
```

**2. (Recommended) Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
.venv\Scripts\activate         # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Data Setup

The dataset is **not included** in this repository (proprietary data from Skipper NDT).

Place the data in a sibling folder named `skipperndt/`, at the same level as `tache2_lstm/`:

```
parent_folder/
├── tache2_lstm/          ← this repository
└── skipperndt/
    ├── data/
    │   └── nettoye/
    │       └── avec_fourreau/   ← synthetic .npz files
    └── real_data/
        ├── pipe_presence_width_detection_label.csv
        └── real_data_*.npz      ← real .npz files
```

The CSV file must contain the columns: `field_file`, `label`, `width_m`.

---

## Usage

```bash
python main.py
```

The script will:
1. Load and preprocess synthetic and real data
2. Build train / validation / test DataLoaders
3. Train the model with early stopping
4. Print MAE and RMSE for each split
5. Save results to the `results/` folder

**Output files:**
| File | Description |
|------|-------------|
| `results/lstm_width_model.pth` | Best model weights |
| `results/lstm_width_curves.png` | Train / val learning curve |

---

## Hyperparameters

| Parameter | Value | Description |
|-----------|:-----:|-------------|
| NUM_EPOCHS | 50 | Maximum number of training epochs |
| BATCH_SIZE | 32 | Mini-batch size |
| LEARNING_RATE | 0.001 | Initial learning rate (Adam) |
| PATIENCE | 10 | Early stopping patience |
| MAX_SEQ_LEN | 3000 | Maximum sequence length (columns) |
| hidden_size | 64 | LSTM hidden state size (per direction) |
| num_layers | 2 | Number of stacked LSTM layers |

---

## Data Split

| Split | Synthetic data | Real data |
|-------|:--------------:|:---------:|
| Train | 85 % | 20 % |
| Validation | 15 % | 20 % |
| Test | — | 60 % |

Real data is included in training and validation to improve generalisation. The model is never evaluated on data it was trained on.

---

## Results

| Split | MAE | RMSE |
|-------|:---:|:----:|
| Val (Synth + Real) | — | — |
| Real Test | 4.49 m | 6.37 m |

### Comparison

| Model | MAE (Real Test) |
|-------|:---------------:|
| CNN Regression | 14.91 m |
| **LSTM — this model** | **4.49 m** |
| Physical Measurement | 2.40 m |

The LSTM reduces CNN error by **70 %**. With more annotated real data, it is expected to approach or match the physical measurement baseline.

---

## Contributors

| Name | Email |
|------|-------|
| AHMED Filali | ahmedfillali905@gmail.com |
| FOLLIVI Edem Roberto | robertfollivi49@gmail.com |
| MAFORIKAN Harald | haraldmaforikan@gmail.com |
| TAMBOU NGUEMO Franck Kevin | ktambou99@gmail.com |
| WILSON-BAHUN A. Denis | wilsonvry@gmail.com |
