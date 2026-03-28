# Brugada Syndrome Detection from 12-Lead ECG

Automated classification of **Brugada syndrome** vs Normal using the PhysioNet [Brugada-HUCA dataset](https://physionet.org/content/brugada-huca/1.0.0/) (363 patients, 12-lead ECG @ 100 Hz).

---

## Dataset

| Property | Value |
|---|---|
| Source | PhysioNet `brugada-huca/1.0.0` |
| Records | 363 (69 Brugada, 294 Normal) |
| Signal | 12 leads × 1200 samples @ 100 Hz (12 sec) |
| Format | WFDB (`.dat` + `.hea`) + `metadata.csv` |
| Key diagnostic leads | V1, V2, V3 (coved ST-elevation) |

---

## Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download dataset

```bash
python src/download.py
```

This uses `wfdb.dl_database()` (no `wget` required) and saves to `data/raw/`.

### 3. Run notebooks in order

```bash
jupyter notebook
```

| Notebook | Description |
|---|---|
| `01_eda.ipynb` | Exploratory data analysis — class distribution, ECG visualisation, signal quality |
| `02_preprocessing.ipynb` | Filter pipeline → normalisation → stratified splits → save `data/processed/*.npz` |
| `03_training.ipynb` | Feature extraction → Logistic Regression → Random Forest (GridSearchCV) → 1D ResNet, CNN1D, Recurrent-based model training |
| `04_evaluation.ipynb` | Test-set metrics, threshold optimisation, ROC/PR curves, confusion matrices, error analysis |
| `05_interpretability.ipynb` | SHAP (Random Forest), GradCAM (ResNet1D), ECG saliency overlays, lead importance |

---

## Methodology

### Preprocessing
- **Bandpass filter**: 0.5–40 Hz, Butterworth order 4, zero-phase
- **Notch filter**: 50 Hz (EU powerline), Q = 30
- **Normalisation**: Z-score per lead per patient
- **Splits**: 70 / 15 / 15 % stratified by label, `random_seed=42`

### Features (Classical ML, ~200 per patient)
- Time-domain per lead: mean, std, min, max, range, RMS, skewness, kurtosis
- Frequency-domain per lead: power + relative power in 3 bands (0.5–5 / 5–15 / 15–40 Hz)
- Brugada morphology (V1/V2/V3): J-point amplitude, ST elevation, ST slope, coved ratio
- HRV from lead II: HR mean/std, SDNN, RMSSD, pNN50

### Models
| Model | Type | Imbalance handling |
|---|---|---|
| Logistic Regression | Classical ML | Class weights |
| Random Forest | Classical ML | `class_weight='balanced'` + GridSearchCV |
| **1D ResNet** (primary) | Deep Learning | `BCEWithLogitsLoss` with `pos_weight` + augmentation |
| **CNN1D** | Deep Learning | `BCEWithLogitsLoss` with `pos_weight` + augmentation |
| **LSTM1D** | Deep Learning | `BCEWithLogitsLoss` with `pos_weight` + augmentation |
| **BiLSTM1D** | Deep Learning | `BCEWithLogitsLoss` with `pos_weight` + augmentation |
| **RNN1D** | Deep Learning | `BCEWithLogitsLoss` with `pos_weight` + augmentation |

**1D ResNet architecture**: Stem (Conv1d k=15, s=2 → MaxPool) → 4 residual stages (64→128→256→512 channels) → GlobalAvgPool → FC(1). ~1.4 M parameters.

**CNN1D architecture**: Stem (Conv1d k=15, s=2 → MaxPool) → 3 convolutional blocks (k=7) with BatchNorm, ReLU, and Dropout, with channel progression 64→128→256 and downsampling via stride-2 convolutions → GlobalAvgPool → FC(1). ~328.2 K parameters.

**LSTM1D architecture**: Input sequence → 2-layer LSTM (hidden size 64) → temporal GlobalAvgPool → Fully connected head (64→32→1) with ReLU and Dropout. ~55.3 K parameters.

**BiLSTM1D architecture**: Input sequence → 2-layer bidirectional LSTM (hidden size 64 per direction) → temporal GlobalAvgPool → Fully connected head (128→32→1) with ReLU and Dropout. ~143.4 K parameters.

**RNN1D architecture**: Input sequence → 2-layer RNN (hidden size 64) → temporal GlobalAvgPool → Fully connected head (64→32→1) with ReLU and Dropout. ~15.4 K parameters.


**Training**: AdamW (lr=1e-3, wd=1e-4), CosineAnnealingLR, early stopping (patience=10), gradient clipping (max_norm=1.0). Augmentation: Gaussian noise + time shift + amplitude scaling.

### Interpretability
- **SHAP** (TreeExplainer): global feature importance, beeswarm, per-patient waterfall
- **GradCAM** (1D): time-point saliency overlaid on ECG trace, per-lead importance aggregation

---

## Key Metrics (Test Set)

Results populated after running notebooks 03 → 04.

| Model | AUROC | AUPRC | Sensitivity | Specificity | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7345 | 0.5951 | 0.8333 | 0.3488 | 0.4000 |
| Random Forest | 0.9341 | 0.7848 | **1.0000** | 0.7442 | 0.6857 |
| **1D ResNet** | 0.9574 | 0.9351 | 0.9167 | **0.9302** | **0.8462** |
| CNN1D | **0.9767** | **0.9493** | 0.9167 | 0.9070 | 0.8148 |
| RNN1D | 0.5465 | 0.3131 | 0.5000 | 0.4884 | 0.3000 |
| BiLSTM1D | 0.8178 | 0.5465 | 0.8333 | 0.6047 | 0.5128 |
| LSTM1D | 0.8934 | 0.7810 | **1.0000** | 0.3721 | 0.4706 |


---

## Project Structure

```
IDSC/
├── data/
│   ├── raw/              ← PhysioNet download (wget or wfdb)
│   └── processed/        ← .npz splits + .parquet feature caches
├── models/               ← best_resnet1d.pt, best_rnn1d.pt, best_lstm1d.pt, best_cnn1d, best_bilstm1d  rf_best.pkl, lr_best.pkl, scaler.pkl, etc.
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_interpretability.ipynb
├── reports/              ← all .png figures (auto-generated)
├── src/
│   ├── config.py         ← all hyperparameters and paths
│   ├── download.py       ← dataset download script
│   ├── preprocessing.py  ← filtering, normalisation, splits
│   ├── features.py       ← handcrafted ECG features
│   ├── dataset.py        ← PyTorch Dataset + DataLoader factory
│   ├── models/
│   │   ├── __init__.py
│   │   ├── bilstm1d.py ← 1D BiLSTM architecture
│   │   ├── cnn1d.py ← 1D CNN architecture
│   │   ├── lstm1d.py ← 1D LSTM architecture
│   │   ├── resnet1d.py ← 1D ResNet architecture
│   │   └── rnn1d.py ← 1D RNN architecture  
│   ├── train.py          ← training loop, early stopping, checkpointing
│   └── evaluate.py       ← metrics, threshold search, plotting
└── requirements.txt
```

---

## Citation

```
Costa Cortez, N., & Garcia Iglesias, D. (2026).
Brugada-HUCA: 12-Lead ECG Recordings for the Study of Brugada Syndrome (version 1.0.0).
PhysioNet. https://doi.org/10.13026/0m2w-dy83

Goldberger et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
Circulation 101(23): e215–e220.
```
