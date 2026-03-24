"""Central configuration for the Brugada detection project."""

import os
from pathlib import Path

# --- Paths ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw" / "physionet.org" / "files" / "brugada-huca" / "1.0.0"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"

# --- Dataset ---
METADATA_FILE = DATA_RAW / "metadata.csv"
LABEL_COL = "brugada"          # "Brugada" or "Normal"
PATIENT_ID_COL = "patient_id"
POSITIVE_CLASS = 1                 # brugada column: >=1 = Brugada (types 1&2), 0 = Normal

# --- Signal ---
FS = 100                        # Hz
DURATION = 12                   # seconds
N_SAMPLES = FS * DURATION       # 1200 samples per lead
N_LEADS = 12
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF",
              "V1", "V2", "V3", "V4", "V5", "V6"]

# Key leads for Brugada (coved ST pattern appears here)
BRUGADA_LEADS = ["V1", "V2", "V3"]

# --- Preprocessing ---
LOWCUT = 0.5                    # Hz  — bandpass low
HIGHCUT = 40.0                  # Hz  — bandpass high
NOTCH_FREQ = 50.0               # Hz  — powerline (Europe)
FILTER_ORDER = 4

# --- Splits ---
RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15                 # fraction of remaining after test split

# --- Training ---
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 10                   # early stopping

# --- Augmentation ---
AUGMENT_NOISE_STD = 0.01        # Gaussian noise std (mV)
AUGMENT_SHIFT_MAX = 10          # max sample shift
AUGMENT_SCALE_RANGE = (0.9, 1.1)
