"""
Preprocessing pipeline for the Brugada-HUCA ECG dataset.

Provides functions for:
  - Loading WFDB records
  - Bandpass + notch filtering
  - Per-lead Z-score normalization
  - Stratified train / val / test splitting
  - SMOTE oversampling on the training set
  - Saving / loading processed arrays
"""

import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, iirnotch, sosfilt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import (
    DATA_RAW, DATA_PROCESSED, METADATA_FILE,
    LABEL_COL, PATIENT_ID_COL, POSITIVE_CLASS,
    FS, N_SAMPLES, N_LEADS,
    LOWCUT, HIGHCUT, NOTCH_FREQ, FILTER_ORDER,
    TEST_SIZE, VAL_SIZE, RANDOM_SEED,
)


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------

def list_records(data_dir: Path = DATA_RAW) -> list[str]:
    """Return record names by scanning the 'files' subdirectory."""
    files_dir = data_dir / "files"
    if not files_dir.exists():
        raise FileNotFoundError(f"'files' directory not found at {files_dir}")
    return sorted(p.name for p in files_dir.iterdir() if p.is_dir())


def load_record(record_name: str, data_dir: Path = DATA_RAW):
    """
    Load a single WFDB record.

    Returns
    -------
    signal : np.ndarray, shape (N_LEADS, N_SAMPLES)  — float32, in mV
    fields : dict  — header metadata
    """
    # Actual structure: files/{patient_id}/{patient_id}.dat
    record_path = str(data_dir / "files" / record_name / record_name)
    record = wfdb.rdrecord(record_path)
    signal = record.p_signal.T.astype(np.float32)   # (leads, samples)

    # Pad or trim to exactly N_SAMPLES
    if signal.shape[1] < N_SAMPLES:
        pad = N_SAMPLES - signal.shape[1]
        signal = np.pad(signal, ((0, 0), (0, pad)), mode="edge")
    elif signal.shape[1] > N_SAMPLES:
        signal = signal[:, :N_SAMPLES]

    return signal, record.__dict__


def load_all_records(
    metadata: pd.DataFrame,
    data_dir: Path = DATA_RAW,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load all records listed in *metadata*.

    Returns
    -------
    X : np.ndarray, shape (N, N_LEADS, N_SAMPLES)
    y : np.ndarray, shape (N,)  — 1 = Brugada, 0 = Normal
    ids : list[str]  — patient IDs in the same order
    """
    signals, labels, ids = [], [], []

    for _, row in metadata.iterrows():
        pid = str(row[PATIENT_ID_COL])
        try:
            sig, _ = load_record(pid, data_dir)
            signals.append(sig)
            labels.append(1 if int(row[LABEL_COL]) >= 1 else 0)  # 0=Normal, 1/2=Brugada
            ids.append(pid)
        except Exception as exc:
            if verbose:
                print(f"  [WARN] Skipping {pid}: {exc}")

    X = np.stack(signals, axis=0)   # (N, 12, 1200)
    y = np.array(labels, dtype=np.int64)
    return X, y, ids


# ---------------------------------------------------------------------------
# Signal filtering
# ---------------------------------------------------------------------------

def _bandpass_sos(fs: float, lowcut: float, highcut: float, order: int):
    nyq = 0.5 * fs
    return butter(order, [lowcut / nyq, highcut / nyq], btype="band", output="sos")


def _notch_sos(fs: float, freq: float, quality: float = 30.0):
    b, a = iirnotch(freq / (0.5 * fs), quality)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    return tf2sos(b, a)


def bandpass_filter(
    signal: np.ndarray,
    fs: float = FS,
    lowcut: float = LOWCUT,
    highcut: float = HIGHCUT,
    order: int = FILTER_ORDER,
) -> np.ndarray:
    """
    Apply zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    signal : (leads, samples) or (samples,)
    Returns same shape.
    """
    sos = _bandpass_sos(fs, lowcut, highcut, order)
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def notch_filter(
    signal: np.ndarray,
    fs: float = FS,
    freq: float = NOTCH_FREQ,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase notch filter to remove powerline interference."""
    sos = _notch_sos(fs, freq, quality)
    return sosfiltfilt(sos, signal, axis=-1).astype(np.float32)


def filter_signal(signal: np.ndarray) -> np.ndarray:
    """Apply the full filter chain: bandpass → notch."""
    signal = bandpass_filter(signal)
    signal = notch_filter(signal)
    return signal


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_signal(signal: np.ndarray) -> np.ndarray:
    """
    Z-score normalize each lead independently.

    Parameters
    ----------
    signal : (..., leads, samples)
    """
    mean = signal.mean(axis=-1, keepdims=True)
    std  = signal.std(axis=-1, keepdims=True)
    std  = np.where(std == 0, 1.0, std)          # avoid division by zero
    return ((signal - mean) / std).astype(np.float32)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_batch(X: np.ndarray) -> np.ndarray:
    """
    Apply filter + normalize to a batch of signals.

    Parameters
    ----------
    X : (N, leads, samples)

    Returns
    -------
    X_clean : (N, leads, samples)  — float32
    """
    out = np.empty_like(X, dtype=np.float32)
    for i in range(len(X)):
        out[i] = normalize_signal(filter_signal(X[i]))
    return out


# ---------------------------------------------------------------------------
# Data augmentation (training only)
# ---------------------------------------------------------------------------

def augment_signal(
    signal: np.ndarray,
    noise_std: float = 0.01,
    shift_max: int = 10,
    scale_range: tuple = (0.9, 1.1),
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply random augmentation to a single signal (leads, samples).

    Augmentations applied independently and randomly:
      1. Additive Gaussian noise
      2. Random time-shift (circular)
      3. Amplitude scaling
    """
    if rng is None:
        rng = np.random.default_rng()

    sig = signal.copy()

    # 1. Gaussian noise
    sig += rng.normal(0, noise_std, size=sig.shape).astype(np.float32)

    # 2. Time shift
    shift = int(rng.integers(-shift_max, shift_max + 1))
    sig = np.roll(sig, shift, axis=-1)

    # 3. Amplitude scale
    scale = rng.uniform(*scale_range)
    sig *= scale

    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# Train / Val / Test splitting
# ---------------------------------------------------------------------------

def make_splits(
    X: np.ndarray,
    y: np.ndarray,
    ids: list[str],
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
) -> dict:
    """
    Stratified patient-level split into train / val / test.

    Returns
    -------
    splits : dict with keys
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        ids_train, ids_val, ids_test
    """
    idx = np.arange(len(X))

    # First split off test set
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_size, stratify=y, random_state=random_seed
    )

    # Then split val from remaining
    y_trainval = y[idx_trainval]
    idx_train, idx_val = train_test_split(
        idx_trainval,
        test_size=val_size / (1.0 - test_size),
        stratify=y_trainval,
        random_state=random_seed,
    )

    ids_arr = np.array(ids)
    return {
        "X_train": X[idx_train], "y_train": y[idx_train], "ids_train": ids_arr[idx_train].tolist(),
        "X_val":   X[idx_val],   "y_val":   y[idx_val],   "ids_val":   ids_arr[idx_val].tolist(),
        "X_test":  X[idx_test],  "y_test":  y[idx_test],  "ids_test":  ids_arr[idx_test].tolist(),
    }


# ---------------------------------------------------------------------------
# Class-imbalance handling
# ---------------------------------------------------------------------------

def compute_pos_weight(y_train: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.

    pos_weight = n_negative / n_positive
    """
    n_pos = (y_train == 1).sum()
    n_neg = (y_train == 0).sum()
    return float(n_neg) / float(n_pos)


def smote_oversample(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_seed: int = RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to the flattened training set, then reshape back.

    Requires imbalanced-learn (pip install imbalanced-learn).

    Returns
    -------
    X_res : (N_resampled, leads, samples)
    y_res : (N_resampled,)
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError(
            "imbalanced-learn is required for SMOTE: pip install imbalanced-learn"
        )

    n, leads, samples = X_train.shape
    X_flat = X_train.reshape(n, leads * samples)

    sm = SMOTE(random_state=random_seed)
    X_res_flat, y_res = sm.fit_resample(X_flat, y_train)

    X_res = X_res_flat.reshape(-1, leads, samples).astype(np.float32)
    return X_res, y_res


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_processed(splits: dict, output_dir: Path = DATA_PROCESSED) -> None:
    """Save all split arrays to *output_dir* as compressed .npz."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "val", "test"):
        X = splits[f"X_{split_name}"]
        y = splits[f"y_{split_name}"]
        ids = splits[f"ids_{split_name}"]
        out_path = output_dir / f"{split_name}.npz"
        np.savez_compressed(out_path, X=X, y=y, ids=np.array(ids))
        print(f"  Saved {split_name}: {X.shape}  →  {out_path}")


def load_processed(
    split: str,
    data_dir: Path = DATA_PROCESSED,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load a saved split.

    Parameters
    ----------
    split : "train" | "val" | "test"

    Returns
    -------
    X, y, ids
    """
    path = data_dir / f"{split}.npz"
    data = np.load(path, allow_pickle=True)
    return data["X"], data["y"], data["ids"].tolist()
