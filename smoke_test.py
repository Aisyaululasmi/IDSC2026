"""Smoke test: load metadata → load all records → preprocess one batch."""
import sys
sys.path.insert(0, r'C:\Users\T480s\IDSC\src')

import numpy as np
import pandas as pd
import config as cfg
from preprocessing import (
    list_records, load_record, load_all_records,
    bandpass_filter, notch_filter, normalize_signal,
    preprocess_batch, make_splits, compute_pos_weight,
)

# 1. Metadata
meta = pd.read_csv(cfg.METADATA_FILE)
print(f"[1] Metadata: {meta.shape}  Brugada={( meta[cfg.LABEL_COL]==1).sum()}  Normal={(meta[cfg.LABEL_COL]==0).sum()}")

# 2. Load first 5 records only (quick check)
print("[2] Loading first 5 records...")
subset = meta.head(5)
X, y, ids = load_all_records(subset, cfg.DATA_RAW, verbose=True)
print(f"    X={X.shape}  y={y}  ids={ids}")

# 3. Filter pipeline on first record
sig = X[0]
bp  = bandpass_filter(sig)
ntch = notch_filter(bp)
norm = normalize_signal(ntch)
print(f"[3] Filter pipeline: raw.mean={sig.mean():.4f}  norm.mean={norm.mean():.4f}  norm.std={norm.std():.4f}")

# 4. Preprocess batch
X_clean = preprocess_batch(X)
print(f"[4] preprocess_batch: {X_clean.shape}  mean={X_clean.mean():.4f}  std={X_clean.std():.4f}")

print("\nAll checks passed!")
