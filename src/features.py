"""
Handcrafted ECG feature extraction for the Brugada-HUCA dataset.

Features per patient (~200 total):
  - Time-domain per lead  : mean, std, min, max, range, skewness, kurtosis, rms
  - Frequency-domain per lead : power and relative power in 3 bands
  - Brugada-specific (V1/V2/V3): J-point, ST elevation, ST slope, coved ratio
  - HRV from lead II      : HR mean/std, SDNN, RMSSD, pNN50
"""

import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from scipy.signal import welch
from pathlib import Path
from tqdm import tqdm

from config import FS, LEAD_NAMES, BRUGADA_LEADS, N_SAMPLES


# ---------------------------------------------------------------------------
# Time-domain features (per lead)
# ---------------------------------------------------------------------------

def _time_features(lead: np.ndarray) -> dict:
    return {
        'mean':     float(lead.mean()),
        'std':      float(lead.std()),
        'min':      float(lead.min()),
        'max':      float(lead.max()),
        'range':    float(lead.max() - lead.min()),
        'abs_mean': float(np.abs(lead).mean()),
        'rms':      float(np.sqrt(np.mean(lead ** 2))),
        'skew':     float(skew(lead)),
        'kurtosis': float(kurtosis(lead)),
    }


# ---------------------------------------------------------------------------
# Frequency-domain features (per lead)
# ---------------------------------------------------------------------------

FREQ_BANDS = {
    'low':  (0.5,  5.0),
    'mid':  (5.0,  15.0),
    'high': (15.0, 40.0),
}


def _freq_features(lead: np.ndarray, fs: float = FS) -> dict:
    freqs, psd = welch(lead, fs=fs, nperseg=min(256, len(lead)))
    _trapz = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    total = _trapz(psd, freqs) + 1e-12
    out = {}
    for name, (flo, fhi) in FREQ_BANDS.items():
        mask = (freqs >= flo) & (freqs <= fhi)
        power = _trapz(psd[mask], freqs[mask])
        out[f'{name}_power'] = float(power)
        out[f'{name}_ratio'] = float(power / total)
    return out


# ---------------------------------------------------------------------------
# Brugada-specific morphology features (V1, V2, V3)
# ---------------------------------------------------------------------------

def _brugada_features(signal: np.ndarray) -> dict:
    """
    Extract Brugada morphology proxies from V1/V2/V3.

    Uses the first beat (≈ first 600 ms) of each lead.
    Regions:
      QRS region : 50–200 ms  (J-point proxy)
      ST  region : 200–400 ms (ST elevation / slope)
    """
    qrs_s = int(0.05 * FS)
    qrs_e = int(0.20 * FS)
    st_s  = int(0.20 * FS)
    st_e  = int(0.40 * FS)
    beat_end = int(0.60 * FS)

    out = {}
    for lead_name in BRUGADA_LEADS:
        li   = LEAD_NAMES.index(lead_name)
        beat = signal[li, :beat_end]
        p    = lead_name.lower()

        # J-point amplitude proxy
        out[f'{p}_j_point']    = float(beat[qrs_s:qrs_e].max())

        # ST elevation: mean amplitude in ST region
        st_seg = beat[st_s:st_e]
        out[f'{p}_st_elev']    = float(st_seg.mean())

        # ST slope via linear regression
        x = np.arange(len(st_seg), dtype=float)
        out[f'{p}_st_slope']   = float(np.polyfit(x, st_seg, 1)[0]) if len(x) > 1 else 0.0

        # Coved ratio: peak / mean in QRS
        qrs_region = beat[qrs_s:qrs_e]
        denom = np.abs(qrs_region).mean() + 1e-8
        out[f'{p}_coved_ratio'] = float(qrs_region.max() / denom)

    return out


# ---------------------------------------------------------------------------
# HRV features (lead II)
# ---------------------------------------------------------------------------

_HRV_DEFAULTS = dict(
    hr_mean=0.0, hr_std=0.0,
    rr_mean=0.0, rr_std=0.0,
    sdnn=0.0, rmssd=0.0, pnn50=0.0,
    n_rpeaks=0,
)


def _hrv_features(signal: np.ndarray, fs: float = FS) -> dict:
    """Compute HRV from lead II R-peaks. Falls back to zeros on failure."""
    lead_ii = signal[LEAD_NAMES.index('II')]
    try:
        import neurokit2 as nk
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            _, info = nk.ecg_peaks(lead_ii, sampling_rate=int(fs), method='neurokit')
        rpeaks = info['ECG_R_Peaks']
        if len(rpeaks) < 2:
            return _HRV_DEFAULTS.copy()
        rr   = np.diff(rpeaks) / fs      # seconds
        hr   = 60.0 / rr                 # bpm
        diff = np.diff(rr)
        return dict(
            hr_mean  = float(hr.mean()),
            hr_std   = float(hr.std()),
            rr_mean  = float(rr.mean()),
            rr_std   = float(rr.std()),
            sdnn     = float(rr.std()),
            rmssd    = float(np.sqrt(np.mean(diff ** 2))) if len(diff) else 0.0,
            pnn50    = float(np.mean(np.abs(diff) > 0.05)) if len(diff) else 0.0,
            n_rpeaks = int(len(rpeaks)),
        )
    except Exception:
        return _HRV_DEFAULTS.copy()


# ---------------------------------------------------------------------------
# Full feature vector
# ---------------------------------------------------------------------------

def extract_single(signal: np.ndarray) -> dict:
    """
    Extract the full flat feature dict from one (N_LEADS, N_SAMPLES) signal.
    """
    out = {}

    for li, lead in enumerate(LEAD_NAMES):
        p = lead.lower()
        for k, v in _time_features(signal[li]).items():
            out[f'{p}_{k}'] = v
        for k, v in _freq_features(signal[li]).items():
            out[f'{p}_{k}'] = v

    out.update(_brugada_features(signal))
    out.update(_hrv_features(signal))

    return out


def extract_batch(
    X: np.ndarray,
    ids: list | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Extract features for all N signals.

    Parameters
    ----------
    X : (N, 12, 1200)
    ids : optional patient IDs for the DataFrame index

    Returns
    -------
    df : (N, n_features) DataFrame
    """
    rows = []
    iterator = tqdm(range(len(X)), desc='Feature extraction') if verbose else range(len(X))

    for i in iterator:
        try:
            rows.append(extract_single(X[i]))
        except Exception as exc:
            if verbose:
                print(f'  [WARN] index {i}: {exc}')
            rows.append({})

    df = pd.DataFrame(rows)
    if ids is not None:
        df.index = ids

    # Replace any NaN (failed extractions) with column median
    df = df.fillna(df.median(numeric_only=True))
    return df
