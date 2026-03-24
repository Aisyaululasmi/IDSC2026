"""
Download the Brugada-HUCA dataset from PhysioNet.

Usage:
    python src/download.py

The dataset will be saved to data/raw/.
Requires wget to be installed, or uses Python fallback via wfdb.dl_database().
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT_DIR / "data" / "raw"


def download_with_wfdb():
    """Download via the wfdb Python library (no wget needed)."""
    import wfdb
    dest = DATA_RAW / "physionet.org" / "files" / "brugada-huca" / "1.0.0"
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading Brugada-HUCA dataset to {dest} ...")
    wfdb.dl_database("brugada-huca", str(dest))
    print("Download complete.")


def download_with_wget():
    """Download via wget (mirrors the competition command)."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    url = "https://physionet.org/files/brugada-huca/1.0.0/"
    cmd = ["wget", "-r", "-N", "-c", "-np", url]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=str(DATA_RAW))
    if result.returncode != 0:
        print("wget failed. Falling back to wfdb download...")
        download_with_wfdb()


if __name__ == "__main__":
    try:
        download_with_wget()
    except FileNotFoundError:
        print("wget not found. Using wfdb fallback...")
        download_with_wfdb()
