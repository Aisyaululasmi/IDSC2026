"""
PyTorch Dataset and DataLoader factory for the Brugada-HUCA ECG dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import BATCH_SIZE, RANDOM_SEED
from preprocessing import augment_signal


class ECGDataset(Dataset):
    """
    12-lead ECG dataset.

    Parameters
    ----------
    X       : (N, 12, 1200) float32 preprocessed signals
    y       : (N,) binary labels — 1=Brugada, 0=Normal
    augment : apply random augmentation (training only)
    seed    : RNG seed for reproducible augmentation
    """

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        augment: bool = False,
        seed: int = RANDOM_SEED,
    ):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
        self.augment = augment
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.augment:
            x = augment_signal(x, rng=self._rng)
        return (
            torch.from_numpy(x),
            torch.tensor(self.y[idx], dtype=torch.float32),
        )

    @property
    def pos_weight(self) -> torch.Tensor:
        """Return pos_weight scalar tensor for BCEWithLogitsLoss."""
        n_pos = (self.y == 1).sum()
        n_neg = (self.y == 0).sum()
        return torch.tensor(n_neg / (n_pos + 1e-8), dtype=torch.float32)


def make_loaders(
    splits: dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> dict[str, DataLoader]:
    """
    Build train / val / test DataLoaders from a splits dict
    (as returned by preprocessing.make_splits).

    Returns
    -------
    {'train': DataLoader, 'val': DataLoader, 'test': DataLoader}
    """
    train_ds = ECGDataset(splits['X_train'], splits['y_train'], augment=True)
    val_ds   = ECGDataset(splits['X_val'],   splits['y_val'],   augment=False)
    test_ds  = ECGDataset(splits['X_test'],  splits['y_test'],  augment=False)

    return {
        'train': DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        'val': DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
        'test': DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory,
        ),
    }
