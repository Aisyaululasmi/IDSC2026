"""
Training utilities for the Brugada ResNet1D.

Public API
----------
  run_training(model, train_loader, val_loader, ...) -> history dict
  save_checkpoint(state, path)
  load_checkpoint(path, model, optimizer=None) -> epoch
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from config import MODELS_DIR, NUM_EPOCHS, PATIENCE


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = PATIENCE, min_delta: float = 1e-4, mode: str = 'min'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.best      = None
        self.triggered = False

    def step(self, value: float) -> bool:
        """Return True if training should stop."""
        if self.best is None:
            self.best = value
            return False

        better = (value < self.best - self.min_delta) if self.mode == 'min' \
                 else (value > self.best + self.min_delta)

        if better:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False


# ---------------------------------------------------------------------------
# Single-epoch routines
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, criterion, device) -> dict:
    """One training epoch. Returns loss and accuracy."""
    model.train()
    total_loss = total_correct = total = 0
    all_probs, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)

        optimizer.zero_grad()
        logit = model(X_b).squeeze(1)
        loss  = criterion(logit, y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        prob  = torch.sigmoid(logit).detach().cpu().numpy()
        pred  = (prob > 0.5).astype(int)
        label = y_b.cpu().numpy().astype(int)

        total_loss    += loss.item() * len(y_b)
        total_correct += int((pred == label).sum())
        total         += len(y_b)
        all_probs.append(prob)
        all_labels.append(label)

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    try:
        auroc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        auroc = 0.5

    return {
        'loss':  total_loss / total,
        'acc':   total_correct / total,
        'auroc': auroc,
    }


@torch.no_grad()
def val_epoch(model, loader, criterion, device) -> dict:
    """One validation epoch. Returns loss, accuracy, and AUROC."""
    model.eval()
    total_loss = total_correct = total = 0
    all_probs, all_labels = [], []

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)

        logit = model(X_b).squeeze(1)
        loss  = criterion(logit, y_b)

        prob  = torch.sigmoid(logit).cpu().numpy()
        pred  = (prob > 0.5).astype(int)
        label = y_b.cpu().numpy().astype(int)

        total_loss    += loss.item() * len(y_b)
        total_correct += int((pred == label).sum())
        total         += len(y_b)
        all_probs.append(prob)
        all_labels.append(label)

    all_probs  = np.concatenate(all_probs)
    all_labels = np.concatenate(all_labels)

    try:
        auroc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        auroc = 0.5

    return {
        'loss':  total_loss / total,
        'acc':   total_correct / total,
        'auroc': auroc,
    }


# ---------------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------------

def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    scheduler=None,
    device: str = 'cpu',
    num_epochs: int = NUM_EPOCHS,
    patience: int = PATIENCE,
    checkpoint_path: Path = None,
    verbose: bool = True,
) -> dict:
    """
    Train with early stopping, save the best checkpoint.

    Returns
    -------
    history : dict with keys
        train_loss, train_acc, train_auroc,
        val_loss,   val_acc,   val_auroc
    """
    if checkpoint_path is None:
        checkpoint_path = MODELS_DIR / 'best_resnet1d.pt'
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    early_stop    = EarlyStopping(patience=patience, mode='min')
    best_val_loss = float('inf')
    history = {k: [] for k in [
        'train_loss', 'train_acc', 'train_auroc',
        'val_loss',   'val_acc',   'val_auroc',
    ]}

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        tr = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = val_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step()

        for key in ('loss', 'acc', 'auroc'):
            history[f'train_{key}'].append(tr[key])
            history[f'val_{key}'].append(vl[key])

        # Save best model (by val loss)
        if vl['loss'] < best_val_loss:
            best_val_loss = vl['loss']
            save_checkpoint(
                {'epoch': epoch, 'model_state': model.state_dict(),
                 'optimizer_state': optimizer.state_dict(), 'val_loss': best_val_loss},
                checkpoint_path,
            )

        if verbose:
            print(
                f'Epoch {epoch:3d}/{num_epochs} | '
                f'Train loss={tr["loss"]:.4f} acc={tr["acc"]:.3f} AUROC={tr["auroc"]:.3f} | '
                f'Val   loss={vl["loss"]:.4f} acc={vl["acc"]:.3f} AUROC={vl["auroc"]:.3f} | '
                f'{time.time()-t0:.1f}s'
            )

        if early_stop.step(vl['loss']):
            if verbose:
                print(f'  Early stopping at epoch {epoch} (patience={patience}).')
            break

    return history


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: Path) -> None:
    torch.save(state, path)


def load_checkpoint(path: Path, model, optimizer=None) -> int:
    """Load a checkpoint. Returns the saved epoch number."""
    ckpt = torch.load(path, map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    if optimizer is not None and 'optimizer_state' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state'])
    return ckpt.get('epoch', 0)
