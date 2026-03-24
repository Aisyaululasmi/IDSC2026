"""
Evaluation utilities: metrics, threshold selection, and plotting.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    accuracy_score,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compute all classification metrics at a fixed threshold.

    Returns
    -------
    dict with keys: auroc, auprc, accuracy, sensitivity, specificity,
                    precision, f1, threshold, tp, tn, fp, fn
    """
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    try:
        auroc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auroc = float('nan')

    try:
        auprc = float(average_precision_score(y_true, y_prob))
    except Exception:
        auprc = float('nan')

    return {
        'auroc':       round(auroc, 4),
        'auprc':       round(auprc, 4),
        'accuracy':    round(float(accuracy_score(y_true, y_pred)), 4),
        'sensitivity': round(tp / (tp + fn + 1e-8), 4),
        'specificity': round(tn / (tn + fp + 1e-8), 4),
        'precision':   round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        'f1':          round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        'threshold':   threshold,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    min_sensitivity: float = 0.85,
) -> float:
    """
    Sweep thresholds and return the one maximising *metric*
    subject to sensitivity >= min_sensitivity.

    metric : 'f1' | 'youden'
    """
    best_thresh, best_score = 0.5, -1.0

    for t in np.linspace(0.01, 0.99, 199):
        y_pred = (y_prob >= t).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn + 1e-8)
        spec = tn / (tn + fp + 1e-8)

        if sens < min_sensitivity:
            continue

        score = (f1_score(y_true, y_pred, zero_division=0) if metric == 'f1'
                 else sens + spec - 1.0)

        if score > best_score:
            best_score, best_thresh = score, float(t)

    return best_thresh


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_dl(model, loader, device='cpu') -> tuple[np.ndarray, np.ndarray]:
    """
    Run DL model inference over a DataLoader.

    Returns
    -------
    y_true : (N,)
    y_prob : (N,)  sigmoid probabilities
    """
    model.eval()
    all_probs, all_labels = [], []
    for X_b, y_b in loader:
        X_b = X_b.to(device)
        prob = torch.sigmoid(model(X_b).squeeze(1)).cpu().numpy()
        all_probs.append(prob)
        all_labels.append(y_b.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ax=None,
    title: str = 'Confusion Matrix',
    save_path: Path = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    labels = ['Normal', 'Brugada']

    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(5, 4))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=labels, yticklabels=labels,
        linewidths=0.5, ax=ax,
    )
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)

    if created:
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150)
        plt.show()


def plot_roc_pr_curves(
    results: dict,
    save_path: Path = None,
) -> None:
    """
    Overlay ROC and PR curves for multiple models.

    results : { model_name: {'y_true': ..., 'y_prob': ...} }
    """
    palette = ['#C62828', '#1565C0', '#2E7D32', '#F57F17', '#6A1B9A']
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for (name, data), color in zip(results.items(), palette):
        yt, yp = data['y_true'], data['y_prob']

        fpr, tpr, _ = roc_curve(yt, yp)
        axes[0].plot(fpr, tpr, color=color, lw=2,
                     label=f'{name} (AUROC={roc_auc_score(yt, yp):.3f})')

        prec, rec, _ = precision_recall_curve(yt, yp)
        axes[1].plot(rec, prec, color=color, lw=2,
                     label=f'{name} (AUPRC={average_precision_score(yt, yp):.3f})')

    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].set(xlabel='FPR', ylabel='TPR', title='ROC Curves',
                xlim=(0, 1), ylim=(0, 1.02))
    axes[0].legend(loc='lower right', fontsize=9)

    prev = list(results.values())[0]['y_true'].mean()
    axes[1].axhline(prev, color='k', linestyle='--', lw=1, alpha=0.5,
                    label=f'Prevalence ({prev:.2f})')
    axes[1].set(xlabel='Recall', ylabel='Precision', title='PR Curves',
                xlim=(0, 1), ylim=(0, 1.02))
    axes[1].legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_training_curves(history: dict, save_path: Path = None) -> None:
    """Plot loss, accuracy, and AUROC curves from a training history dict."""
    epochs = range(1, len(history['train_loss']) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, tr_key, vl_key, ylabel in [
        (axes[0], 'train_loss',  'val_loss',  'BCE Loss'),
        (axes[1], 'train_acc',   'val_acc',   'Accuracy'),
        (axes[2], 'train_auroc', 'val_auroc', 'AUROC'),
    ]:
        ax.plot(epochs, history[tr_key], 'b-o', ms=3, label='Train')
        ax.plot(epochs, history[vl_key], 'r-o', ms=3, label='Val')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()

    best_ep = int(np.argmax(history['val_auroc'])) + 1
    axes[2].axvline(best_ep, color='green', linestyle='--', lw=1,
                    label=f'Best ep={best_ep}')
    axes[2].legend()

    plt.suptitle('Training Curves — ResNet1D', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def compare_models(metrics_dict: dict, save_path: Path = None) -> pd.DataFrame:
    """
    Build a comparison DataFrame from {model_name: metrics_dict}.
    Optionally renders as a figure for the tech report.
    """
    cols = ['AUROC', 'AUPRC', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'Accuracy']
    rows = []
    for name, m in metrics_dict.items():
        rows.append({
            'Model':       name,
            'AUROC':       m['auroc'],
            'AUPRC':       m['auprc'],
            'Sensitivity': m['sensitivity'],
            'Specificity': m['specificity'],
            'Precision':   m['precision'],
            'F1':          m['f1'],
            'Accuracy':    m['accuracy'],
        })
    df = pd.DataFrame(rows).set_index('Model')

    if save_path:
        fig, ax = plt.subplots(figsize=(14, max(3, len(df) + 1)))
        ax.axis('off')
        tbl = ax.table(
            cellText=df.values,
            rowLabels=df.index,
            colLabels=df.columns,
            cellLoc='center', loc='center',
        )
        tbl.auto_set_font_size(True)
        tbl.scale(1.2, 1.8)
        ax.set_title('Model Comparison — Test Set', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()

    return df
