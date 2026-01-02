#!/usr/bin/env python3


"""
Model evaluation utilities.

WHY this exists:
- Accuracy alone can be misleading, especially with many classes.
- Macro-F1 treats each disease equally, so rare diseases matter.
- A confusion matrix helps diagnose which diseases the model confuses.
"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score


def load_itos(vocab_path: str) -> List[str]:
    """
    Loads vocab.itos saved as JSON.

    WHY:
    - The model expects token IDs derived from the same vocabulary used in training.
    """
    with open(vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj["itos"]


@torch.no_grad()
def predict_bilstm(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """
    Runs inference for BiLSTM.
    """
    model.eval()
    all_preds = []
    all_labels = []

    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)

        logits = model(input_ids, lengths)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes core metrics.

    WHY:
    - macro_f1 is crucial for multi-class healthcare tasks where each class matters.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro")),
    }


def pretty_report(y_true: np.ndarray, y_pred: np.ndarray, id_to_label: Dict[int, str]) -> str:
    """
    Produces a readable per-class report showing per-class precision/recall/F1.
    """
    labels_sorted = sorted(id_to_label.keys())
    target_names = [id_to_label[i] for i in labels_sorted]
    return classification_report(
        y_true,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, id_to_label: Dict[int, str], top_n: int = 10):
    """
    Returns the most common misclassification pairs.

    WHY:
    - Helps you explain model failure modes critical for healthcare-adjacent models.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)

    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((cm[i, j], id_to_label[i], id_to_label[j]))

    pairs.sort(reverse=True, key=lambda x: x[0])
    return pairs[:top_n]
