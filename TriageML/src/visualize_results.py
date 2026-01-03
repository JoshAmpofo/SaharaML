#!/usr/bin/env python3

"""
Visualization module for model training and evaluation results.

Generates both standard ML graphs and medical-relevant visualizations:
- Training curves (loss, accuracy)
- Confusion matrix
- Per-disease performance metrics
- Error analysis visualizations
"""

from __future__ import annotations

import json
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src.config import ARTIFACTS_DIR, IMAGES_DIR, BILSTM_CKPT_PATH, LABEL_ENCODER_PATH, TRANSFORMER_CKPT_PATH
from src.dataset_bilstm import SymptomDataset, collate_batch, Vocab
from src.dataset_transformer import TransformerSymptomDataset, collate_transformer_batch
from src.evaluate import compute_metrics
from src.preprocess import run_preprocessing
from src.split import make_group_splits
from models.model_bilstm import BiLSTMClassifier
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100


def load_vocab(vocab_path):
    """Load vocabulary from JSON file."""
    with open(vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    itos = obj["itos"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def load_tokenizer(model_name: str):
    """Load HuggingFace tokenizer."""
    return AutoTokenizer.from_pretrained(model_name)


def load_training_history(history_path):
    """Load training history from JSON file."""
    with open(history_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_training_curves(history: Dict, model_type: str = "BiLSTM", save_path: str = None):
    """
    Plot training and validation loss/accuracy curves.
    
    Standard ML visualization to track model learning progress.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-s", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].set_title(f"{model_type} Training and Validation Loss", fontsize=14, fontweight="bold")
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-s", label="Val Accuracy", linewidth=2)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy", fontsize=12)
    axes[1].set_title(f"{model_type} Training and Validation Accuracy", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 1.05])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved {model_type} training curves to {save_path}")
    plt.close()


@torch.no_grad()
def predict_bilstm(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference for BiLSTM model."""
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


@torch.no_grad()
def predict_transformer(model, loader, device) -> Tuple[np.ndarray, np.ndarray]:
    """Runs inference for Transformer model."""
    model.eval()
    all_preds = []
    all_labels = []

    for batch in loader:
        labels = batch["labels"].numpy()
        batch = {k: v.to(device) for k, v in batch.items()}

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        preds = outputs.logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    return np.concatenate(all_preds), np.concatenate(all_labels)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         id_to_label: Dict[int, str], model_type: str = "BiLSTM",
                         save_path: str = None):
    """
    Plot confusion matrix heatmap.
    
    Medical relevance: Shows which diseases are being confused,
    critical for understanding model failure modes in clinical context.
    """
    labels_sorted = sorted(id_to_label.keys())
    label_names = [id_to_label[i] for i in labels_sorted]
    
    # Limit to top diseases if too many classes
    if len(label_names) > 20:
        # Show only top 20 most common diseases
        unique, counts = np.unique(y_true, return_counts=True)
        top_indices = np.argsort(counts)[-20:]
        top_labels = unique[top_indices]
        
        # Filter predictions and true labels
        mask = np.isin(y_true, top_labels)
        y_true_filtered = y_true[mask]
        y_pred_filtered = y_pred[mask]
        
        label_names = [id_to_label[i] for i in top_labels]
        cm = confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_labels)
        title_suffix = " (Top 20 Diseases)"
    else:
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        title_suffix = ""
    
    # Normalize by row (true labels)
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_normalized = np.nan_to_num(cm_normalized)  # Handle division by zero
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(
        cm_normalized,
        annot=False,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=label_names,
        yticklabels=label_names,
        cbar_kws={"label": "Proportion"},
        ax=ax,
    )
    
    ax.set_xlabel("Predicted Disease", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Disease", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_type} Confusion Matrix{title_suffix} (Normalized by True Label)", 
                fontsize=14, fontweight="bold")
    
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved {model_type} confusion matrix to {save_path}")
    plt.close()


def plot_per_disease_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                             id_to_label: Dict[int, str], model_type: str = "BiLSTM",
                             save_path: str = None):
    """
    Plot per-disease precision, recall, and F1 scores.
    
    Medical relevance: Identifies which diseases the model diagnoses well
    and which need improvement. Critical for clinical deployment decisions.
    """
    labels_sorted = sorted(id_to_label.keys())
    target_names = [id_to_label[i] for i in labels_sorted]
    
    # Get classification report as dict
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_sorted,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    
    # Extract metrics for each disease
    diseases = []
    precisions = []
    recalls = []
    f1_scores = []
    supports = []
    
    for disease_name in target_names:
        if disease_name in report:
            diseases.append(disease_name)
            precisions.append(report[disease_name]["precision"])
            recalls.append(report[disease_name]["recall"])
            f1_scores.append(report[disease_name]["f1-score"])
            supports.append(report[disease_name]["support"])
    
    # Sort by F1 score for better visualization
    sorted_indices = np.argsort(f1_scores)
    diseases = [diseases[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]
    supports = [supports[i] for i in sorted_indices]
    
    # Limit to show all if reasonable, or top/bottom performers
    if len(diseases) > 25:
        # Show bottom 15 and top 10 performers
        indices = list(range(15)) + list(range(-10, 0))
        diseases = [diseases[i] for i in indices]
        precisions = [precisions[i] for i in indices]
        recalls = [recalls[i] for i in indices]
        f1_scores = [f1_scores[i] for i in indices]
        supports = [supports[i] for i in indices]
    
    # Create horizontal bar plot
    x = np.arange(len(diseases))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, max(8, len(diseases) * 0.3)))
    
    bars1 = ax.barh(x - width, precisions, width, label="Precision", color="#3498db")
    bars2 = ax.barh(x, recalls, width, label="Recall", color="#2ecc71")
    bars3 = ax.barh(x + width, f1_scores, width, label="F1-Score", color="#e74c3c")
    
    ax.set_xlabel("Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Disease", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_type} Per-Disease Performance Metrics", fontsize=14, fontweight="bold")
    ax.set_yticks(x)
    ax.set_yticklabels(diseases, fontsize=9)
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1.05])
    ax.grid(axis="x", alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved {model_type} per-disease metrics to {save_path}")
    plt.close()


def plot_top_confusions(y_true: np.ndarray, y_pred: np.ndarray,
                       id_to_label: Dict[int, str], model_type: str = "BiLSTM",
                       top_n: int = 15, save_path: str = None):
    """
    Plot the most common misclassification pairs.
    
    Medical relevance: Understanding which diseases are confused helps
    identify symptom overlaps and guide feature engineering or data collection.
    """
    cm = confusion_matrix(y_true, y_pred)
    np.fill_diagonal(cm, 0)  # Remove correct predictions
    
    # Find top confusions
    pairs = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] > 0:
                pairs.append((cm[i, j], id_to_label[i], id_to_label[j]))
    
    if not pairs:
        print("No confusions found - perfect predictions!")
        return
    
    # Sort by count and take top N
    pairs.sort(reverse=True)
    top_pairs = pairs[:top_n]
    
    if not top_pairs:
        return
    
    counts = [p[0] for p in top_pairs]
    labels = [f"{p[1]}\n→ {p[2]}" for p in top_pairs]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(top_pairs) * 0.4)))
    
    colors = plt.cm.Reds(np.linspace(0.4, 0.8, len(counts)))
    bars = ax.barh(range(len(counts)), counts, color=colors)
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Number of Misclassifications", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_type} Top {len(top_pairs)} Most Common Misclassifications", 
                fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f" {count}", va="center", fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved {model_type} top confusions to {save_path}")
    plt.close()


def plot_class_distribution(y_true: np.ndarray, id_to_label: Dict[int, str],
                           save_path: str = None):
    """
    Plot the distribution of diseases in the test set.
    
    Medical relevance: Shows class imbalance which affects model reliability
    for rare vs common diseases. Important for understanding deployment risks.
    """
    unique, counts = np.unique(y_true, return_counts=True)
    
    # Sort by count
    sorted_indices = np.argsort(counts)[::-1]
    unique = unique[sorted_indices]
    counts = counts[sorted_indices]
    
    disease_names = [id_to_label[i] for i in unique]
    
    # Limit display if too many
    if len(disease_names) > 25:
        disease_names = disease_names[:25]
        counts = counts[:25]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(disease_names) * 0.3)))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(counts)))
    bars = ax.barh(range(len(counts)), counts, color=colors)
    
    ax.set_yticks(range(len(disease_names)))
    ax.set_yticklabels(disease_names, fontsize=9)
    ax.set_xlabel("Number of Samples", fontsize=12, fontweight="bold")
    ax.set_title("Disease Distribution in Test Set", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    
    # Add value labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax.text(count, i, f" {count}", va="center", fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"✓ Saved class distribution to {save_path}")
    plt.close()


def generate_bilstm_visualizations():
    """
    Generate all visualizations for BiLSTM model.
    """
    print("="*60)
    print("Generating Visualizations for BiLSTM Model")
    print("="*60)
    
    # 1. Load and plot training history
    print("\n[1/6] Plotting training curves...")
    history = load_training_history(ARTIFACTS_DIR / "bilstm_history.json")
    plot_training_curves(history, model_type="BiLSTM", 
                        save_path=IMAGES_DIR / "bilstm_training_curves.png")
    
    # 2. Load test data and model for evaluation plots
    print("[2/6] Loading model and test data...")
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")
    
    # Load label encoder
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Load vocab and model
    vocab = load_vocab(ARTIFACTS_DIR / "vocab.json")
    
    test_ds = SymptomDataset(
        symptom_texts=splits.test["symptom_text"].tolist(),
        labels=splits.test["label_id"].tolist(),
        vocab=vocab,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(BILSTM_CKPT_PATH, map_location=device)
    
    model = BiLSTMClassifier(
        vocab_size=ckpt["vocab_size"],
        num_classes=ckpt["num_classes"],
        pad_id=ckpt["pad_id"],
        embedding_dim=ckpt["embedding_dim"],
        hidden_dim=ckpt["hidden_dim"],
        dropout=ckpt["dropout"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Get predictions
    print("[3/6] Generating predictions on test set...")
    y_pred, y_true = predict_bilstm(model, test_loader, device)
    
    # 3. Plot confusion matrix
    print("[4/6] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, id_to_label, model_type="BiLSTM",
                         save_path=IMAGES_DIR / "bilstm_confusion_matrix.png")
    
    # 4. Plot per-disease metrics
    print("[5/6] Plotting per-disease performance metrics...")
    plot_per_disease_metrics(y_true, y_pred, id_to_label, model_type="BiLSTM",
                            save_path=IMAGES_DIR / "bilstm_per_disease_metrics.png")
    
    # 5. Plot top confusions
    print("[6/6] Plotting top misclassifications...")
    plot_top_confusions(y_true, y_pred, id_to_label, model_type="BiLSTM",
                       top_n=15, save_path=IMAGES_DIR / "bilstm_top_confusions.png")
    
    # 6. Plot class distribution
    print("[Bonus] Plotting disease distribution...")
    plot_class_distribution(y_true, id_to_label,
                          save_path=IMAGES_DIR / "bilstm_class_distribution.png")
    
    # Print summary metrics
    print("\n" + "=" * 60)
    print("BiLSTM SUMMARY METRICS")
    print("=" * 60)
    metrics = compute_metrics(y_true, y_pred)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Micro F1:      {metrics['micro_f1']:.4f}")
    
    return metrics


def generate_transformer_visualizations():
    """
    Generate all visualizations for Transformer model.
    """
    print("\n" + "="*60)
    print("Generating Visualizations for Transformer Model")
    print("="*60)
    
    # 1. Load and plot training history
    print("\n[1/6] Plotting training curves...")
    history = load_training_history(ARTIFACTS_DIR / "transformer_training_history.json")
    plot_training_curves(history, model_type="Transformer",
                        save_path=IMAGES_DIR / "transformer_training_curves.png")
    
    # 2. Load test data and model for evaluation plots
    print("[2/6] Loading model and test data...")
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")
    
    # Load label encoder
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}
    
    # Load checkpoint and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(TRANSFORMER_CKPT_PATH, map_location=device)
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]
    
    tokenizer = load_tokenizer(model_name)
    
    test_ds = TransformerSymptomDataset(
        texts=splits.test["symptom_text"].tolist(),
        labels=splits.test["label_id"].tolist(),
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=collate_transformer_batch,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=ckpt["num_classes"],
    ).to(device)
    
    model.load_state_dict(ckpt["model_state_dict"])
    
    # Get predictions
    print("[3/6] Generating predictions on test set...")
    y_pred, y_true = predict_transformer(model, test_loader, device)
    
    # 3. Plot confusion matrix
    print("[4/6] Plotting confusion matrix...")
    plot_confusion_matrix(y_true, y_pred, id_to_label, model_type="Transformer",
                         save_path=IMAGES_DIR / "transformer_confusion_matrix.png")
    
    # 4. Plot per-disease metrics
    print("[5/6] Plotting per-disease performance metrics...")
    plot_per_disease_metrics(y_true, y_pred, id_to_label, model_type="Transformer",
                            save_path=IMAGES_DIR / "transformer_per_disease_metrics.png")
    
    # 5. Plot top confusions
    print("[6/6] Plotting top misclassifications...")
    plot_top_confusions(y_true, y_pred, id_to_label, model_type="Transformer",
                       top_n=15, save_path=IMAGES_DIR / "transformer_top_confusions.png")
    
    # 6. Plot class distribution (same for both models, but save separately)
    print("[Bonus] Plotting disease distribution...")
    plot_class_distribution(y_true, id_to_label,
                          save_path=IMAGES_DIR / "transformer_class_distribution.png")
    
    # Print summary metrics
    print("\n" + "="*60)
    print("Transformer SUMMARY METRICS")
    print("="*60)
    metrics = compute_metrics(y_true, y_pred)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Micro F1:      {metrics['micro_f1']:.4f}")
    
    return metrics


def generate_all_visualizations():
    """
    Main function to generate all training and evaluation visualizations
    for both BiLSTM and Transformer models.
    """
    # Create images directory
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations for both models
    bilstm_metrics = generate_bilstm_visualizations()
    transformer_metrics = generate_transformer_visualizations()
    
    # Print comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    print(f"{'Metric':<20} {'BiLSTM':>12} {'Transformer':>12}")
    print("-"*60)
    print(f"{'Test Accuracy':<20} {bilstm_metrics['accuracy']:>12.4f} {transformer_metrics['accuracy']:>12.4f}")
    print(f"{'Macro F1':<20} {bilstm_metrics['macro_f1']:>12.4f} {transformer_metrics['macro_f1']:>12.4f}")
    print(f"{'Micro F1':<20} {bilstm_metrics['micro_f1']:>12.4f} {transformer_metrics['micro_f1']:>12.4f}")
    
    print("\n" + "="*60)
    print(f"All visualizations saved to: {IMAGES_DIR}")
    print("="*60)


if __name__ == "__main__":
    generate_all_visualizations()
