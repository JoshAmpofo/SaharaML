#!/usr/bin/env python3

"""
Training script for BiLSTM baseline.

- keeps training loop reproducible
- saves artifacts (vocab + model) for later inference
"""


from __future__ import annotations

import json
from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.preprocess import run_preprocessing
from src.split import make_splits, make_group_splits
from src.dataset_bilstm import SymptomDataset, build_vocab, collate_batch
from models.model_bilstm import BiLSTMClassifier
from src.audit import audit_overlap, audit_duplicates
from src.config import ARTIFACTS_DIR, BILSTM_CKPT_PATH



def save_vocab(vocab, path) -> None:
    """
    saves vocabulary for inference.
    
    - if vocab changes between training and inference, model performance will degrade
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"itos": vocab.itos}, f, indent=4)
        

def accuracy(pred: torch.Tensor, y: torch.Tensor) -> float:
    """
    computes accuracy given logits and true labels
    """
    preds = torch.argmax(pred, dim=1)
    correct = (preds == y).float().mean().item()
    return correct


@torch.no_grad()
def evaluate(model, loader, device) -> Dict[str, float]:
    """
    evaluates model on given dataset
    """
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    for input_ids, lengths, labels in loader:
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        
        logits = model(input_ids, lengths)
        loss = loss_fn(logits, labels)
        
        total_loss += loss.item()
        total_acc += accuracy(logits, labels)
        n_batches += 1
        
    return {
        "loss": total_loss / n_batches,
        "acc": total_acc / n_batches
    }
    


def main() -> None:
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")
    
    dup_stats = audit_duplicates(df)
    overlap_stats = audit_overlap(splits.train, splits.val, splits.test)
    
    print("Data Audit Results:")
    print("Duplicates:", dup_stats)
    print("Overlaps:", overlap_stats)
    
    # build vocab from TRAIN only to aovid data leakage
    vocab = build_vocab(splits.train['symptom_text'].tolist(), min_freq=2)
    
    train_ds = SymptomDataset(
        symptom_texts=splits.train["symptom_text"].tolist(),
        labels=splits.train['label_id'].tolist(),
        vocab=vocab,
    )
    val_ds = SymptomDataset(
        symptom_texts=splits.val["symptom_text"].tolist(),
        labels=splits.val['label_id'].tolist(),
        vocab=vocab,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_id=vocab.pad_id),
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        num_classes=df['label_id'].nunique(),
        pad_id=vocab.pad_id,
        embedding_dim=128,
        hidden_dim=128,
        dropout=0.2,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-2)
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # save vocab for reproducibility
    save_vocab(vocab, ARTIFACTS_DIR / "vocab.json")
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    # train for just 10 epochs
    for epoch in range(1, 11):
        model.train()
        running_loss = 0.0
        train_acc = 0.0
        n_batches = 0
        
        for input_ids, lengths, labels in train_loader:
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            logits = model(input_ids, lengths)
            loss = loss_fn(logits, labels)
            
            loss.backward()
            
            # gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item()
            train_acc += accuracy(logits, labels)
            n_batches += 1
            
        train_loss = running_loss / n_batches
        train_acc = train_acc / n_batches
        val_metrics = evaluate(model, val_loader, device)
        
        # Track metrics
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["acc"])
        
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['acc']:.4f}"
        )
        
        # save best model based on val accuracy
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            torch.save(
                {
                    "model_state_dict": model.state_dict(), 
                    "vocab_size": len(vocab.itos),
                    "num_classes": df["label_id"].nunique(),
                    "pad_id": vocab.pad_id,
                    "embedding_dim": 128,
                    "hidden_dim": 128,
                    "dropout": 0.2,
                    "best_val_acc": best_val_acc,
                    "best_epoch": epoch,
                    "voacb_artifact": str(ARTIFACTS_DIR / "vocab.json"),
                    },
                BILSTM_CKPT_PATH
            )
            print(f"  → New best model saved with val_acc={best_val_acc:.4f}")
    
    # Save history at end
    with open(ARTIFACTS_DIR / "bilstm_history.json", "w") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()