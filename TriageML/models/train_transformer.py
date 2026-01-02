#!/usr/bin/env python3


"""
Training script for Transformer model (DistilBERT)
"""


from __future__ import annotations

import json
from typing import Dict

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

from src.preprocess import run_preprocessing
from src.split import make_splits, make_group_splits
from src.dataset_transformer import TransformerSymptomDataset, collate_transformer_batch
from models.model_transformer import TransformerClassifier
from src.audit import audit_overlap, audit_duplicates
from src.config import ARTIFACTS_DIR, MODELS_DIR, TRANSFORMER_CKPT_PATH


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """
    computes accuracy for a batch
    
    - final reporting would use macro-F1 on test set
    """
    preds = torch.argmax(logits, dim=1)
    correct = (preds == labels).float().mean().item()
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
    
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits
        
        batch_acc = accuracy_from_logits(logits, labels)
        
        total_loss += loss.item()
        total_acc += batch_acc
        n_batches += 1
    
    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    
    return {
        "loss": avg_loss,
        "accuracy": avg_acc,
    }
    
    
def main():
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")
    
    dup_stats = audit_duplicates(df)
    overlap_stats = audit_overlap(splits.train, splits.val, splits.test)
    
    print("Data Audit Results:")
    print("Duplicates:", dup_stats)
    print("Overlaps:", overlap_stats)
    
    num_classes = df['label_id'].nunique()
    
    # DistilBERT is much smaller than BERT-base; faster to train on CPU/GPU
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # max-length=64 because symptom_text strings are short and short max_length dramatcally reduces compute cost
    train_ds = TransformerSymptomDataset(
        texts=splits.train['symptom_text'].tolist(),
        labels=splits.train['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=64,
    )
    val_ds = TransformerSymptomDataset(
        texts=splits.val['symptom_text'].tolist(),
        labels=splits.val['label_id'].tolist(),
        tokenizer=tokenizer,
        max_length=64,
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_transformer_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_transformer_batch,
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    backbone = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_classes,
    )
    model = TransformerClassifier(backbone=backbone)
    model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    
    # scheduler with warmup prevents unstable early updates and linear decay helps convergence
    num_training_steps = len(train_loader) * 5  # 5 epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    best_val_acc = -1.0
    best_epoch = -1
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # save tokenizer for reproducibility
    with open(ARTIFACTS_DIR / "transformer_tokenizer.json", "w", encoding="utf-8") as f:
        json.dump({
            "model_name": model_name,
            "max_length": 64,
            }, f, indent=4
        )
    
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }
    
    for epoch in range(1, 6):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_batches = 0
        
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            
            batch_acc = accuracy_from_logits(logits, labels)
            
            running_loss += loss.item()
            running_acc += batch_acc
            n_batches += 1
        
        train_loss = running_loss / n_batches
        train_acc = running_acc / n_batches
        val_metrics = evaluate(model, val_loader, device)
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_metrics["loss"])
        history["val_acc"].append(val_metrics["accuracy"])
        
        print(
            f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} "
            f"| val_loss={val_metrics['loss']:.4f} | val_acc={val_metrics['accuracy']:.4f}"
        )
        
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": model_name,
                    "num_classes": num_classes,
                    "max_length": 64,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                },
                TRANSFORMER_CKPT_PATH,
            )
            
            print(f" -> New best model saved with val_acc={best_val_acc:.4f} at epoch {best_epoch}")
    
    print(f"Best val_acc: {best_val_acc:.4f} at epoch {best_epoch}")
    # save training history
    with open(ARTIFACTS_DIR / "transformer_training_history.json", "w") as f:
        json.dump(history, f, indent=4)
    print(f"Saved best Transformer model to {TRANSFORMER_CKPT_PATH}")
    

if __name__ == "__main__":
    main()
    