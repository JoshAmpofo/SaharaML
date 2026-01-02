#!/usr/bin/env python3


"""
Evaluate the saved best BiLSTM checkpoint on the test split.
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import DataLoader

from src.preprocess import run_preprocessing
from src.split import make_group_splits
from src.dataset_bilstm import SymptomDataset, collate_batch, Vocab
from models.model_bilstm import BiLSTMClassifier
from src.evaluate import compute_metrics, pretty_report, predict_bilstm, top_confusions
from src.config import ARTIFACTS_DIR, BILSTM_CKPT_PATH, LABEL_ENCODER_PATH


def load_vocab(vocab_path):
    with open(vocab_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    itos = obj["itos"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def main():
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")

    # Load label encoder and invert it for readable reporting
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}

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

    y_pred, y_true = predict_bilstm(model, test_loader, device)

    metrics = compute_metrics(y_true, y_pred)
    print("TEST METRICS:", metrics)
    print("\nPER-CLASS REPORT:\n")
    print(pretty_report(y_true, y_pred, id_to_label))

    conf = top_confusions(y_true, y_pred, id_to_label, top_n=10)
    if conf:
        print("\nTOP CONFUSIONS (true → predicted):")
        for count, true_lbl, pred_lbl in conf:
            print(f"{count:4d} | {true_lbl} → {pred_lbl}")
    else:
        print("\nNo confusions on test set.")


if __name__ == "__main__":
    main()
