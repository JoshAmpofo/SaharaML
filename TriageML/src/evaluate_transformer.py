#!/usr/bin/env python3

"""
Test set evaluation for Transformer model
"""


from __future__ import annotations

import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.preprocess import run_preprocessing
from src.split import make_group_splits
from src.dataset_transformer import TransformerSymptomDataset, collate_transformer_batch
from src.evaluate import compute_metrics, pretty_report, top_confusions
from src.config import TRANSFORMER_CKPT_PATH, LABEL_ENCODER_PATH


@torch.no_grad()
def predict_transformer(model, loader, device):
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


def main():
    df = run_preprocessing()
    splits = make_group_splits(df, group_col="symptom_text")

    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}

    ckpt = torch.load(TRANSFORMER_CKPT_PATH, map_location="cpu")
    model_name = ckpt["model_name"]
    max_length = ckpt["max_length"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=ckpt["num_classes"],
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])

    y_pred, y_true = predict_transformer(model, test_loader, device)

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
