#!/usr/bin/env python3

"""
Shared inference utilities for both models.

- The API should not duplicate model-loading and prediction logic.
- Centralizing inference reduces bugs and guarantees consistent behavior.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    ARTIFACTS_DIR,
    BILSTM_CKPT_PATH,
    TRANSFORMER_CKPT_PATH,
    LABEL_ENCODER_PATH,
    PRECAUTION_MAP_PATH,
)
from src.dataset_bilstm import Vocab, simple_tokenize
from models.model_bilstm import BiLSTMClassifier


@dataclass(frozen=True)
class Assets:
    label_to_id: Dict[str, int]
    id_to_label: Dict[int, str]
    precautions: Dict[str, List[str]]


def load_assets() -> Assets:
    """
    Loads JSON artifacts shared across models.

    - Ensures label decoding and precautions are identical at train and serve time.
    """
    with open(LABEL_ENCODER_PATH, "r", encoding="utf-8") as f:
        label_to_id = json.load(f)
    id_to_label = {v: k for k, v in label_to_id.items()}

    with open(PRECAUTION_MAP_PATH, "r", encoding="utf-8") as f:
        precautions = json.load(f)

    return Assets(label_to_id=label_to_id, id_to_label=id_to_label, precautions=precautions)


def load_vocab() -> Vocab:
    """
    Loads vocab.json for BiLSTM inference.

    - The BiLSTM requires the same token-to-id mapping used during training.
    """
    with open(ARTIFACTS_DIR / "vocab.json", "r", encoding="utf-8") as f:
        obj = json.load(f)
    itos = obj["itos"]
    stoi = {tok: i for i, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


# def build_symptom_text_from_user(symptoms: str) -> str:
#     """
#     Converts raw user input into the same text style seen during training.


#     - Train/inference mismatch is one of the most common deployment bugs.
#     - We normalize input similarly: lowercase, underscores, comma-separated list.
#     """
#     # Accept either a comma list or free text; keep it simple and robust.
#     normalized = symptoms.strip().lower().replace(" ", "_")
#     # Allow users to type "fever headache nausea" and still work.
#     normalized = normalized.replace("__", "_")
#     return "symptoms: " + normalized.replace(",", ", ")



def build_symptom_text_from_user(symptoms: str) -> str:
    """
    Normalizes user symptom input into training-style text.

    WHY:
    - Training data uses tokens like 'body_aches' not '_body_aches'.
    - Consistent normalization improves model reliability and makes outputs easier to read.
    """
    raw = symptoms.strip().lower()

    # Split on commas if present; otherwise split on whitespace
    if "," in raw:
        parts = [p.strip() for p in raw.split(",")]
    else:
        parts = raw.split()

    cleaned = []
    for p in parts:
        # Replace internal whitespace with underscores, remove leading/trailing underscores
        token = re.sub(r"\s+", "_", p)
        token = re.sub(r"^_+|_+$", "", token)
        if token:
            cleaned.append(token)

    return "symptoms: " + ", ".join(cleaned)



class BiLSTMPredictor:
    def __init__(self, device: torch.device):
        self.device = device
        self.vocab = load_vocab()

        ckpt = torch.load(BILSTM_CKPT_PATH, map_location=device)
        self.model = BiLSTMClassifier(
            vocab_size=ckpt["vocab_size"],
            num_classes=ckpt["num_classes"],
            pad_id=ckpt["pad_id"],
            embedding_dim=ckpt["embedding_dim"],
            hidden_dim=ckpt["hidden_dim"],
            dropout=ckpt["dropout"],
        ).to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_topk(self, symptom_text: str, k: int = 5) -> List[Tuple[int, float]]:
        """
        Returns top-k (class_id, probability).

        
        - For triage use-cases, showing alternatives is more useful than a single label.
        """
        tokens = simple_tokenize(symptom_text)
        ids = self.vocab.encode(tokens)

        input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long).to(self.device)

        logits = self.model(input_ids, lengths)
        probs = F.softmax(logits, dim=1).squeeze(0)

        topk = torch.topk(probs, k=min(k, probs.shape[0]))
        return [(int(i), float(p)) for i, p in zip(topk.indices, topk.values)]


class TransformerPredictor:
    def __init__(self, device: torch.device):
        self.device = device

        ckpt = torch.load(TRANSFORMER_CKPT_PATH, map_location="cpu")
        self.model_name = ckpt["model_name"]
        self.max_length = ckpt["max_length"]
        self.num_classes = ckpt["num_classes"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Load base model then load fine-tuned weights
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_classes,
        ).to(device)

        state = ckpt["model_state_dict"]
        # Handle previous wrapper prefix if it exists
        if any(k.startswith("backbone.") for k in state.keys()):
            state = {k.replace("backbone.", "", 1): v for k, v in state.items()}

        self.model.load_state_dict(state)
        self.model.eval()

    @torch.no_grad()
    def predict_topk(self, symptom_text: str, k: int = 5) -> List[Tuple[int, float]]:
        enc = self.tokenizer(
            symptom_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**enc)
        probs = F.softmax(outputs.logits, dim=1).squeeze(0)

        topk = torch.topk(probs, k=min(k, probs.shape[0]))
        return [(int(i), float(p)) for i, p in zip(topk.indices, topk.values)]
