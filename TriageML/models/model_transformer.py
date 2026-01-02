#!/usr/bin/env python3


"""
Transformer classifier model wrapper

- encapsulates the Huggingface model
- thin wrapper also makes easier to swap models (e.g., MiniLM) later
"""


from __future__ import annotations

import torch
import torch.nn as nn


class TransformerClassifier(nn.Module):
    def __init__(self, backbone):
        """
        backbone: a HuggingFace AutoModelForSequenceClassification instance
        
        - this gives a standard classification head.
        - forward pass returns loss (if labels provided) and logits.
        """
        super().__init__()
        self.backbone = backbone
        
    
    def forward(self, input_ids, attention_mask, labels=None):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )