#!/usr/bin/env python3

"""
BiLSTM classifier baseline

- strong, classic neural baseline for text classification
- learns sequence patterns without needing pre-training
- fast to train and provides meaningful comparison point vs transformers
"""


from __future__ import annotations

import torch
import torch.nn as nn


class BiLSTMClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        pad_id: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float =0.2,
    ):
        super().__init__()
        
        # padding_idx ensures PAD tokens get zero embeddings and not learn semantics
        self.embedding = nn.Embedding(
            num_embedings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_id,
        )
        
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        
        self.dropout = nn.Dropout(dropout)
        # BiLSTM outputs 2*hidden_dim because forward and backward states are concatenated
        self.classifier = nn.Layer(2 * hidden_dim, num_classes)
        
        