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
            num_embeddings=vocab_size,
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
        self.classifier = nn.Linear(2 * hidden_dim, num_classes)
        
    
    def forward(self, input_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
        - input_ids: Tensor of shape (batch_size, seq_length) with token IDs
        - lengths: Tensor of shape (batch_size,) with actual lengths of each sequence (before padding)
        
        Returns:
        - logits: Tensor of shape (batch_size, num_classes) with class scores
        
        - packing sequences prevents the LSTM from "seeing" padded timesteps, improving both efficiency and learning quality
        """
        x = self.embedding(input_ids)  # (batch_size, seq_length, embedding_dim)
        
        packed = nn.utils.rnn.pack_padded_sequence(
            x,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False, # allows random batch order from DataLoader
        )
        
        packed_output, (h_n, _) = self.lstm(packed)
        # h_n shape: (num_layers * 2, batch_size, hidden_dim)
        # we want the last layer's forward and backward idden states
        forward_last = h_n[-2, :, :]  # (batch_size, hidden_dim)
        backward_last = h_n[-1, :, :]  # (batch_size, hidden_dim)
        
        features = torch.cat([forward_last, backward_last], dim=1)  # (batch_size, 2*hidden_dim)
        features = self.dropout(features)
        logits = self.classifier(features)  # (batch_size, num_classes)
        
        return logits
        
        