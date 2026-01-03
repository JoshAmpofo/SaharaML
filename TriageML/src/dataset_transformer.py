#!/usr/bin/env python3

"""
Transformer dataset + collation

- Transformers require tokenized inputs (input_ids, attention_mask)
"""


from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class TransformerSymptomDataset(Dataset):
    """
    Dataset producing tokenized transformer inputs for symptom_text classification.
    
    - returning a dict per item allows a clean collate_fn to stack batches.
    """
    def __init__(
        self,
        texts: List[str],
        labels: List[int],
        tokenizer,
        max_length: int = 64
    ):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    
    def __len__(self) -> int:
        return len(self.texts)
    
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        label = int(self.labels[idx])
        
        encoding = self.tokenizer(
            text,
            padding='max_length', # ensures tensors can be stacked into a batch
            truncation=True, # ensures a fixed upper bound on compute and memory
            max_length=self.max_length,
            return_tensors='pt',
        )
        # tokenizer returns shape (1, T); squeeze to (T,)
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }
        
        return item
    
    
def collate_transformer_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    collates list of dicts into a single dict of stacked tensors
    
    - Huggingface models accept batched dict inputs: input_ids, attention_mask, labels
    """
    out = {}
    keys = batch[0].keys()
    for key in keys:
        out[key] = torch.stack([item[key] for item in batch], dim=0)
    return out
