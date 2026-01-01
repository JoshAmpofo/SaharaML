#!/usr/bin/env python3

"""
Dataset + vocabulary for BiLSTM baseline

- BiLSTM cannot consume raw text directly; it needs token IDs.
- A "train-only" vocabulary is built to avoid data leakage from val/test into training.
- Preprocessing will be kept in the code so inference uses the exact same rules. 
"""


from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from collections import Counter


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def simple_tokenize(text: str) -> List[str]:
    """
    Tokenize symptom text.
    
    - symptoms already normalized with underscores and commas
    - simple whitespace + punctuation split is enough and keeps this baseline lightweight
    """
    text = text.replace(",", " ")
    return [t for t in text.split() if t]


@dataclass(frozen=True)
class Vocab:
    """
    Token -> ID mapping
    
    - IDs are required for embedding lookup in the neural net.
    """
    stoi: Dict[str, int]
    itos: List[str]
    
    @property
    def pad_id(self) -> int:
        return self.stoi[PAD_TOKEN]
    
    @property
    def unk_id(self) -> int:
        return self.stoi[UNK_TOKEN]
    
    
    def encode(self, tokens: List[str]) -> List[int]:
        return [self.stoi.get(t, self.unk_id) for t in tokens]
    


def build_vocab(texts: List[str], min_freq: int = 1) -> Vocab:
    """
    Builds a vocabulary from training texts only.
    
    - Prevents train/val/test data leakage
    - min_freq allows pruning extremely rare tokens
    """
    counter = Counter()
    for t in texts:
        counter.update(simple_tokenize(t))
    
    itos = [PAD_TOKEN, UNK_TOKEN]
    for token, freq in counter.items():
        if freq >= min_freq and token not in (PAD_TOKEN, UNK_TOKEN):
            itos.append(token)
    
    stoi = {tok: idx for idx, tok in enumerate(itos)}
    return Vocab(stoi=stoi, itos=itos)


def collate_batch(
    batch: List[Tuple[List[int], int]],
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pads variable-length sequences into a batch tensor.
    
    Returns:
        - Padded token ID (input_ids) tensor of shape (batch_size, max_seq_len)
        - Lengths tensor of shape (batch_size,)
        - Labels tensor of shape (batch_size,)
    
    - RNNs need consistent tensor shapes per batch.
    - Return lengths so the BiLSTM can ignore padding using packing.
    """
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    
    max_len = max(lengths).item()
    input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
    
    return input_ids, lengths, torch.tensor(labels, dtype=torch.long)




class SymptomDataset(Dataset):
    """
    PyTorch dataset for symptom_text -> label_id.
    """
    def __init__(self, symptom_texts: List[str], labels: List[int], vocab: Vocab):
        self.texts = symptom_texts
        self.labels = labels
        self.vocab = vocab
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Tuple[List[int], int]:
        tokens = simple_tokenize(self.texts[idx])
        token_ids = self.vocab.encode(tokens)
        label = self.labels[idx]
        return token_ids, label
