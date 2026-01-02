#!/usr/bin/env python3

"""
Train/val/test split logic

- make_splits: classic stratified row split
- make_group_splits: group-aware split to prevent symptom_text leakage

- stratified splitting avoids accidentally excluding rare diseases from validation/test.
- keeping splits deterministic makes the results reproducible.
"""


from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def make_splits(
        df: pd.DataFrame,
        test_size: float = 0.20,
        val_size: float = 0.20,
        seed: int = 42,
) -> Splits:
    """
    makes stratified train/val/test splits from the full dataset, stratifying on label_id.

    Args:
        df: Full DataFrame to split.
        test_size: Proportion of data to allocate to the test set.
        val_size: Proportion of data to allocate to the validation set.
        seed: Random seed for reproducibility.
    
    Returns:
        Splits dataclass containing train, val, and test DataFrames.
    """
    full_temp_df, full_test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label_id"],
    )

    # split temp_df into val and test
    relative_val = val_size / (1 - test_size)

    full_train_df, val_df = train_test_split(
        full_temp_df,
        test_size=relative_val,
        random_state=seed,
        stratify=full_temp_df["label_id"],
    )

    return Splits(
        train=full_train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=full_test_df.reset_index(drop=True), 
    )
    

def make_group_splits(
        df: pd.DataFrame,
        group_col: str = "symptom_text",
        test_size: float = 0.20,
        val_size: float = 0.20,
        seed: int = 42,
) -> Splits:
    """
    Makes stratified group-aware train/val/test splits to prevent leakage by identical symptom_text
    while maintaining label_id distribution.

    Args:
        df: Full DataFrame to split.
        group_col: Column name to use for grouping (e.g., symptom_text).
        test_size: Proportion of data to allocate to the test set.
        val_size: Proportion of data to allocate to the validation set.
        seed: Random seed for reproducibility.
    
    Returns:
        Splits with stratified group distribution across train/val/test.
    
    Note:
        For diseases with only 1-2 groups, perfect stratification is impossible.
        Groups are assigned to maximize class balance across splits.
    """
    rng = np.random.default_rng(seed)
    
    train_groups = []
    val_groups = []
    test_groups = []
    
    # For each disease (label_id), split its groups proportionally
    for label_id in df["label_id"].unique():
        # Get all groups (symptom_text) for this disease
        disease_df = df[df["label_id"] == label_id]
        groups = disease_df[group_col].unique().tolist()
        
        # Shuffle groups for this disease
        rng.shuffle(groups)
        
        n_groups = len(groups)
        
        # Calculate splits for this disease
        n_test = max(1, int(round(n_groups * test_size)))
        n_val = max(1, int(round(n_groups * val_size)))
        n_train = n_groups - n_test - n_val
        
        # Handle edge case: if disease has very few groups
        if n_train < 1:
            # Ensure at least 1 group in train if possible
            if n_groups == 1:
                # Only 1 group: put it in train
                n_train, n_val, n_test = 1, 0, 0
            elif n_groups == 2:
                # 2 groups: train=1, val or test=1
                n_train, n_val, n_test = 1, 1, 0
            else:
                # 3+ groups: train=1, distribute rest
                n_train = 1
                n_test = min(n_test, n_groups - n_train - 1)
                n_val = n_groups - n_train - n_test
        
        # Assign groups to splits
        test_groups.extend(groups[:n_test])
        val_groups.extend(groups[n_test:n_test + n_val])
        train_groups.extend(groups[n_test + n_val:])
    
    # Convert to sets for efficient lookup
    test_groups_set = set(test_groups)
    val_groups_set = set(val_groups)
    train_groups_set = set(train_groups)
    
    # Create DataFrames based on group assignments
    train_df = df[df[group_col].isin(train_groups_set)].copy()
    val_df = df[df[group_col].isin(val_groups_set)].copy()
    test_df = df[df[group_col].isin(test_groups_set)].copy()
    
    # Validate: ensure all diseases present in training
    train_diseases = set(train_df["label_id"].unique())
    all_diseases = set(df["label_id"].unique())
    if train_diseases != all_diseases:
        missing = all_diseases - train_diseases
        print(f"Warning: {len(missing)} disease(s) missing from training set: {missing}")
    
    # reset indices for clean DataLoaders and reproducibility
    return Splits(
        train=train_df.reset_index(drop=True),
        val=val_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True), 
    )