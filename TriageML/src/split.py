#!/usr/bin/env python3

"""
Train/val/test split logic

- stratified splitting avoids accidentally excluding rare diseases from validation/test.
- keeping splits deterministic makes the results reproducible.
"""


from dataclasses import dataclass
from typing import Tuple

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