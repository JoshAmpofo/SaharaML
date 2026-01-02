#!/usr/bin/env python3


"""
Data audit utilities to detect leakage and overly-optimistic validation results.

WHY this exists:
- Extremely high validation accuracy can be legitimate, but it can also indicate leakage.
- Auditing improves trust in the results and strengthens the project narrative.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd


def audit_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, int]:
    """
    Checks if exact symptom_text rows appear across splits.

    WHY:
    - If the same symptom pattern appears in both train and val, the model can memorize rather than generalize.
    - This is common in symptom datasets that include repeated symptom combinations.
    """
    train_texts = set(train_df["symptom_text"].tolist())
    val_texts = set(val_df["symptom_text"].tolist())
    test_texts = set(test_df["symptom_text"].tolist())

    return {
        "train_val_overlap": len(train_texts.intersection(val_texts)),
        "train_test_overlap": len(train_texts.intersection(test_texts)),
        "val_test_overlap": len(val_texts.intersection(test_texts)),
    }


def audit_duplicates(df: pd.DataFrame) -> Dict[str, int]:
    """
    Checks duplicates in the full dataset.

    WHY:
    - High duplication rates can inflate validation performance.
    """
    return {
        "total_rows": len(df),
        "duplicate_symptom_text_rows": int(df.duplicated(subset=["symptom_text"]).sum()),
        "duplicate_symptom_text_and_label_rows": int(df.duplicated(subset=["symptom_text", "label_id"]).sum()),
    }
