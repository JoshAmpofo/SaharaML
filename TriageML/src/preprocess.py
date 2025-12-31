#!/usr/bin/env python3

"""
Preprocessing for symptom -> disease classification + precaution enrichment.

- Both neural networks must receive the same *semantic* inputs (symptom text) and labels.
- Centralizing preprocessing prevents train/inference skew (a common production bug).
"""


import json
from typing import Dict, List, Tuple

import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    LABEL_ENCODER_PATH,
    PRECAUTION_MAP_PATH,
    PRECAUTIONS_CSV,
    SYMPTOMS_CSV,
)


def _normalize_token(text: str) -> str:
    """
    normalizes symptom/precaution tokens by stripping whitespace and lowercasing.
    
    This reduces the vocabulary size, helping small datasets generalize better.
    """
    return str(text).strip().lower().replace(" ", "_")


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    loads the raw datasets from CSV files.
    
    Returns:
        symptoms_df: DataFrame containing disease-symptom mappings.
        precautions_df: DataFrame containing disease-precaution mappings.
    """
    symptoms_df = pd.read_csv(SYMPTOMS_CSV)
    precautions_df = pd.read_csv(PRECAUTIONS_CSV)
    return symptoms_df, precautions_df


def build_symptom_text(symptoms_df: pd.DataFrame) -> pd.DataFrame:
    """
    constructs a 'Symptom_Text' column by concatenating all symptom columns into a single free-text field.

    - NaNs here mean 'no more symptoms', not 'missing data'.
    - Sequence models (BiLSTM/Transformers) work best with variable-length text inputs.

    Args:
        df: DataFrame containing disease-symptom mappings.
    Returns:
        DataFrame with an added 'Symptom_Text' column.
    """
    symptom_cols = [c for c in symptoms_df.columns if c.startswith("Symptom_")]

    def row_to_text(row) -> str:
        symptoms = []
        for c in symptom_cols:
            if pd.notna(row[c]):
                symptoms.append(_normalize_token(row[c]))
        # prefix with 'symptoms:' helps the transformer treat the text as a semantic field
        return "symptoms: " + ", ".join(symptoms)
    
    symptoms_df = symptoms_df.copy()
    symptoms_df["symptom_text"] = symptoms_df.apply(row_to_text, axis=1)

    return symptoms_df


def build_precaution_map(precautions_df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    converts the precaution table into a Disease -> [precautions] dictionary.

    - at inference time, the API needs O(1) lookup for precautions after predicting a disease.
    - keeping this map as a JSON artifact ensures consistency between training and serving.

    Args:
        precautions_df: DataFrame containing disease-precaution mappings.
    
    Returns:
        Dictionary mapping diseases to lists of normalized precautions.
    """
    precaution_cols = [c for c in precautions_df.columns if c.startswith("Precaution_")]

    mapping: Dict[str, List[str]] = {}
    for _, row in precautions_df.iterrows():
        disease = _normalize_token(row["Disease"])
        precautions = [
            _normalize_token(row[c])
            for c in precaution_cols
            if pd.notna(row[c])
        ]
        mapping[disease] = precautions
    
    return mapping


def build_label_encoder(diseases: List[str]) -> Dict[str, int]:
    """
    creates a stable mapping Disease -> class_id.

    - neural networks output class IDs, which must be mapped back to diseases at inference time.
    - saving mapping guarantees API decodes predictions correctly.

    Args:
        diseases: List of disease names from the dataset.
    
    Returns:
        Dictionary mapping diseases to integer class IDs.
    """
    unique = sorted(set(diseases))
    return {disease: idx for idx, disease in enumerate(unique)}


def save_json(obj: Dict, path: str) -> None:
    """
    saves a dictionary as a JSON file.

    Args:
        obj: Dictionary to save.
        path: File path to save the JSON.
    """
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=4)


def run_preprocessing() -> pd.DataFrame:
    """
    main preprocessing pipeline.

    - loads raw data
    - builds symptom text inputs
    - creates label encoder and precaution map artifacts
    - saves artifacts to disk

    Returns:
        DataFrame ready for model training with 'symptom_text' and 'Disease' columns.
    """
    # load raw datasets
    symptoms_df, precautions_df = load_raw_data()

    # normalize disease labels early so joins and encoders are consistent
    symptoms_df = symptoms_df.copy()
    symptoms_df["Disease"] = symptoms_df["Disease"].apply(_normalize_token)

    precautions_df = precautions_df.copy()
    precautions_df["Disease"] = precautions_df["Disease"].apply(_normalize_token)

    # build symptom text inputs
    processed_df = build_symptom_text(symptoms_df)

    precaution_map = build_precaution_map(precautions_df)
    save_json(precaution_map, PRECAUTION_MAP_PATH)

    # create and save label encoder
    label_encoder = build_label_encoder(processed_df["Disease"].tolist())
    save_json(label_encoder, LABEL_ENCODER_PATH)

    # add numeric labels for training
    processed_df['label_id'] = processed_df['Disease'].map(label_encoder)

    # add precaution for downstream API response (not needed for training but useful for integration tests)
    processed_df["precaution_list"] = processed_df["Disease"].apply(lambda d: precaution_map.get(d, []))

    # keep only what is needed for training
    keep_cols = ["Disease", "symptom_text", "label_id", "precaution_list"]
    processed_df = processed_df[keep_cols] 

    return processed_df


if __name__ == "__main__":
    df = run_preprocessing()
    print(f"Preprocessing complete. Processed {len(df)} records.")
    print(df.head(5))
    print("Rows:", len(df))
    print("classes:", df["label_id"].nunique())