#!/usr/bin/env python3

"""
Central configuration module for TriageML project

- Keeping paths and constraints in one place prevents "magic values" scattered throughout the codebase.
- Makes pipeline reproducible and easy to run from any machine
"""


from pathlib import Path


# Project root paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"
IMAGES_DIR = PROJECT_ROOT / "images"

# Data paths
SYMPTOMS_CSV = DATA_DIR / "DiseaseAndSymptoms.csv"
PRECAUTIONS_CSV = DATA_DIR / "Disease precaution.csv"

# Output artifacts saved after preprocessing
LABEL_ENCODER_PATH = ARTIFACTS_DIR / "label_encoder.json"
PRECAUTION_MAP_PATH = ARTIFACTS_DIR / "precaution_map.json"

# Model checkpoints
BILSTM_CKPT_PATH = MODELS_DIR / "bilstm.pt"
TRANSFORMER_CKPT_PATH = MODELS_DIR / "transformer.pt"