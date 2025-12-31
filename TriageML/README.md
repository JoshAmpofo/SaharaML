# TriageML 🩺  

## **An End-to-End Machine Learning System for Automated Medical Symptom Triage**

## 📌 Problem Statement

Healthcare systems, especially in resource-constrained settings, often face challenges with timely medical triage. Patients typically present with **free-form descriptions of symptoms**, and determining the likely condition and appropriate next steps (urgency, precautions) requires clinical expertise that may not always be immediately available.

Manual triage is:

- time-consuming,
- inconsistent across practitioners,
- and difficult to scale with increasing patient load.

**TriageML** aims to address this challenge by leveraging **machine learning and natural language processing (NLP)** to automatically analyze symptom descriptions and predict the most likely disease category, while also providing relevant precautionary guidance.

## 🤖 Why Machine Learning?

This problem is well-suited to machine learning because:

- Symptom descriptions can be represented as **text data**, which modern NLP models handle effectively.
- Disease prediction is a **multi-class classification** problem, well supported by neural networks.
- ML models can learn symptom–disease patterns from historical data and generalize to unseen cases.
- Once trained, models can provide **fast, consistent, and scalable triage support**.

The project is designed as an **end-to-end ML system**, covering data preprocessing, model training, evaluation, and deployment via an API.

## 📂 Dataset Overview

The project uses two related datasets:

1. **DiseaseAndSymptoms.csv**
   - 4,920 records
   - 41 unique diseases
   - 17 symptom columns per record (`Symptom_1` … `Symptom_17`)
   - Each row represents a patient case with a variable number of symptoms

2. **disease precaution.csv**
   - Disease-level precautionary recommendations
   - Up to 4 precautions per disease

Null values in symptom columns indicate the **absence of additional symptoms**, not missing data.

## 🧠 Project Approach

The overall approach is to:

1. Convert structured symptom columns into **free-text symptom descriptions**
2. Train multiple neural network models to classify diseases from text
3. Enrich predictions with precautionary guidance
4. Serve predictions through a clean, documented API

## ✅ Work Completed So Far

### 1️⃣ Data Exploration & Cleaning (`data_exploration.ipynb`)

- Inspected dataset shape, schema, and class distribution
- Verified disease overlap between symptom and precaution datasets
- Normalized column names and values (lowercase, underscores)
- Confirmed:
  - 4,920 total records
  - 41 matching diseases across both datasets
- Identified that null values represent **variable-length symptom lists**, not missing data

### 2️⃣ Centralized Configuration (`src/config.py`)

- Defined all file paths and artifact locations in one place
- Ensures:
  - reproducibility
  - clean separation between code and environment-specific paths
- Paths include:
  - raw data
  - preprocessing artifacts
  - model checkpoints

### 3️⃣ Preprocessing Pipeline (`src/preprocess.py`)

A reusable preprocessing pipeline was implemented to ensure **training–inference consistency**.

Key steps:

- **Symptom text construction**
  - Collapsed `Symptom_1`–`Symptom_17` into a single free-text field
  - Example:

    ```bash
    symptoms: itching, skin_rash, nodal_skin_eruptions
    ```

- **Label encoding**
  - Created a stable `Disease → label_id` mapping
  - Saved as `label_encoder.json`
- **Precaution mapping**
  - Converted precaution columns into structured lists
  - Saved as `precaution_map.json`
- **Artifact persistence**
  - All mappings saved as JSON to ensure identical behavior during inference

After preprocessing:

- All rows contain:
  - `symptom_text`
  - `label_id`
  - `precaution_list`
- No null values remain in model inputs

### 4️⃣ Stratified Data Splitting (`src/split.py`)

- Implemented deterministic **train / validation / test splits**
- Stratified by `label_id` to preserve class distribution
- Separation of concerns:
  - Splitting logic operates at the **row level**
  - Feature/target selection is deferred to model-specific datasets

## 🧱 Current Project Structure

```bash
TriageML/
├── pyproject.toml
├── README.md
├── artifacts/
│   ├── bilstm.pt
│   ├── label_encoder.json
│   ├── precaution_map.json
│   ├── transformer.pt
│   └── vocab.json
├── data/
│   ├── DiseaseAndSymptoms.csv
│   └── Disease precaution.csv
├── models/
│   ├── model_bilstm.py
│   ├── model_transformer.py
│   ├── train_bilstm.py
│   └── train_transformer.py
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── config.py
│   ├── dataset_bilstm.py
│   ├── dataset_transformer.py
│   ├── evaluate.py
│   ├── preprocess.py
│   ├── serve.py
│   └── split.py
└── tests/
```

## 🚧 Work In Progress / Next Steps

The following components are planned and will be implemented next:

### 🔜 Model Development

- **Model 1: BiLSTM Classifier (PyTorch)**
  - Baseline neural network for symptom text classification
  - Built from scratch to demonstrate core deep learning fundamentals
- **Model 2: Transformer-based Classifier (PyTorch)**
  - Fine-tuned pretrained language model (e.g. DistilBERT)
  - Used to compare performance against the BiLSTM baseline

### 🔜 Model Evaluation

- Accuracy and **macro-F1 score** (to account for class imbalance)
- Confusion matrix analysis for selected diseases

### 🔜 Inference & Deployment

- Containerization unsing Docker
- Possible kubernetes serving
- FastAPI service exposing:
  - `/health`
  - `/predict`
- API responses will include:
  - predicted disease
  - confidence scores
  - precautionary guidance
- Support for switching between trained models
- Model will be deployed to GCP

## 🎯 Project Goal

The final goal of **TriageML** is to deliver a fully documented, reproducible, and deployable machine learning system that demonstrates:

- strong ML engineering practices
- deep learning proficiency in PyTorch
- real-world applicability in healthcare triage

## 📌 Disclaimer

This project is for **educational purposes only** and is not intended for real clinical diagnosis or treatment decisions.
