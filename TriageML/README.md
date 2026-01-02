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
- Two splitting strategies:
  - `make_splits`: Classic stratified row-level split
  - `make_group_splits`: **Stratified group-aware split** to prevent data leakage
- Group-aware splitting ensures:
  - Identical `symptom_text` values never appear in multiple splits
  - Each disease's symptom groups are proportionally distributed
  - All diseases guaranteed to appear in training set
- Separation of concerns:
  - Splitting logic operates at the **row level**
  - Feature/target selection is deferred to model-specific datasets

### 5️⃣ BiLSTM Baseline Model — **✅ COMPLETE**

A production-ready BiLSTM classifier has been implemented and trained end-to-end.

#### Model Architecture (`models/model_bilstm.py`)

- Bidirectional LSTM with:
  - Embedding layer with padding support
  - 2-layer BiLSTM (128 hidden units per direction)
  - Dropout regularization (0.2)
  - Linear classifier head
- Implements sequence packing to handle variable-length inputs efficiently
- Key improvements made:
  - Fixed critical typos (e.g., `num_embeddings`, `batch_first`, `nn.Linear`)
  - Proper hidden state extraction for bidirectional LSTM
  - Gradient clipping to prevent exploding gradients

#### Dataset Implementation (`src/dataset_bilstm.py`)

- Custom PyTorch `Dataset` for symptom text
- Vocabulary built from **training data only** to prevent leakage
- Features:
  - Simple tokenization optimized for symptom text
  - Token-to-ID mapping with `<pad>` and `<unk>` tokens
  - Dynamic batching with proper padding
  - Empty sequence validation (ensures minimum length)
- Collation function returns:
  - Padded input tensors
  - Sequence lengths (for packing)
  - Labels

#### Training Pipeline (`models/train_bilstm.py`)

- 10-epoch training with early stopping based on validation accuracy
- Features implemented:
  - Train/validation accuracy and loss tracking
  - Gradient clipping (max norm = 1.0)
  - Best model checkpointing
  - Training history saved to JSON for visualization
- Fixed critical bugs:
  - Accuracy calculation (boolean to float conversion)
  - Tuple initialization error
  - Print statement indentation
- Training metrics tracked per epoch:
  - Train loss & accuracy
  - Validation loss & accuracy

#### Evaluation & Visualization (`src/evaluate_bilstm.py`, `src/visualize_results.py`)

- Comprehensive evaluation metrics:
  - Accuracy, Macro-F1, Micro-F1
  - Per-class precision/recall/F1
  - Confusion matrix analysis
  - Top misclassification pairs
- **Medical-relevant visualizations**:
  1. Training curves (loss & accuracy)
  2. Confusion matrix heatmap (normalized)
  3. Per-disease performance metrics
  4. Top misclassifications (disease confusion patterns)
  5. Disease distribution in test set
  6. Class imbalance analysis
- All visualizations saved to `images/` directory at 300 DPI

#### Results

- **Test Accuracy**: ~99%+ on validation set
- **Macro F1**: High performance across all disease classes
- Model artifacts saved:
  - `artifacts/bilstm.pt` (model checkpoint)
  - `artifacts/vocab.json` (vocabulary)
  - `artifacts/bilstm_history.json` (training history)

## 🧱 Current Project Structure

```bash
TriageML/
├── pyproject.toml
├── README.md
├── artifacts/
│   ├── bilstm.pt
│   ├── bilstm_history.json
│   ├── label_encoder.json
│   ├── precaution_map.json
│   ├── transformer.pt
│   └── vocab.json
├── data/
│   ├── DiseaseAndSymptoms.csv
│   └── Disease precaution.csv
├── images/
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_disease_metrics.png
│   ├── top_confusions.png
│   └── class_distribution.png
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
│   ├── evaluate_bilstm.py
│   ├── preprocess.py
│   ├── serve.py
│   ├── split.py
│   └── visualize_results.py
└── tests/
```

## 🚧 Work In Progress / Next Steps

The following components are planned and will be implemented next:

### 🔜 Transformer-Based Model

- **Model 2: Transformer-based Classifier (PyTorch)**
  - Fine-tuned pretrained language model (e.g., DistilBERT)
  - Comparison with BiLSTM baseline performance
  - Leverage transfer learning from large-scale pretraining
  - Expected improvements in handling complex symptom descriptions

### 🔜 Model Comparison & Analysis

- Side-by-side performance comparison:
  - BiLSTM vs Transformer
  - Inference speed benchmarks
  - Model size trade-offs
- Error analysis across both models
- Identify strengths/weaknesses of each architecture

### 🔜 Inference & Deployment

- **AWS Lambda Deployment**
  - Serverless inference endpoint
  - Automatic scaling based on request load
  - Cost-efficient pay-per-request model
- **FastAPI Service on GCP**
  - RESTful API with comprehensive documentation
  - Endpoints:
    - `GET /health` - Service health check
    - `POST /predict` - Disease prediction from symptoms
    - `GET /models` - List available models
  - API responses include:
    - Predicted disease
    - Confidence scores (top-k predictions)
    - Precautionary guidance
  - Model selection support (BiLSTM vs Transformer)
- **Containerization**
  - Docker image for reproducible deployment
  - Optimized for inference performance

## 🎯 Project Goals

**TriageML** aims to deliver a fully documented, reproducible, and deployable machine learning system that demonstrates:

- ✅ **Strong ML Engineering Practices**
  - Reproducible preprocessing and data splitting
  - Prevention of data leakage through group-aware splits
  - Comprehensive model evaluation and visualization
  - Artifact versioning and model checkpointing

- ✅ **Deep Learning Proficiency in PyTorch**
  - Custom BiLSTM architecture implementation
  - Proper handling of variable-length sequences
  - Training pipeline with best practices (gradient clipping, early stopping)
  - Efficient data loading and batching

- 🔄 **Real-World Applicability**
  - Medical-relevant evaluation metrics
  - Confusion analysis for clinical insights
  - Production-ready code structure
  - API deployment for real-time inference (in progress)

## 📊 Key Achievements

- ✅ **Data Quality**: Handled 4,920 records across 41 disease classes with proper preprocessing
- ✅ **Model Performance**: Achieved 99%+ validation accuracy with BiLSTM baseline
- ✅ **Reproducibility**: All artifacts, configs, and random seeds versioned
- ✅ **Visualization**: Medical-focused performance analysis and error detection
- ✅ **Code Quality**: Production-ready codebase with proper error handling

## 📌 Disclaimer

This project is for **educational purposes only** and is not intended for real clinical diagnosis or treatment decisions.
