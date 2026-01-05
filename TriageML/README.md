# TriageML 🩺 

### An End-to-End Machine Learning System for Automated Medical Symptom Triage



## 📌 Problem Statement

Healthcare systems, particularly in resource-constrained environments, often struggle with timely and consistent medical triage. Patients typically present with **free-form descriptions of symptoms**, and determining the most likely condition — along with appropriate precautions — requires trained clinical expertise that may not always be immediately available.

Manual triage is:
- time-consuming,
- inconsistent across practitioners,
- and difficult to scale with increasing patient volumes.

**TriageML** addresses this challenge by using **machine learning and natural language processing (NLP)** to analyze symptom descriptions and automatically:
- predict the most likely disease category,
- provide confidence-aware predictions,
- and recommend basic precautionary guidance.

> ⚠️ **Note:** TriageML is an educational system designed to demonstrate ML engineering practices. It is **not a clinical diagnostic tool**.



## 🤖 Why Machine Learning?

This problem is well-suited to machine learning because:

- Symptom descriptions can be represented as **text data**, which modern NLP models handle effectively.
- Disease prediction is a **multi-class classification** problem (41 disease classes).
- ML models can learn symptom–disease patterns from historical data and generalize to new cases.
- Once trained, models provide **fast, consistent, and scalable triage support**.

TriageML is designed as a **fully end-to-end ML system**, covering:
- data preprocessing,
- model training and evaluation,
- visualization,
- and deployment via an API.



## 📂 Dataset Overview

Two publicly available datasets were used:

### 1️⃣ DiseaseAndSymptoms.csv
- **4,920 records**
- **41 unique diseases**
- Up to **17 symptoms per case**
- Symptoms are stored across multiple columns (`Symptom_1` … `Symptom_17`)
- Null values indicate **absence of additional symptoms**, not missing data

### 2️⃣ Disease precaution.csv
- Disease-level precautionary recommendations
- Up to **4 precautions per disease**

Both datasets were aligned and validated to ensure consistent disease labels.



## 🧠 Project Approach

The overall pipeline follows these steps:

1. Convert structured symptom columns into **free-text symptom descriptions**
2. Train **two neural network models** for disease classification
3. Evaluate models using robust, leakage-free splits
4. Visualize training dynamics and performance
5. Serve predictions through a **FastAPI-based service**
6. Containerize the application using Docker



## 🏗️ System Architecture (High Level)

User Symptoms
↓
Preprocessing & Normalization
↓
ML Model (BiLSTM or Transformer)
↓
Top-K Disease Predictions
↓
Precautionary Guidance
↓
FastAPI Service (Dockerized)



## ✅ Work Completed

### 1️⃣ Data Exploration & Cleaning (`notebooks/data_exploration.ipynb`)
- Inspected schema, class distribution, and duplicates
- Verified disease alignment between datasets
- Standardized column names and values
- Confirmed:
  - 4,920 total records
  - 41 matching disease classes



### 2️⃣ Centralized Configuration (`src/config.py`)
- All paths and artifact locations defined in one place
- Improves:
  - reproducibility,
  - maintainability,
  - separation of concerns



### 3️⃣ Preprocessing Pipeline (`src/preprocess.py`)
Key steps:
- Merged multiple symptom columns into a single text field  
  Example:
  ```bash
  symptoms: itching, skin_rash, nodal_skin_eruptions
  ```

- Created stable **label encodings** (`label_encoder.json`)
- Created disease → precautions mapping (`precaution_map.json`)
- Ensured training–inference consistency via saved artifacts



### 4️⃣ Data Splitting & Leakage Prevention (`src/split.py`)
Two strategies implemented:
- **Row-level stratified split** (baseline)
- **Group-aware split (used for final models)**

Group-aware splitting ensures:
- identical symptom patterns never appear across train/val/test,
- realistic generalization evaluation,
- no data leakage.



## 🧠 Models Implemented

### 🔹 Model 1: BiLSTM (Baseline)

A custom **Bidirectional LSTM** model implemented from scratch in PyTorch.

**Why BiLSTM?**
- Strong baseline for sequential text data
- Efficient and lightweight
- Good at learning fixed symptom patterns

Key features:
- Token embedding layer
- Bidirectional LSTM with packed sequences
- Dropout regularization
- Gradient clipping
- Early stopping and checkpointing

Artifacts:
- `artifacts/bilstm.pt`
- `artifacts/vocab.json`
- `artifacts/bilstm_history.json`



### 🔹 Model 2: Transformer (DistilBERT)

A fine-tuned **Transformer-based model** using DistilBERT.

**Why Transformer?**
- Pretrained on large-scale language corpora
- Better semantic understanding of text
- More robust to phrasing variation and unseen combinations

Although both models achieve high accuracy, the Transformer:
- provides **stronger semantic generalization**,
- is better suited for real-world free-text inputs,
- is more resilient to noisy or reordered symptom descriptions.



## 📊 Model Comparison Results

| Metric        | BiLSTM | Transformer |
|--------------|--------|-------------|
| Test Accuracy | 0.9945 | 0.9945 |
| Macro F1      | 0.9941 | 0.9707 |
| Micro F1      | 0.9945 | 0.9945 |

**Interpretation (Non-Technical):**
- Both models are very accurate overall.
- The Transformer struggles slightly on very rare diseases (few examples),
which lowers its Macro-F1 score.
- Despite this, the Transformer is considered **more robust for real deployments**
because it understands language context better than sequence-only models.



## 📈 Visualization & Analysis

The file `src/visualize_results.py` generates graphs for:
- training loss and accuracy curves,
- confusion matrices,
- per-disease performance,
- class imbalance analysis,
- common misclassifications.

All plots are saved to the `images/` directory and support deeper model analysis.



## 🚀 Running the Project Locally

### 1️⃣ Create environment & install dependencies (using `uv`)

```bash
uv sync
```

### 2️⃣ Run the FastAPI service
```bash
uvicorn src.serve:app --reload
```

### 3️⃣ Open API documentation
```bash
http://127.0.0.1:8000/docs
```

## 🐳 Running with Docker

### Build the Image

```bash
docker build -t triageml .
```

### Run the container

```bash
sudo docker run -it --rm  -p 9696:9696 triageml:latest
```

### Access the API

```bash
http://localhost:9696/docs
```

## 🔌 API Usage Example (Swagger & cURL)

The TriageML system exposes a RESTful API built with **FastAPI**.  
Once the service is running, interactive API documentation is available via **Swagger UI**: http://localhost:9696/docs

### Example: Disease Prediction Request

The following example demonstrates how to send a symptom-based request to the API using `curl`.

#### Request
```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
    "symptoms": "itching, skin_rash, nodal_skin_eruptions",
    "model": "transformer",
    "top_k": 5
  }'
```

#### Response
```bash
{
  "model_used": "transformer",
  "symptom_text": "symptoms: itching, skin_rash, nodal_skin_eruptions",
  "predicted_disease": "fungal_infection",
  "predicted_probability": 0.6312073469161987,
  "top_k": [
    {
      "disease": "fungal_infection",
      "probability": 0.6312073469161987
    },
    {
      "disease": "acne",
      "probability": 0.039966657757759094
    },
    {
      "disease": "gastroenteritis",
      "probability": 0.018220238387584686
    },
    {
      "disease": "psoriasis",
      "probability": 0.01699964702129364
    },
    {
      "disease": "hepatitis_e",
      "probability": 0.016799531877040863
    }
  ],
  "precautions": [
    "bath_twice",
    "use_detol_or_neem_in_bathing_water",
    "keep_infected_area_dry",
    "use_clean_cloths"
  ],
  "confidence": "high",
  "disclaimer": "Educational use only. Not for clinical diagnosis. Seek a clinician for medical advice."
}
```


## 🧱 Project Structure

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
│   ├── vocab.json
│   ├── transformer_tokenizer.json
│   └── transformer_training_history.json
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

## ☁️ Deployment & Cloud Considerations

### Why AWS EKS & Lambda were not used

Due to limited AWS credits, full deployment using:

- Amazon EKS (Kubernetes) and

- AWS Lambda

was deferred to avoid unexpected costs.

### Planned Improvements

Given sufficient cloud resources, future work would include:

- Deploying the Docker image to EKS for scalable orchestration

- Creating a Lambda-based lightweight inference endpoint

Adding CI/CD for automated model deployment


