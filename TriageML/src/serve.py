#!/usr/bin/env python3

"""
FastAPI app for TriageML.
"""
from __future__ import annotations

from typing import Literal, List, Dict, Any

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.inference import (
    load_assets,
    build_symptom_text_from_user,
    BiLSTMPredictor,
    TransformerPredictor,
)

app = FastAPI(title="TriageML API", version="1.0.0")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

assets = load_assets()
bilstm = BiLSTMPredictor(device=device)
transformer = TransformerPredictor(device=device)


class PredictRequest(BaseModel):
    symptoms: str = Field(..., description="User-reported symptoms as comma-separated text or short free text.")
    model: Literal["bilstm", "transformer"] = Field("bilstm", description="Which model to use.")
    top_k: int = Field(5, ge=1, le=10, description="Number of top predictions to return.")


class Prediction(BaseModel):
    disease: str
    probability: float


class PredictResponse(BaseModel):
    model_used: str
    symptom_text: str
    predicted_disease: str
    predicted_probability: float
    top_k: List[Prediction]
    precautions: List[str]
    confidence: str
    disclaimer: str


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "device": str(device),
        "num_classes": len(assets.id_to_label),
    }

def confidence_band(p: float) -> str:
    """
    Converts probability into a human-friendly confidence label.

    - End users interpret labels better than raw probabilities.
    - Helps prevent over-trust in low-confidence predictions.
    """
    if p >= 0.60:
        return "high"
    if p >= 0.30:
        return "medium"
    return "low"


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    symptom_text = build_symptom_text_from_user(req.symptoms)

    predictor = bilstm if req.model == "bilstm" else transformer
    topk = predictor.predict_topk(symptom_text, k=req.top_k)

    decoded = [
        Prediction(disease=assets.id_to_label[class_id], probability=prob)
        for class_id, prob in topk
    ]

    predicted = decoded[0].disease
    conf = confidence_band(decoded[0].probability)
    precautions = assets.precautions.get(predicted, [])

    return PredictResponse(
        model_used=req.model,
        symptom_text=symptom_text,
        predicted_disease=predicted,
        predicted_probability=decoded[0].probability,
        confidence=conf,
        top_k=decoded,
        precautions=precautions,
        disclaimer="Educational use only. Not for clinical diagnosis. Seek a clinician for medical advice.",
    )
