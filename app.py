"""
app.py
-------
FastAPI REST API for the AI-Powered Loan Approval Risk Prediction System.

Architecture

Browser / Postman
        │
        ▼
Flask REST API
        │
        ▼
LoanPredictionService (predict.py)
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Gradient Boosting Model

Run:
    python app.py

Open:
    http://127.0.0.1:5000
"""
import logging
import os
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from predict import LoanPredictionService


# --------------------------------------------------------------------
# FastAPI App
# --------------------------------------------------------------------

app = FastAPI(
    title="Loan Approval Prediction API",
    description="Predict loan approval using ML model",
    version="2.0"
)

# --------------------------------------------------------------------
# Load prediction service ONCE
# --------------------------------------------------------------------

service = LoanPredictionService()

# --------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    filename=os.path.join(LOG_DIR, "predictions.log"),
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
)

# --------------------------------------------------------------------
# Request Schema (VERY IMPORTANT)
# --------------------------------------------------------------------

class LoanRequest(BaseModel):
    loan_id: str
    gender: str
    married: str
    dependents: str
    education: str
    self_employed: str
    applicant_income: float
    coapplicant_income: float
    loan_amount: float
    loan_amount_term: float
    credit_history: float
    property_area: str


# --------------------------------------------------------------------
# Home
# --------------------------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "Loan Approval Prediction API is running",
        "docs": "/docs"
    }


# --------------------------------------------------------------------
# Health Check
# --------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "Loan Approval Prediction API",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat(),
    }


# --------------------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------------------

@app.post("/predict")
def predict(request: LoanRequest):

    try:
        payload = request.dict()

        result = service.predict_from_dict(payload)

        logging.info(
            "LoanID=%s | Approved=%s | Probability=%.4f | Confidence=%.4f",
            result.loan_id,
            result.approved,
            result.risk_probability,
            result.confidence,
        )

        return {
            "success": True,
            "prediction": result.to_dict(),
        }

    except ValueError as ex:
        raise HTTPException(status_code=400, detail=str(ex))

    except Exception as ex:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=str(ex))


# --------------------------------------------------------------------
# API Info
# --------------------------------------------------------------------

@app.get("/api")
def api():
    return {
        "name": "Loan Approval Risk Prediction API",
        "version": "2.0",
        "endpoints": {
            "/": "Home",
            "/health": "Health Check",
            "/predict": "POST Prediction",
            "/docs": "Swagger UI",
        },
    }