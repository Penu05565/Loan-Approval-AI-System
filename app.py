"""
app.py
-------
Flask REST API for the AI-Powered Loan Approval Risk Prediction System.

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

from flask import Flask, jsonify, render_template, request

from predict import LoanPredictionService


# --------------------------------------------------------------------
# Flask App
# --------------------------------------------------------------------

app = Flask(__name__)

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
# Home Page
# --------------------------------------------------------------------


@app.route("/")
def home():
    return render_template("index.html")


# --------------------------------------------------------------------
# Health Check
# --------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "healthy",
            "service": "Loan Approval Prediction API",
            "model_loaded": True,
            "timestamp": datetime.now().isoformat(),
        }
    )


# --------------------------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------------------------


@app.route("/predict", methods=["POST"])
def predict():

    try:

        payload = request.get_json()

        if payload is None:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": "Request body must contain JSON."
                    }
                ),
                400,
            )

        required_fields = [
            "loan_id",
            "gender",
            "married",
            "dependents",
            "education",
            "self_employed",
            "applicant_income",
            "coapplicant_income",
            "loan_amount",
            "loan_amount_term",
            "credit_history",
            "property_area",
        ]

        missing = [
            field for field in required_fields
            if field not in payload
        ]

        if missing:
            return (
                jsonify(
                    {
                        "success": False,
                        "missing_fields": missing,
                    }
                ),
                400,
            )

        result = service.predict_from_dict(payload)

        logging.info(
            "LoanID=%s | Approved=%s | Probability=%.4f | Confidence=%.4f",
            result.loan_id,
            result.approved,
            result.risk_probability,
            result.confidence,
        )

        return jsonify(
            {
                "success": True,
                "prediction": result.to_dict(),
            }
        )

    except ValueError as ex:

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(ex),
                }
            ),
            400,
        )

    except Exception as ex:

        logging.exception("Prediction failed")

        return (
            jsonify(
                {
                    "success": False,
                    "error": str(ex),
                }
            ),
            500,
        )


# --------------------------------------------------------------------
# API Information
# --------------------------------------------------------------------


@app.route("/api", methods=["GET"])
def api():

    return jsonify(
        {
            "name": "Loan Approval Risk Prediction API",
            "version": "1.0",
            "endpoints": {
                "/": "Frontend",
                "/health": "Health Check",
                "/predict": "POST Prediction",
            },
        }
    )


# --------------------------------------------------------------------
# Main
# --------------------------------------------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True,
    )
