import os

import joblib
import pandas as pd

from config import DATA_PATH, FEATURE_PATH, MODEL_PATH
from predict import LoanPredictionService


def test_model_artifacts_exist_and_loadable():
    assert os.path.exists(MODEL_PATH), f"Model file missing: {MODEL_PATH}"
    assert os.path.exists(FEATURE_PATH), f"Feature columns missing: {FEATURE_PATH}"

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURE_PATH)

    assert hasattr(model, "predict"), "Loaded model has no predict"
    assert hasattr(model, "predict_proba"), "Loaded model has no predict_proba"
    assert isinstance(features, (list, tuple)), "feature columns should be a list/tuple"


def test_data_quality_on_raw_dataset():
    df = pd.read_csv(DATA_PATH)
    # basic checks
    assert len(df) > 0, "Raw dataset is empty"
    assert "Loan_ID" in df.columns, "Loan_ID missing from raw dataset"
    assert "Loan_Status" in df.columns, "Loan_Status missing from raw dataset"


def test_service_inference_runs():
    service = LoanPredictionService()
    sample = {
        "loan_id": "LPTEST123",
        "gender": "Female",
        "married": "Yes",
        "dependents": "0",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 4000,
        "coapplicant_income": 0,
        "loan_amount": 100,
        "loan_amount_term": 360,
        "credit_history": 1.0,
        "property_area": "Urban",
    }

    result = service.predict_from_dict(sample)
    assert result.loan_id == sample["loan_id"]
    assert 0.0 <= result.risk_probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0
