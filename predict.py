"""
predict.py
==========
Online prediction path for the Risk Scoring Service. This is the module the
Flask/FastAPI prediction endpoint (see System Architecture, "ML Prediction
Service") calls for every incoming loan application.

It reuses the *already-fitted* data preparation filters produced by
run_pipeline.py (saved under output/artifacts/) plus the trained model
(saved under models/ by train_model.py). It never re-fits anything --
re-fitting at request time would mean every prediction used a different
scaler/encoder than the one the model was trained on, silently corrupting
every result.

Stage order mirrors LoanDataPreparationPipeline.transform_single() exactly:
    missing_value_handler -> categorical_encoder -> feature_scaler ->
    feature_selector
(DataCollector, DuplicateRemover, and TrainTestSplitter are training-only
stages and are intentionally skipped here, same as in the pipeline module.)

Usage:
    python predict.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import joblib
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_pipeline.base import PipelineStage  # noqa: E402
from data_pipeline import LoanApplication  # noqa: E402

BASE_DIR = os.path.dirname(__file__)
ARTIFACTS_DIR = os.path.join(BASE_DIR, "output", "artifacts")
MODELS_DIR = os.path.join(BASE_DIR, "models")


@dataclass
class PredictionResult:
    loan_id: str
    approved: bool
    risk_probability: float
    confidence: float

    def to_dict(self) -> dict:
        return {
            "loan_id": self.loan_id,
            "approved": self.approved,
            "risk_probability": round(self.risk_probability, 4),
            "confidence": round(self.confidence, 4),
        }


class LoanPredictionService:
    """
    Loads the fitted data preparation filters and the trained classifier
    once at startup, then serves predictions for individual applications.
    Intended to be instantiated a single time (e.g. as a module-level
    singleton in the Flask app) rather than per-request, since loading
    joblib artifacts repeatedly would add unnecessary latency.
    """

    def __init__(
        self,
        artifacts_dir: str = ARTIFACTS_DIR,
        models_dir: str = MODELS_DIR,
    ) -> None:
        # --- Fitted data preparation filters (order matters, see module docstring) ---
        self.missing_value_handler = PipelineStage.load(
            os.path.join(artifacts_dir, "missing_value_handler.joblib")
        )
        self.categorical_encoder = PipelineStage.load(
            os.path.join(artifacts_dir, "categorical_encoder.joblib")
        )
        self.feature_scaler = PipelineStage.load(
            os.path.join(artifacts_dir, "feature_scaler.joblib")
        )
        self.feature_selector = PipelineStage.load(
            os.path.join(artifacts_dir, "feature_selector.joblib")
        )

        # --- Trained model + the exact column order it was trained on ---
        self.model = joblib.load(os.path.join(models_dir, "loan_model.pkl"))
        self.feature_columns = joblib.load(
            os.path.join(models_dir, "feature_columns.pkl")
        )

    def _feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the same engineered features used during training."""
        df = df.copy()
        df["total_income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["income_loan_ratio"] = df["total_income"] / (
            df["LoanAmount"].fillna(df["LoanAmount"].median()) + 1
        )
        df["loan_per_income"] = df["LoanAmount"].fillna(df["LoanAmount"].median()) / (
            df["total_income"] + 1
        )
        df["is_high_loan"] = (df["LoanAmount"] > df["LoanAmount"].median()).astype(int)
        return df

    def _prepare(self, raw_row: pd.DataFrame) -> pd.DataFrame:
        """Runs a single raw application row through the fitted filter chain."""
        df = self.missing_value_handler.transform(raw_row)
        df = self._feature_engineering(df)
        df = self.categorical_encoder.transform(df)
        df = self.feature_scaler.transform(df)
        df = self.feature_selector.transform(df)

        # Defensive check: if the fitted feature_selector's output doesn't
        # exactly match what the model was trained on, fail loudly here
        # rather than let sklearn silently reorder/misalign columns.
        missing = set(self.feature_columns) - set(df.columns)
        extra = set(df.columns) - set(self.feature_columns)
        if missing:
            raise ValueError(
                f"Prepared data is missing columns the model expects: {missing}. "
                "The data pipeline artifacts and the trained model appear to "
                "be out of sync -- retrain or re-run the pipeline."
            )
        if extra:
            df = df.drop(columns=list(extra))

        return df[self.feature_columns]

    def predict(self, application: LoanApplication) -> PredictionResult:
        raw_row = application.to_frame_row()
        prepared = self._prepare(raw_row)

        prediction = self.model.predict(prepared)[0]
        probabilities = self.model.predict_proba(prepared)[0]
        prob_approved = float(probabilities[1])
        risk_probability = float(1-probabilities[1])  # P(Loan_Status == 1)
        approved = bool(prediction == 1)
        confidence = prob_approved if approved else (1-prob_approved)

        return PredictionResult(
            loan_id=application.loan_id,
            approved=approved,
            risk_probability=risk_probability,
            confidence=confidence,
        )

    def predict_from_dict(self, payload: dict) -> PredictionResult:
        """Convenience entry point for a Flask route: request.json -> result."""
        application = LoanApplication(**payload)
        return self.predict(application)


def main() -> None:
    service = LoanPredictionService()

    sample_application = LoanApplication(
        loan_id="LP999999",
        gender="Female",
        married="Yes",
        dependents="1",
        education="Graduate",
        self_employed="No",
        applicant_income=5200,
        coapplicant_income=1500,
        loan_amount=110,
        loan_amount_term=360,
        credit_history=1.0,
        property_area="Urban",
    )

    result = service.predict(sample_application)
    print("\n--- Prediction ---")
    print(result.to_dict())


if __name__ == "__main__":
    main()