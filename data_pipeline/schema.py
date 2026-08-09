"""
schema.py
=========
Data Model layer for the Loan Approval Risk Prediction data-preparation pipeline.

This module is the single source of truth for:
  1. The *domain entity* representing one loan application (a typed, validated
     record — this is the "Data Model" other modules and services import
     instead of passing raw dicts / DataFrame rows around).
  2. The *column contract* (which raw columns exist, which are numerical,
     which are categorical, which is the target) that every pipeline stage
     in this package reads from, so column names only ever live in ONE place.

Any other module (the offline training pipeline, the online Feature
Assembly Service, unit tests, notebooks, etc.) should import from here
rather than hard-coding column names — that is what makes the rest of the
package reusable and safe to refactor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# --------------------------------------------------------------------------- #
# Column contract — the ONE place column names/types are declared.
# --------------------------------------------------------------------------- #

ID_COLUMN: str = "Loan_ID"
TARGET_COLUMN: str = "Loan_Status"

NUMERICAL_FEATURES: list[str] = [
    "ApplicantIncome",
    "CoapplicantIncome",
    "LoanAmount",
    "Loan_Amount_Term",
    "Credit_History",
]

CATEGORICAL_FEATURES: list[str] = [
    "Gender",
    "Married",
    "Dependents",
    "Education",
    "Self_Employed",
    "Property_Area",
]

ALL_FEATURES: list[str] = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
RAW_COLUMNS: list[str] = [ID_COLUMN] + ALL_FEATURES + [TARGET_COLUMN]

# Values considered "positive" (approved) after Loan_Status is encoded to int
POSITIVE_CLASS = 1


def raw_dtypes() -> dict[str, str]:
    """Expected pandas dtypes for the raw CSV, used to validate input on load."""
    dtypes = {ID_COLUMN: "object", TARGET_COLUMN: "int64"}
    dtypes.update({c: "object" for c in CATEGORICAL_FEATURES})
    dtypes.update({c: "float64" for c in NUMERICAL_FEATURES})
    return dtypes


# --------------------------------------------------------------------------- #
# Domain entity — typed record for a single loan application.
# --------------------------------------------------------------------------- #


@dataclass
class LoanApplication:
    """
    Typed representation of a single loan application record.

    Used to validate/normalize a single row (e.g. an incoming API request in
    the online serving path) before it is handed to the same preprocessing
    filters used offline — guaranteeing training/serving consistency.
    """

    loan_id: str
    gender: Optional[str]
    married: Optional[str]
    dependents: Optional[str]
    education: str
    self_employed: Optional[str]
    applicant_income: float
    coapplicant_income: float
    loan_amount: Optional[float]
    loan_amount_term: Optional[float]
    credit_history: Optional[float]
    property_area: str
    loan_status: Optional[int] = field(default=None)  # None at inference time

    def __post_init__(self) -> None:
        if self.applicant_income < 0 or self.coapplicant_income < 0:
            raise ValueError(f"{self.loan_id}: income values cannot be negative")
        if self.loan_amount is not None and self.loan_amount < 0:
            raise ValueError(f"{self.loan_id}: loan_amount cannot be negative")
        if self.credit_history is not None and self.credit_history not in (0.0, 1.0):
            raise ValueError(f"{self.loan_id}: credit_history must be 0.0 or 1.0")

    @classmethod
    def from_series(cls, row: pd.Series) -> "LoanApplication":
        """Build a LoanApplication from a raw pandas row (raw CSV column names)."""
        return cls(
            loan_id=row.get(ID_COLUMN),
            gender=row.get("Gender"),
            married=row.get("Married"),
            dependents=row.get("Dependents"),
            education=row.get("Education"),
            self_employed=row.get("Self_Employed"),
            applicant_income=float(row.get("ApplicantIncome", 0) or 0),
            coapplicant_income=float(row.get("CoapplicantIncome", 0) or 0),
            loan_amount=_safe_float(row.get("LoanAmount")),
            loan_amount_term=_safe_float(row.get("Loan_Amount_Term")),
            credit_history=_safe_float(row.get("Credit_History")),
            property_area=row.get("Property_Area"),
            loan_status=_safe_int(row.get(TARGET_COLUMN)),
        )

    def to_frame_row(self) -> pd.DataFrame:
        """Convert back into a one-row DataFrame with raw column names, so it
        can be pushed through the exact same pipeline stages used offline."""
        data = {
            ID_COLUMN: self.loan_id,
            "Gender": self.gender,
            "Married": self.married,
            "Dependents": self.dependents,
            "Education": self.education,
            "Self_Employed": self.self_employed,
            "ApplicantIncome": self.applicant_income,
            "CoapplicantIncome": self.coapplicant_income,
            "LoanAmount": self.loan_amount,
            "Loan_Amount_Term": self.loan_amount_term,
            "Credit_History": self.credit_history,
            "Property_Area": self.property_area,
        }
        if self.loan_status is not None:
            data[TARGET_COLUMN] = self.loan_status
        return pd.DataFrame([data])


def _safe_float(value) -> Optional[float]:
    try:
        return float(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> Optional[int]:
    try:
        return int(value) if pd.notna(value) else None
    except (TypeError, ValueError):
        return None
