"""
missing_value_handler.py
=========================
Filter 2: "Missing Value Handling".

Numerical columns -> median imputation (robust to the income/loan-amount
outliers present in this dataset).
Categorical columns -> mode (most frequent category) imputation.

Imputation values are LEARNED on fit_transform() (offline/training data)
and stored, then reapplied as-is via transform() on new/serving data — this
prevents "leaking" test-set or live-request statistics back into training.
"""

from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer

from . import schema
from .base import PipelineStage


class MissingValueHandler(PipelineStage):
    """Median-imputes numerical features, mode-imputes categorical features."""

    name = "MissingValueHandler"

    def __init__(self) -> None:
        super().__init__()
        self._numeric_imputer = SimpleImputer(strategy="median")
        self._categorical_imputer = SimpleImputer(strategy="most_frequent")

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[schema.NUMERICAL_FEATURES] = self._numeric_imputer.fit_transform(
            df[schema.NUMERICAL_FEATURES]
        )
        df[schema.CATEGORICAL_FEATURES] = self._categorical_imputer.fit_transform(
            df[schema.CATEGORICAL_FEATURES]
        )
        self._is_fitted = True
        self.log_shape(df, "imputed missing values")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = df.copy()
        df[schema.NUMERICAL_FEATURES] = self._numeric_imputer.transform(
            df[schema.NUMERICAL_FEATURES]
        )
        df[schema.CATEGORICAL_FEATURES] = self._categorical_imputer.transform(
            df[schema.CATEGORICAL_FEATURES]
        )
        return df
