"""
feature_scaler.py
===================
Filter 5: "Feature Scaling".

Standardizes the original numerical features (zero mean, unit variance)
using sklearn's StandardScaler. Scaling is fit only on the numeric columns —
one-hot encoded columns from the previous filter are left as 0/1 indicators,
which is standard practice (scaling binary indicators adds no value and
would only complicate interpretability/SHAP downstream).
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import StandardScaler

from . import schema
from .base import PipelineStage


class FeatureScaler(PipelineStage):
    """Standard-scales numerical features in place."""

    name = "FeatureScaler"

    def __init__(self) -> None:
        super().__init__()
        self._scaler = StandardScaler()

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[schema.NUMERICAL_FEATURES] = self._scaler.fit_transform(
            df[schema.NUMERICAL_FEATURES]
        )
        self._is_fitted = True
        self.log_shape(df, "scaled numerical features")
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        df = df.copy()
        df[schema.NUMERICAL_FEATURES] = self._scaler.transform(
            df[schema.NUMERICAL_FEATURES]
        )
        return df
