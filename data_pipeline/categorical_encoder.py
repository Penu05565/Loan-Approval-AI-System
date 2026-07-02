"""
categorical_encoder.py
========================
Filter 4: "Categorical Encoding".

Uses One-Hot Encoding for the categorical features (Gender, Married,
Dependents, Education, Self_Employed, Property_Area). The fitted
OneHotEncoder (categories learned on the training split) is stored so the
exact same columns, in the exact same order, are produced at serving time —
even if a rare category is absent from a single incoming request.
"""

from __future__ import annotations

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from . import schema
from .base import PipelineStage


class CategoricalEncoder(PipelineStage):
    """One-hot encodes categorical features; leaves other columns untouched."""

    name = "CategoricalEncoder"

    def __init__(self) -> None:
        super().__init__()
        self._encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=False, drop="if_binary"
        )
        self._encoded_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        encoded = self._encoder.fit_transform(df[schema.CATEGORICAL_FEATURES])
        self._encoded_columns = list(
            self._encoder.get_feature_names_out(schema.CATEGORICAL_FEATURES)
        )
        self._is_fitted = True
        result = self._assemble(df, encoded)
        self.log_shape(result, "one-hot encoded categorical features")
        return result

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        encoded = self._encoder.transform(df[schema.CATEGORICAL_FEATURES])
        return self._assemble(df, encoded)

    def _assemble(self, df: pd.DataFrame, encoded) -> pd.DataFrame:
        encoded_df = pd.DataFrame(
            encoded, columns=self._encoded_columns, index=df.index
        )
        remaining = df.drop(columns=schema.CATEGORICAL_FEATURES)
        return pd.concat([remaining, encoded_df], axis=1)

    @property
    def encoded_feature_names(self) -> list[str]:
        return list(self._encoded_columns)
