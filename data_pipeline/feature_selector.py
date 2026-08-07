"""
feature_selector.py
=====================
Filter 6 in the diagram: "Feature Selection".

Applies a two-step selection on top of the (imputed, encoded, scaled)
feature matrix:
  1. VarianceThreshold — drops near-constant columns (e.g. a rarely-hit
     one-hot category that survived encoding with almost no variance).
  2. SelectKBest(f_classif) — ranks remaining features by their ANOVA
     F-statistic against the target and keeps the top `k`.

The learned column list is stored on fit_transform() and re-applied on
transform(), so the online Risk Scoring Service always receives the exact
same feature vector shape the model was trained on.
"""

from __future__ import annotations

import pandas as pd
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif

from . import schema
from .base import PipelineStage, logger


class FeatureSelector(PipelineStage):
    """Selects the top-k most predictive features for the target."""

    name = "FeatureSelector"

    def __init__(self, k: int = 999, variance_threshold: float = 0.01) -> None:
        super().__init__()
        self.k = k
        self._variance_filter = VarianceThreshold(threshold=variance_threshold)
        self._kbest: SelectKBest | None = None
        self._selected_columns: list[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        feature_cols = [
            c for c in df.columns if c not in (schema.ID_COLUMN, schema.TARGET_COLUMN)
        ]
        X = df[feature_cols]
        y = df[schema.TARGET_COLUMN]

        X_var = self._variance_filter.fit_transform(X)
        surviving_after_variance = list(X.columns[self._variance_filter.get_support()])

        k = min(self.k, X_var.shape[1])
        self._kbest = SelectKBest(score_func=f_classif, k=k)
        self._kbest.fit(X_var, y)
        self._selected_columns = [
            col
            for col, keep in zip(surviving_after_variance, self._kbest.get_support())
            if keep
        ]

        self._is_fitted = True
        logger.info("%s: selected %d features -> %s", self.name, len(self._selected_columns), self._selected_columns)
        return self._assemble(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._require_fitted()
        return self._assemble(df)

    def _assemble(self, df: pd.DataFrame) -> pd.DataFrame:
        keep = [schema.ID_COLUMN] + self._selected_columns
        if schema.TARGET_COLUMN in df.columns:
            keep = keep + [schema.TARGET_COLUMN]
        result = df[keep].copy()
        self.log_shape(result, "selected top features")
        return result

    @property
    def selected_columns(self) -> list[str]:
        return list(self._selected_columns)
