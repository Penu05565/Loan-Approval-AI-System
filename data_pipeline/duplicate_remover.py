"""
duplicate_remover.py
=====================
Filter 3: "Duplicate Removal".

Removes:
  1. Exact duplicate rows.
  2. Duplicate Loan_ID entries (keeps the first occurrence), since Loan_ID is
     the natural key of the Applicant/Application entity in the GR4ML Data
     Preparation View and must be unique in a clean training set.

Stateless (nothing is "learned"), so transform() and fit_transform() are the
same operation — but it still respects the common interface so it can slot
into the same pipeline chain as every other filter.
"""

from __future__ import annotations

import pandas as pd

from . import schema
from .base import PipelineStage


class DuplicateRemover(PipelineStage):
    """Drops exact duplicate rows and duplicate Loan_ID records."""

    name = "DuplicateRemover"

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates()
        df = df.drop_duplicates(subset=[schema.ID_COLUMN], keep="first")
        df = df.reset_index(drop=True)
        self._is_fitted = True
        removed = before - len(df)
        self.log_shape(df, f"removed {removed} duplicate row(s)")
        return df
