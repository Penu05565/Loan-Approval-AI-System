"""
data_collection.py
===================
Filter 1: "Data Collection".

Responsible ONLY for getting raw data into a validated pandas DataFrame with
the expected column contract (schema.RAW_COLUMNS). Kept deliberately thin and
swappable: it reads a CSV, but the same interface could pull from a
data warehouse table, an API, or a Kafka topic in the Event-Driven path
without changing any downstream filter.
"""

from __future__ import annotations

import pandas as pd

from . import schema
from .base import PipelineStage, logger


class DataCollector(PipelineStage):
    """Loads and validates raw loan application data from a CSV source."""

    name = "DataCollector"

    def __init__(self, source_path: str) -> None:
        super().__init__()
        self.source_path = source_path

    def fit_transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        raw = self._load()
        self._is_fitted = True
        return raw

    def transform(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        # Collection has no "learned" state — re-reading the source is
        # deterministic, so transform() and fit_transform() behave the same.
        return self.fit_transform(df)

    def _load(self) -> pd.DataFrame:
        df = pd.read_csv(self.source_path)

        # Drop any stray unnamed index columns that sometimes leak into
        # public CSV exports (e.g. "Unnamed: 0").
        unnamed = [c for c in df.columns if c.startswith("Unnamed")]
        if unnamed:
            df = df.drop(columns=unnamed)

        missing_cols = set(schema.RAW_COLUMNS) - set(df.columns)
        if missing_cols:
            raise ValueError(
                f"DataCollector: source is missing expected columns: {missing_cols}"
            )

        df = df[schema.RAW_COLUMNS].copy()
        self.log_shape(df, "collected raw data")
        logger.info(
            "DataCollector: target distribution -> %s",
            df[schema.TARGET_COLUMN].value_counts(normalize=True).round(3).to_dict(),
        )
        return df
