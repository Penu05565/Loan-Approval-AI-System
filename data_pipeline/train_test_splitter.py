"""
train_test_splitter.py
========================
"Train-Test Split", which branches the single
incoming pipe into two outputs — Training Dataset and Testing Dataset.

Kept as its own small class (rather than forcing it into the single-input/
single-output PipelineStage interface every other filter uses) because a
split is inherently a 1-to-2 fan-out, matching the branch drawn in the
diagram. Stratifies on the target column so the class balance seen in
EDA (~70% approved / 30% not approved) is preserved in both splits.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from . import schema
from .base import logger


@dataclass
class SplitResult:
    train_df: pd.DataFrame
    test_df: pd.DataFrame

    def summary(self) -> dict:
        return {
            "train_rows": len(self.train_df),
            "test_rows": len(self.test_df),
            "train_positive_rate": round(
                float(self.train_df[schema.TARGET_COLUMN].mean()), 3
            ),
            "test_positive_rate": round(
                float(self.test_df[schema.TARGET_COLUMN].mean()), 3
            ),
        }


class TrainTestSplitter:
    """Splits the fully-prepared dataset into stratified train/test sets."""

    name = "TrainTestSplitter"

    def __init__(self, test_size: float = 0.2, random_state: int = 42) -> None:
        self.test_size = test_size
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> SplitResult:
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[schema.TARGET_COLUMN],
        )
        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        result = SplitResult(train_df=train_df, test_df=test_df)
        logger.info("%s: %s", self.name, result.summary())
        return result
