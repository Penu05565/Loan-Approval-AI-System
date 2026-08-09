"""
data_pipeline
=============
Reusable Pipe-and-Filter data-preparation package for the
Loan Approval Risk Prediction System (GR4ML Data Preparation View,
Objective 2 Architectural Pattern implementation).

Public API:
    from data_pipeline import LoanDataPreparationPipeline, LoanApplication, schema
"""

import pandas as pd

from . import schema
from .base import PipelineStage
from .pipeline import LoanDataPreparationPipeline
from .schema import LoanApplication


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """A lightweight preprocessing helper for tests and quick validation."""
    return df.copy()


__all__ = [
    "LoanDataPreparationPipeline",
    "LoanApplication",
    "PipelineStage",
    "schema",
    "preprocess_data",
]
