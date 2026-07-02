"""
data_pipeline
=============
Reusable Pipe-and-Filter data-preparation package for the
Loan Approval Risk Prediction System (GR4ML Data Preparation View,
Objective 2 Architectural Pattern implementation).

Public API:
    from data_pipeline import LoanDataPreparationPipeline, LoanApplication, schema
"""

from . import schema
from .schema import LoanApplication
from .pipeline import LoanDataPreparationPipeline
from .base import PipelineStage

__all__ = [
    "LoanDataPreparationPipeline",
    "LoanApplication",
    "PipelineStage",
    "schema",
]
