"""
pipeline.py
============
Orchestrator that wires the individual filters into the exact Pipe-and-Filter
chain

    Raw CSV -> DataCollector -> MissingValueHandler -> DuplicateRemover ->
    CategoricalEncoder -> FeatureScaler -> FeatureSelector -> TrainTestSplitter
    -> (Training Dataset, Testing Dataset)

Each arrow is a "pipe": the DataFrame produced by one filter is passed
untouched as the input to the next. Because every filter shares the
PipelineStage interface, stages can be reordered, swapped, or reused
individually by other modules (e.g. the online Feature Assembly Service only
needs MissingValueHandler + CategoricalEncoder + FeatureScaler +
FeatureSelector — loaded from disk via `PipelineStage.load()` — without
DataCollector or TrainTestSplitter at all).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os

import pandas as pd

from .base import logger
from .data_collection import DataCollector
from .missing_value_handler import MissingValueHandler
from .duplicate_remover import DuplicateRemover
from .categorical_encoder import CategoricalEncoder
from .feature_scaler import FeatureScaler
from .feature_selector import FeatureSelector
from .train_test_splitter import TrainTestSplitter, SplitResult


@dataclass
class LoanDataPreparationPipeline:
    """Pipe-and-Filter orchestrator for the loan-approval data model."""

    source_path: str
    top_k_features: int = 12
    test_size: float = 0.2
    random_state: int = 42

    missing_value_handler: MissingValueHandler = field(init=False)
    duplicate_remover: DuplicateRemover = field(init=False)
    categorical_encoder: CategoricalEncoder = field(init=False)
    feature_scaler: FeatureScaler = field(init=False)
    feature_selector: FeatureSelector = field(init=False)
    splitter: TrainTestSplitter = field(init=False)

    def __post_init__(self) -> None:
        self.collector = DataCollector(self.source_path)
        self.missing_value_handler = MissingValueHandler()
        self.duplicate_remover = DuplicateRemover()
        self.categorical_encoder = CategoricalEncoder()
        self.feature_scaler = FeatureScaler()
        self.feature_selector = FeatureSelector(k=self.top_k_features)
        self.splitter = TrainTestSplitter(
            test_size=self.test_size, random_state=self.random_state
        )

    def feature_engineering(self, df):
        df = df.copy()
        df["total_income"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
        df["income_loan_ratio"] = df["total_income"] / (df["LoanAmount"].fillna(df["LoanAmount"].median()) + 1)
        df["loan_per_income"] = df["LoanAmount"].fillna(df["LoanAmount"].median()) / (df["total_income"] + 1)
        df["is_high_loan"] = (df["LoanAmount"] > df["LoanAmount"].median()).astype(int)
        return df
    
    def run(self) -> SplitResult:
        """Executes every filter in sequence and returns the train/test split."""
        logger.info("=== Loan Data Preparation Pipeline: START ===")

        df = self.collector.fit_transform()
        df = self.missing_value_handler.fit_transform(df)
        df = self.feature_engineering(df)
        df = self.duplicate_remover.fit_transform(df)
        df = self.categorical_encoder.fit_transform(df)
        df = self.feature_scaler.fit_transform(df)
        df = self.feature_selector.fit_transform(df)

        result = self.splitter.split(df)
        logger.info("=== Loan Data Preparation Pipeline: COMPLETE ===")
        return result

    def transform_single(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Reuse the already-fitted filters (minus collection/splitting) to
        prepare a single new record for inference — this is the method the
        online Risk Scoring Service would call.
        """
        df = self.missing_value_handler.transform(df)
        df = self.feature_engineering(df)
        df = self.categorical_encoder.transform(df)
        df = self.feature_scaler.transform(df)
        df = self.feature_selector.transform(df)
        return df

    def save_artifacts(self, output_dir: str) -> None:
        """Persist every fitted filter so other modules/services can reuse them."""
        os.makedirs(output_dir, exist_ok=True)
        self.missing_value_handler.save(os.path.join(output_dir, "missing_value_handler.joblib"))
        self.duplicate_remover.save(os.path.join(output_dir, "duplicate_remover.joblib"))
        self.categorical_encoder.save(os.path.join(output_dir, "categorical_encoder.joblib"))
        self.feature_scaler.save(os.path.join(output_dir, "feature_scaler.joblib"))
        self.feature_selector.save(os.path.join(output_dir, "feature_selector.joblib"))
        logger.info("Pipeline: fitted filter artifacts saved to %s", output_dir)
