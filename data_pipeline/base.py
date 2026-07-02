"""
base.py
=======
Defines the common "Filter" contract used by every stage in the pipeline
(Pipe-and-Filter architectural pattern).

Each concrete filter:
  * is independent and self-contained (imports only schema.py + this base)
  * exposes fit_transform(df) for the offline/training path
  * exposes transform(df) for the online/serving path, reusing whatever the
    filter learned during fit_transform (e.g. imputation medians, encoders,
    scalers) so training and serving stay consistent
  * can be persisted with save()/load() so a fitted filter produced by the
    offline pipeline can be reused by other modules (e.g. the online Feature
    Assembly Service in the system architecture).

"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import joblib
import pandas as pd

logger = logging.getLogger("data_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class PipelineStage(ABC):
    """Base 'Filter' in the Pipe-and-Filter pattern."""

    name: str = "PipelineStage"

    def __init__(self) -> None:
        self._is_fitted: bool = False

    @abstractmethod
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Learn any parameters from df (offline) and return the transformed df."""
        raise NotImplementedError

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply previously-learned parameters to new data (online/serving)."""
        raise NotImplementedError

    def _require_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name} has not been fitted yet. Call fit_transform() "
                f"during the offline pipeline run before calling transform()."
            )

    def save(self, path: str) -> None:
        joblib.dump(self, path)
        logger.info("%s: saved fitted filter to %s", self.name, path)

    @staticmethod
    def load(path: str) -> "PipelineStage":
        return joblib.load(path)

    def log_shape(self, df: pd.DataFrame, action: str) -> None:
        logger.info("%s: %s -> shape=%s", self.name, action, df.shape)
