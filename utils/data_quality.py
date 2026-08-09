"""
utils/data_quality.py
-----------------------
Lightweight data-quality checks: missing-value reporting and a simple
drift signal comparing training data to recent production traffic.
"""

import pandas as pd


def missing_value_report(df: pd.DataFrame) -> pd.Series:
    """Percentage of nulls per column."""
    return (df.isnull().mean() * 100).round(2)


def simple_drift_report(
    train_df: pd.DataFrame,
    recent_df: pd.DataFrame,
    numeric_cols: list[str],
) -> dict:
    """
    Very small drift signal: relative change in mean for numeric features
    between the training distribution and a recent batch of data.
    Flag any column where the change exceeds ~20% as worth investigating.
    """
    report = {}
    for col in numeric_cols:
        train_mean = train_df[col].mean()
        recent_mean = recent_df[col].mean()
        pct_change = abs(recent_mean - train_mean) / (abs(train_mean) + 1e-9) * 100
        report[col] = round(pct_change, 2)
    return report


if __name__ == "__main__":
    train_df = pd.read_csv("output/train_dataset.csv")
    print("Missing value report (train):")
    print(missing_value_report(train_df))
