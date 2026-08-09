from pathlib import Path

from data_pipeline import LoanDataPreparationPipeline


def test_pipeline_runs_and_splits():
    root_dir = Path(__file__).resolve().parents[1]
    dataset_path = root_dir / "data" / "loan_dataset.csv"

    pipeline = LoanDataPreparationPipeline(
        source_path=str(dataset_path),
        top_k_features=12,
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    assert result.train_df is not None
    assert result.test_df is not None
    assert len(result.train_df) > 0
    assert len(result.test_df) > 0
    assert set(result.train_df.columns) == set(result.test_df.columns)
    split_fraction = len(result.test_df) / (len(result.train_df) + len(result.test_df))
    assert 0.1 < split_fraction < 0.3
