"""
run_pipeline.py
================
Example usage of the `data_pipeline` package — this is what "another module"
(e.g. the Airflow DAG's PythonOperator, or a notebook, or a test) would call.

Usage:
    python run_pipeline.py
"""

import json
import os

from data_pipeline import LoanApplication, LoanDataPreparationPipeline


def main() -> None:
    data_path = os.path.join(os.path.dirname(__file__), "data", "loan_dataset.csv")
    output_dir = os.path.join(os.path.dirname(__file__), "output")
    os.makedirs(output_dir, exist_ok=True)

    pipeline = LoanDataPreparationPipeline(
        source_path=data_path,
        top_k_features=12,
        test_size=0.2,
        random_state=42,
    )

    result = pipeline.run()

    train_path = os.path.join(output_dir, "train_dataset.csv")
    test_path = os.path.join(output_dir, "test_dataset.csv")
    result.train_df.to_csv(train_path, index=False)
    result.test_df.to_csv(test_path, index=False)

    artifacts_dir = os.path.join(output_dir, "artifacts")
    pipeline.save_artifacts(artifacts_dir)

    # --------------------------------------------------------------
    # Persist the training-time LoanAmount median.
    #
    # predict.py's feature-engineering step (income_loan_ratio,
    # loan_per_income, is_high_loan) needs the SAME LoanAmount median
    # that was used during training. Recomputing it from a single
    # incoming request row at inference time is wrong (the "median"
    # of one row is just that row's own value) and causes train/serve
    # skew. Saving it here, once, as a fitted artifact fixes that.
    #
    # NOTE: if `LoanDataPreparationPipeline` already exposes a fitted
    # missing_value_handler (the way it exposes `feature_selector`
    # below), use that to transform the raw training data first, so
    # this median is computed on the exact same missing-value-handled
    # data predict.py's feature-engineering step will see:
    #
    #   raw_df = pd.read_csv(data_path)
    #   handled_df = pipeline.missing_value_handler.transform(raw_df)
    #   loan_amount_median = float(handled_df["LoanAmount"].median())
    #
    # If that attribute isn't exposed, computing it from the raw
    # source CSV (as below) is a reasonable approximation, since
    # missing-value handling typically doesn't shift the median much.
    # --------------------------------------------------------------
    import pandas as pd

    raw_df = pd.read_csv(data_path)
    loan_amount_median = float(raw_df["LoanAmount"].median())

    stats_path = os.path.join(artifacts_dir, "feature_engineering_stats.json")
    with open(stats_path, "w") as f:
        json.dump({"loan_amount_median": loan_amount_median}, f, indent=2)

    print("\n--- Pipeline summary ---")
    print("Selected features:", pipeline.feature_selector.selected_columns)
    print(result.summary())
    print(f"Training dataset written to: {train_path}")
    print(f"Testing dataset written to: {test_path}")
    print(f"Feature engineering stats written to: {stats_path}")
    print(f"  loan_amount_median = {loan_amount_median}")

    # --- Demonstrates reuse for a single new application (serving path) ---
    print("\n--- Single-record inference-path demo ---")
    new_application = LoanApplication(
        loan_id="LP999999",
        gender="Female",
        married="Yes",
        dependents="1",
        education="Graduate",
        self_employed="No",
        applicant_income=5200,
        coapplicant_income=1500,
        loan_amount=110,
        loan_amount_term=360,
        credit_history=1.0,
        property_area="Urban",
    )
    single_row_df = new_application.to_frame_row()
    prepared = pipeline.transform_single(single_row_df)
    print("Prepared feature vector for scoring service:")
    print(prepared.to_dict(orient="records")[0])


if __name__ == "__main__":
    main()