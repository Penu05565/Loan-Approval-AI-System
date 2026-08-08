"""
run_pipeline.py
================
Example usage of the `data_pipeline` package — this is what "another module"
(e.g. the Airflow DAG's PythonOperator, or a notebook, or a test) would call.

Usage:
    python run_pipeline.py
"""

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

    pipeline.save_artifacts(os.path.join(output_dir, "artifacts"))

    print("\n--- Pipeline summary ---")
    print("Selected features:", pipeline.feature_selector.selected_columns)
    print(result.summary())
    print(f"Training dataset written to: {train_path}")
    print(f"Testing dataset written to:  {test_path}")

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