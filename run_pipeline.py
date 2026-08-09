"""
run_pipeline.py
================
Example usage of the `data_pipeline` package — this is what "another module"
(e.g. the Airflow DAG's PythonOperator, or a notebook, or a test) would call.

Usage:
    python run_pipeline.py
"""

import json
import logging
import os

import pandas as pd

from data_pipeline import LoanApplication, LoanDataPreparationPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    try:
        data_path = os.path.join(os.path.dirname(__file__), "data", "loan_dataset.csv")
        output_dir = os.path.join(os.path.dirname(__file__), "output")
        os.makedirs(output_dir, exist_ok=True)

        pipeline = LoanDataPreparationPipeline(
            source_path=data_path,
            top_k_features=12,
            test_size=0.2,
            random_state=42,
        )

        logger.info("Running data preparation pipeline on %s", data_path)
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
        # used during training. Recomputing it from a single incoming
        # request row at inference time is wrong (the "median" of one row
        # is just that row's own value) and causes train/serve skew.
        # Saving it here, once, as a fitted artifact fixes that.
        # --------------------------------------------------------------
        raw_df = pd.read_csv(data_path)
        loan_amount_median = float(raw_df["LoanAmount"].median())

        stats_path = os.path.join(artifacts_dir, "feature_engineering_stats.json")
        with open(stats_path, "w") as f:
            json.dump({"loan_amount_median": loan_amount_median}, f, indent=2)

        logger.info("Selected features: %s", pipeline.feature_selector.selected_columns)
        logger.info("%s", result.summary())
        logger.info("Training dataset written to: %s", train_path)
        logger.info("Testing dataset written to: %s", test_path)
        logger.info("Feature engineering stats written to: %s", stats_path)
        logger.info("loan_amount_median = %s", loan_amount_median)

        if len(pipeline.feature_selector.selected_columns) < 3:
            logger.warning(
                "Only %d features were selected — this looks low; double-check "
                "top_k_features and the feature_selector configuration.",
                len(pipeline.feature_selector.selected_columns),
            )

        # --- Demonstrates reuse for a single new application (serving path) ---
        logger.info("Running single-record inference-path demo")
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
        logger.info(
            "Prepared feature vector for scoring service: %s",
            prepared.to_dict(orient="records")[0],
        )

    except Exception:
        logger.error("Pipeline run failed", exc_info=True)
        raise


if __name__ == "__main__":
    main()
