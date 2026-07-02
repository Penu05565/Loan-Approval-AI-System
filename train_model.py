"""
train_model.py
---------------
Trains the loan approval classification model for the AI-Powered Loan
Approval Risk Prediction System.

Pipeline stage: Model Training -> Model Evaluation -> Trained Model (.pkl)
(see Analytics Design View: Analytics Goal -> Algorithm -> Softgoals)

Input (produced by src/data_pipeline/pipeline.py, see that module's README):
    output/train_dataset.csv
    output/test_dataset.csv

Output:
    models/loan_model.pkl        trained classifier
    models/feature_columns.pkl   ordered list of feature columns used for
                                  training, so the prediction service can
                                  align incoming request data the same way
    models/metrics.json          evaluation metrics for the report /
                                  Model Monitoring component
"""

import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = Path("output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

TARGET_COL = "Loan_Status"
ID_COL = "Loan_ID"

RANDOM_STATE = 42


def load_data():
    train_df = pd.read_csv(DATA_DIR / "train_dataset.csv")
    test_df = pd.read_csv(DATA_DIR / "test_dataset.csv")
    return train_df, test_df


def split_features_target(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in (ID_COL, TARGET_COL)]
    X = df[feature_cols]
    y = df[TARGET_COL]
    return X, y, feature_cols


def train_model(X_train, y_train):
    # Gradient Boosting Classifier: matches the Algorithm element chosen in
    # the Analytics Design View. n_estimators/learning_rate/max_depth are
    # kept moderate to avoid overfitting on a ~390-row training set.
    model = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=2,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_samples": int(len(y_test)),
    }

    print("\n=== Model Evaluation ===")
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))
    print("Confusion matrix:")
    print(pd.DataFrame(
        metrics["confusion_matrix"],
        index=["Actual: Rejected", "Actual: Approved"],
        columns=["Pred: Rejected", "Pred: Approved"],
    ))

    return metrics


def check_success_criteria(metrics: dict):
    # Targets pulled directly from the report's Success Criteria table
    # (Section 3.5 / NFR-01) so the console output doubles as evidence
    # for the assignment writeup.
    targets = {
        "accuracy": 0.90,
        "precision": 0.88,
        "recall": 0.85,
        "f1_score": 0.87,
    }
    print("\n=== Success Criteria Check ===")
    all_met = True
    for key, target in targets.items():
        achieved = metrics[key]
        met = achieved >= target
        all_met = all_met and met
        status = "PASS" if met else "FAIL"
        print(f"{key:10s}: {achieved:.4f}  (target >= {target:.2f})  [{status}]")
    if not all_met:
        print(
            "\nNote: not all targets met. Consider hyperparameter tuning, "
            "trying an alternative algorithm (e.g. RandomForest / "
            "LogisticRegression), or revisiting feature engineering before "
            "finalizing the model for deployment."
        )
    return all_met


def main():
    start = time.time()

    train_df, test_df = load_data()
    X_train, y_train, feature_cols = split_features_target(train_df)
    X_test, y_test, _ = split_features_target(test_df)

    # Guard against column mismatch between train and test sets
    assert list(X_train.columns) == list(X_test.columns), (
        "Train/test feature columns do not match. Check the data "
        "preparation module output."
    )

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    model = train_model(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)
    check_success_criteria(metrics)

    # Persist artifacts
    joblib.dump(model, MODEL_DIR / "loan_model.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_columns.pkl")
    metrics["training_time_seconds"] = round(time.time() - start, 2)
    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved model to {MODEL_DIR / 'loan_model.pkl'}")
    print(f"Saved feature columns to {MODEL_DIR / 'feature_columns.pkl'}")
    print(f"Saved metrics to {MODEL_DIR / 'metrics.json'}")
    print(f"Total time: {metrics['training_time_seconds']}s")


if __name__ == "__main__":
    main()