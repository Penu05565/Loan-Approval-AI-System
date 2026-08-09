import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "data", "loan_dataset.csv")

MODEL_PATH = os.path.join(BASE_DIR, "models", "loan_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "metrics.json")

LOG_PATH = os.path.join(BASE_DIR, "logs", "pipeline.log")
