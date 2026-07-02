# Loan Approval Risk Prediction

## Data Preparation Pipeline

Implements the Pipe-and-Filter data-model pipeline:

Raw CSV -> Data Collection -> Missing Value Handling -> Duplicate Removal ->
Categorical Encoding -> Feature Scaling -> Feature Selection ->
Train-Test Split -> Training / Testing Dataset

## Dataset
`data/loan_train.csv` — Dream Housing Finance loan-approval dataset (491 records),
sourced from:
https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv
Columns: Loan_ID, Gender, Married, Dependents, Education, Self_Employed,
ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term,
Credit_History, Property_Area, Loan_Status (target: 1=approved, 0=not approved).

## Structure
```
src/data_pipeline/
    schema.py                # Data Model: column contract + LoanApplication entity
    base.py                  # PipelineStage ABC (fit_transform/transform/save/load)
    data_collection.py       # Filter 1
    missing_value_handler.py # Filter 2
    duplicate_remover.py     # Filter 3
    categorical_encoder.py   # Filter 4
    feature_scaler.py        # Filter 5
    feature_selector.py      # Filter 6
    train_test_splitter.py   # Filter 7 (fans out into train/test)
    pipeline.py               # Orchestrator wiring all filters together
run_pipeline.py               # Example end-to-end usage
train_model.py                 # Trains the classifier on the pipeline's output
data/loan_train.csv
output/                       # train_dataset.csv, test_dataset.csv, fitted artifacts/
models/                        # loan_model.pkl, feature_columns.pkl, metrics.json
```

## Setup

Create and activate a virtual environment, then install dependencies.

**macOS / Linux**
```
python3 -m venv venv
source venv/bin/activate
```

**Windows**
```
python -m venv venv
venv\Scripts\activate
```

**Install requirements**
```
pip install -r requirements.txt
```

If `requirements.txt` isn't set up yet, install directly:
```
pip install pandas scikit-learn joblib
```

To leave the environment when you're done: `deactivate`

## Run

**1. Data preparation** — cleans, encodes, scales, and splits the raw dataset:
```
python run_pipeline.py
```
This reads `data/loan_train.csv` and writes `output/train_dataset.csv`,
`output/test_dataset.csv`, and the fitted encoder/scaler artifacts to
`output/artifacts/`.

**2. Model training** — trains the classifier on the pipeline's output:
```
python train_model.py
```
This reads `output/train_dataset.csv` and `output/test_dataset.csv`, trains a
Gradient Boosting classifier (the Algorithm selected in the Analytics Design
View), evaluates it against the report's success criteria (accuracy,
precision, recall, F1), and writes:
- `models/loan_model.pkl` — the trained classifier
- `models/feature_columns.pkl` — the exact feature column order used in
  training, so downstream services align incoming data the same way
- `models/metrics.json` — evaluation metrics for the report and for the
  Model Monitoring component

Run the two scripts in order — training depends on the pipeline's output
existing first.

## Next step
`predict.py` (not yet built) will load `loan_model.pkl` and
`feature_columns.pkl`, chain incoming requests through the same fitted
pipeline stages in `output/artifacts/`, and return a prediction — this is
the piece the Flask/FastAPI prediction service will call.