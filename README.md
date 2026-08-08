# Loan Approval Risk Prediction System

## Overview

This project implements an **AI-Powered Loan Approval Risk Prediction
System** following **Software Engineering for Machine Learning (SE4ML)**
principles. It demonstrates an end-to-end machine learning workflow
including data preparation, model training, online prediction, and
deployment using a FastAPI web application.

### System Workflow

``` text
Raw Dataset
    │
    ▼
Data Preparation Pipeline
    │
    ▼
Train/Test Dataset
    │
    ▼
Random Forest Model Training
    │
    ▼
Model Evaluation
    │
    ▼
Trained Model (.pkl)
    │
    ▼
Prediction Service
    │
    ▼
FastAPI REST API
    │
    ▼
Web User Interface
```

## Data Preparation Pipeline

The preprocessing pipeline follows the **Pipe-and-Filter** architectural
pattern.

``` text
Raw CSV
 │
 ▼
Data Collection
 │
 ▼
Missing Value Handling
 │
 ▼
Duplicate Removal
 │
 ▼
Feature Engineering
 │
 ▼
Categorical Encoding
 │
 ▼
Feature Scaling
 │
 ▼
Feature Selection
 │
 ▼
Train-Test Split
```

## Dataset

**Dataset:** Dream Housing Finance Loan Approval Dataset

Source:

https://raw.githubusercontent.com/dphi-official/Datasets/master/Loan_Data/loan_train.csv

Target:

-   **Loan_Status = 1** → Not Default/ Approved
-   **Loan_Status = 0** → Default/ Not Approved

Features:

-   Loan_ID
-   Gender
-   Married
-   Dependents
-   Education
-   Self_Employed
-   ApplicantIncome
-   CoapplicantIncome
-   LoanAmount
-   Loan_Amount_Term
-   Credit_History
-   Property_Area
-   Loan_Status

## Project Structure

``` text
LoanApprovalProject/
│
├── app.py
├── config.py
├── run_pipeline.py
├── train_model.py
├── predict.py
├── README.md
├── requirements.txt
├── liny-report.txt
├── pytest.ini
├── runtime.txt
├── .render.yaml
│
├── data/
|   ├── loan_dataset.csv
├── models/
|   ├── feature_columns.pkl
|   ├── loan_model.pkl
|   ├── metrics.json
├── output/
|   ├── test_dataset.csv
|   ├── train_dataset.csv
├── logs/
|   ├── pipeline.log
|   ├── predictions.log
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
├── tests/
│   └── conftest.py
│   └── test_pipeline.py
│   └── test_api.py
│   └── test_predict.py
│   └── test_training_inferenece_quality.py
├── venv
└── data_pipeline/
    ├── schema.py                # Data Model: column contract + LoanApplication entity
    ├── base.py                  # PipelineStage ABC (fit_transform/transform/save/load)
    ├── data_collection.py       # Filter 1
    ├── missing_value_handler.py # Filter 2
    ├── duplicate_remover.py     # Filter 3
    ├── categorical_encoder.py   # Filter 4
    ├── feature_scaler.py        # Filter 5
    ├── feature_selector.py      # Filter 6
    ├── train_test_splitter.py   # Filter 7 (fans out into train/test)
    └── pipeline.py              # Orchestrator wiring all filters together
```

## Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib
-   FastAPI
-   HTML5
-   CSS3
-   Bootstrap 5
-   JavaScript
-   Pytest

## Research Notebook

See the lightweight research notebook for dataset exploration and reproducible notes: [notebooks/research.ipynb](notebooks/research.ipynb)

## Linting

A simple lint report placeholder is available at [lint-report.txt](lint-report.txt). To run a local lint check, install `ruff` or `flake8` and run, for example:

```bash
pip install ruff
ruff check .
```

## Setup

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

``` bash
pip install -r requirements.txt
```

### Run tests

``` bash
pytest -q
```

If you prefer, run the same command through Python:

``` bash
python -m pytest -q
```

## Run the Project

### 1. Data Preparation

``` bash
python run_pipeline.py
```

Outputs:

-   output/train_dataset.csv
-   output/test_dataset.csv
-   output/artifacts/

### 2. Model Training

``` bash
python train_model.py
```

Outputs:

-   models/loan_model.pkl
-   models/feature_columns.pkl
-   models/metrics.json

### 3. Launch the Web Application

``` bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Open:

``` text
http://127.0.0.1:8000
```

The interactive API docs are available at `/docs`.

## REST API

### GET /

Returns the home page with the interactive user interface and links to
API documentation.

### GET /health

Example:

``` json
{
  "status": "healthy",
  "service": "Loan Approval Prediction API",
  "model_loaded": true,
  "timestamp": "2026-08-08T00:00:00"
}
```

### POST /predict

Example request:

``` json
{
  "loan_id":"LP999999",
  "gender":"Female",
  "married":"Yes",
  "dependents":"1",
  "education":"Graduate",
  "self_employed":"No",
  "applicant_income":5200,
  "coapplicant_income":1500,
  "loan_amount":110,
  "loan_amount_term":360,
  "credit_history":1,
  "property_area":"Urban"
}
```

Example response:

``` json
{
  "success": true,
  "prediction": {
    "loan_id":"LP999999",
    "approved":true,
    "risk_probability":0.0358,
    "confidence":0.9642
  }
}
```

## Model Artifacts

``` text
models/
    loan_model.pkl
    feature_columns.pkl
    metrics.json

output/artifacts/
    missing_value_handler.joblib
    categorical_encoder.joblib
    feature_scaler.joblib
    feature_selector.joblib
```

## Model Evaluation

The model was evaluated using multiple performance metrics as required by the assignment.

- **Accuracy (~85%)**  
  Measures overall correctness of predictions.

- **Precision**  
  Indicates how many predicted approvals were actually correct.

- **Recall**  
  Measures how many actual approvals were correctly identified.

- **F1-Score (~0.90)**  
  Provides a balance between precision and recall.

These metrics demonstrate that the model performs reliably and maintains a good balance between false positives and false negatives.

## Logging

Prediction logs are stored in:

``` text
logs/predictions.log
```

Each entry records timestamp, loan ID, prediction, probability, and
confidence.

## Testing

Unit testing is implemented using **Pytest** to ensure correctness and reliability of the pipeline.

### Run Tests

```bash
pytest
```

## Software Engineering Practices

This project follows key Software Engineering for Machine Learning (SE4ML) principles:

### Modularity
- Each pipeline stage is implemented as an independent module
- Components such as preprocessing, training, and prediction are decoupled

### Maintainability
- Clean project structure with separation of concerns
- Reusable pipeline components

### Reproducibility
- Consistent pipeline execution from raw data to prediction
- Saved artifacts ensure repeatable results

### Testability
- Unit tests validate pipeline stages and transformations
- API contract checks validate runtime behavior

### Research vs Production
- Research experiments are separated from the production serving path
- `tmp_model_test.py` compares RandomForest vs GradientBoosting for proof of concept
- RandomForestClassifier was selected for production due to better validation F1 and generalization
- Production artifacts are persisted so the online service loads the exact same preprocessing state and model
- Deployment instructions distinguish between local development and production-ready startup using Uvicorn

### Test Coverage

- Pipeline execution test (ensures full pipeline runs without failure)
- API endpoint tests (health check and prediction contract)
- Data validation checks (LoanApplication and request payloads)
- Feature consistency test (ensures correct feature output)

## Reproducibility

The system ensures reproducibility by:

- Using a fixed pipeline for all transformations
- Saving trained models and preprocessing artifacts
- Allowing re-execution of the pipeline and training scripts

Steps:
1. Run pipeline: `python run_pipeline.py`
2. Train model: `python train_model.py`
3. Start API: `uvicorn app:app --reload`

### Logging
- Prediction outputs are logged for traceability and debugging

## Architectural Patterns

### Pipe-and-Filter

Used for the complete data preprocessing pipeline.

### Layered Architecture

``` text
Presentation Layer
      │
      ▼
FastAPI REST API
      │
      ▼
Prediction Service
      │
      ▼
Machine Learning Model
```

## Linting

A lightweight linting step helps maintain code quality. If `ruff` is installed, run:

```bash
ruff check .
```

## Future Enhancements

-   Docker deployment
-   User authentication
-   Model retraining
-   Prediction dashboard
-   Explainable AI
-   Cloud deployment

## Authors

**Course:** AIMLCZG546 -- Software Engineering for Machine Learning

**Institution:** BITS Pilani WILP

## Live Demo

https://loan-approval-ai-system-0f8h.onrender.com
