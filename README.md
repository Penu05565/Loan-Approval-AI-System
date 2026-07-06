# Loan Approval Risk Prediction System

## Overview

This project implements an **AI-Powered Loan Approval Risk Prediction
System** following **Software Engineering for Machine Learning (SE4ML)**
principles. It demonstrates an end-to-end machine learning workflow
including data preparation, model training, online prediction, and
deployment using a Flask web application.

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
Gradient Boosting Model Training
    │
    ▼
Trained Model (.pkl)
    │
    ▼
Prediction Service
    │
    ▼
Flask REST API
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

-   **Loan_Status = 1** → Not Default
-   **Loan_Status = 0** → Default

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
├── run_pipeline.py
├── train_model.py
├── predict.py
├── README.md
├── requirements.txt
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
|   ├── predictions.log
├── static/
│   ├── style.css
│   └── script.js
├── templates/
│   └── index.html
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
    └── pipeline.py               # Orchestrator wiring all filters together
```

## Technologies

-   Python
-   Pandas
-   NumPy
-   Scikit-learn
-   Joblib
-   Flask
-   HTML5
-   CSS3
-   Bootstrap 5
-   JavaScript

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

or

``` bash
pip install pandas numpy scikit-learn flask joblib
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
python app.py
```

Open:

``` text
http://127.0.0.1:5000
```

## REST API

### GET /

Returns the web application.

### GET /health

Example:

``` json
{
  "status": "healthy",
  "model_loaded": true
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
    "approval_probability":0.9642,
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

## Logging

Prediction logs are stored in:

``` text
logs/predictions.log
```

Each entry records timestamp, loan ID, prediction, probability, and
confidence.

## Architectural Patterns

### Pipe-and-Filter

Used for the complete data preprocessing pipeline.

### Layered Architecture

``` text
Presentation Layer
      │
      ▼
Flask REST API
      │
      ▼
Prediction Service
      │
      ▼
Machine Learning Model
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