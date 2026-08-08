def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["service"] == "Loan Approval Prediction API"
    assert payload["model_loaded"] is True


def test_predict_endpoint_valid_request(client):
    payload = {
        "loan_id": "LP999999",
        "gender": "Female",
        "married": "Yes",
        "dependents": "1",
        "education": "Graduate",
        "self_employed": "No",
        "applicant_income": 5200,
        "coapplicant_income": 1500,
        "loan_amount": 110,
        "loan_amount_term": 360,
        "credit_history": 1.0,
        "property_area": "Urban",
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["prediction"]["loan_id"] == payload["loan_id"]
    assert isinstance(data["prediction"]["approved"], bool)
    assert 0.0 <= data["prediction"]["risk_probability"] <= 1.0
    assert 0.0 <= data["prediction"]["confidence"] <= 1.0
