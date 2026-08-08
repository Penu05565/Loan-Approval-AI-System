from data_pipeline import LoanApplication
from predict import LoanPredictionService


def test_prediction_service_handles_valid_request() -> None:
    service = LoanPredictionService()
    application = LoanApplication(
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

    result = service.predict(application)

    assert result.loan_id == application.loan_id
    assert isinstance(result.approved, bool)
    assert 0.0 <= result.risk_probability <= 1.0
    assert 0.0 <= result.confidence <= 1.0


def test_prediction_service_rejects_invalid_credit_history() -> None:
    service = LoanPredictionService()
    try:
        application = LoanApplication(
            loan_id="LP999998",
            gender="Female",
            married="Yes",
            dependents="1",
            education="Graduate",
            self_employed="No",
            applicant_income=5200,
            coapplicant_income=1500,
            loan_amount=110,
            loan_amount_term=360,
            credit_history=2.0,
            property_area="Urban",
        )
    except ValueError as exc:
        assert "credit_history must be 0.0 or 1.0" in str(exc)
    else:
        raise AssertionError("LoanApplication did not reject invalid credit_history")
