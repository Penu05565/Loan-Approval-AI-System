"""
tests/test_model_inference.py
==============================
Assignment II, Objective 2, Item 7b: "Testing model inference (e.g.,
output shape/range checks, invariance/directional tests)".

Runs against the real, currently-deployed LoanPredictionService (the same
fitted pipeline artifacts + model used by the FastAPI /predict endpoint),
so these tests exercise the actual serving path end-to-end, not a mock.

(Output shape/range checks already exist in tests/test_predict.py and
tests/test_api.py; this file adds the invariance and directional checks
that were not yet covered.)
"""

import pytest

from data_pipeline import LoanApplication
from predict import LoanPredictionService

BASE_PAYLOAD = dict(
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


@pytest.fixture(scope="module")
def service():
    return LoanPredictionService()


def test_invariance_to_loan_id(service):
    """Loan_ID is an identifier, not a model feature -- changing it must
    not change the prediction."""
    payload_a = dict(BASE_PAYLOAD, loan_id="LP-AAAA")
    payload_b = dict(BASE_PAYLOAD, loan_id="LP-BBBB")

    result_a = service.predict(LoanApplication(**payload_a))
    result_b = service.predict(LoanApplication(**payload_b))

    assert result_a.risk_probability == pytest.approx(result_b.risk_probability)
    assert result_a.approved == result_b.approved


def test_directional_credit_history(service):
    """All else equal, having a credit history should never hurt the
    approval probability relative to not having one."""
    good_credit = dict(BASE_PAYLOAD, loan_id="LP-GOOD", credit_history=1.0)
    poor_credit = dict(BASE_PAYLOAD, loan_id="LP-POOR", credit_history=0.0)

    result_good = service.predict(LoanApplication(**good_credit))
    result_poor = service.predict(LoanApplication(**poor_credit))

    approval_prob_good = 1 - result_good.risk_probability
    approval_prob_poor = 1 - result_poor.risk_probability

    assert approval_prob_good >= approval_prob_poor, (
        "Applicant with credit_history=1 should not score lower than an "
        "otherwise identical applicant with credit_history=0"
    )


def test_directional_higher_loan_amount_relative_to_income(service):
    """All else equal, asking for a much larger loan against the same
    income should never *increase* the approval probability -- it should
    stay the same or decrease as the loan-to-income ratio worsens."""
    modest_loan = dict(BASE_PAYLOAD, loan_id="LP-MODEST", loan_amount=80)
    large_loan = dict(BASE_PAYLOAD, loan_id="LP-LARGE", loan_amount=500)

    result_modest = service.predict(LoanApplication(**modest_loan))
    result_large = service.predict(LoanApplication(**large_loan))

    approval_prob_modest = 1 - result_modest.risk_probability
    approval_prob_large = 1 - result_large.risk_probability

    assert approval_prob_large <= approval_prob_modest, (
        "A much larger loan against the same income should not score a "
        "higher approval probability than a modest loan"
    )
