document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("loanForm");
    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    const errorBox = document.getElementById("errorBox");
    const submitBtn = form.querySelector("button[type='submit']");

    //----------------------------------------------------------
    // UI Helpers
    //----------------------------------------------------------

    function showLoading() {
        loading.style.display = "block";
        result.style.display = "none";
        errorBox.style.display = "none";

        submitBtn.disabled = true;
        submitBtn.innerHTML =
            `<span class="spinner-border spinner-border-sm me-2"></span>
             Predicting...`;
    }

    function hideLoading() {
        loading.style.display = "none";

        submitBtn.disabled = false;
        submitBtn.innerHTML = "🔍 Predict Loan Approval";
    }

    function showError(message) {
        errorBox.innerHTML = message;
        errorBox.style.display = "block";
    }

    function showResult(prediction) {

        document.getElementById("loanResult").textContent =
            prediction.loan_id;

        document.getElementById("statusResult").innerHTML =
            prediction.approved
                ? "<span class='badge bg-success'>APPROVED ✅</span>"
                : "<span class='badge bg-danger'>REJECTED ❌</span>";

        document.getElementById("probabilityResult").textContent =
            `${(prediction.approval_probability * 100).toFixed(2)}%`;

        document.getElementById("confidenceResult").textContent =
            `${(prediction.confidence * 100).toFixed(2)}%`;

        result.style.display = "block";

        result.scrollIntoView({
            behavior: "smooth",
            block: "center"
        });
    }

    //----------------------------------------------------------
    // Build JSON Payload
    //----------------------------------------------------------

    function buildPayload() {

        return {

            loan_id: document.getElementById("loan_id").value.trim(),

            gender: document.getElementById("gender").value,

            married: document.getElementById("married").value,

            dependents: document.getElementById("dependents").value,

            education: document.getElementById("education").value,

            self_employed: document.getElementById("self_employed").value,

            applicant_income: Number(
                document.getElementById("applicant_income").value
            ),

            coapplicant_income: Number(
                document.getElementById("coapplicant_income").value
            ),

            loan_amount: Number(
                document.getElementById("loan_amount").value
            ),

            loan_amount_term: Number(
                document.getElementById("loan_amount_term").value
            ),

            credit_history: Number(
                document.getElementById("credit_history").value
            ),

            property_area: document.getElementById("property_area").value
        };
    }

    //----------------------------------------------------------
    // Validate Input
    //----------------------------------------------------------

    function validate(payload) {

        if (!payload.loan_id) {
            return "Loan ID is required.";
        }

        if (payload.applicant_income < 0) {
            return "Applicant income cannot be negative.";
        }

        if (payload.loan_amount <= 0) {
            return "Loan amount must be greater than zero.";
        }

        return null;
    }

    //----------------------------------------------------------
    // Form Submit
    //----------------------------------------------------------

    form.addEventListener("submit", async (event) => {

        event.preventDefault();

        const payload = buildPayload();

        const validationError = validate(payload);

        if (validationError) {
            showError(validationError);
            return;
        }

        showLoading();

        try {

            const controller = new AbortController();

            const timeout = setTimeout(() => {
                controller.abort();
            }, 10000);

            const response = await fetch("/predict", {

                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify(payload),

                signal: controller.signal

            });

            clearTimeout(timeout);

            const data = await response.json();

            hideLoading();

            if (!response.ok || !data.success) {

                showError(
                    data.error ||
                    "Prediction could not be completed."
                );

                return;
            }

            showResult(data.prediction);

        }
        catch (error) {

            hideLoading();

            if (error.name === "AbortError") {

                showError(
                    "The request timed out. Please try again."
                );

            } else {

                showError(
                    "Unable to connect to the prediction server."
                );

                console.error(error);
            }
        }

    });

});