"""Streamlit interface for the Loan Default Prediction API."""

import os

import requests
import streamlit as st


DEFAULT_API_URL = "https://loan-default-prediction-5.onrender.com/"
API_URL = os.getenv("LOAN_API_URL", DEFAULT_API_URL).rstrip("/")


def build_payload(
    age: int,
    income: float,
    loan_amount: float,
    credit_score: int,
    months_employed: int,
    num_credit_lines: int,
    interest_rate: float,
    dti_ratio: float,
    education: str,
    employment_type: str,
    marital_status: str,
    has_mortgage: str,
    has_dependents: str,
    loan_purpose: str,
    has_cosigner: str,
) -> dict[str, int | float | str]:
    """Build the request body expected by the prediction API."""
    return {
        "Age": age,
        "Income": income,
        "LoanAmount": loan_amount,
        "CreditScore": credit_score,
        "MonthsEmployed": months_employed,
        "NumCreditLines": num_credit_lines,
        "InterestRate": interest_rate,
        "DTIRatio": dti_ratio,
        "Education": education,
        "EmploymentType": employment_type,
        "MaritalStatus": marital_status,
        "HasMortgage": has_mortgage,
        "HasDependents": has_dependents,
        "LoanPurpose": loan_purpose,
        "HasCoSigner": has_cosigner,
    }


st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="💰",
    layout="wide",
)

st.markdown(
    """
    <style>
        .block-container {
            max-width: 1100px;
            padding-top: 2.5rem;
        }
        .hero {
            padding: 1.5rem 1.75rem;
            margin-bottom: 1.5rem;
            color: white;
            background: linear-gradient(120deg, #12355b, #1f7a8c);
            border-radius: 16px;
        }
        .hero h1 {
            margin: 0;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0.4rem 0 0;
            opacity: 0.9;
        }
        div.stButton > button {
            width: 100%;
            padding: 0.6rem;
            font-weight: 600;
            border-radius: 8px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
        <h1>Loan Default Risk Predictor</h1>
        <p>Enter applicant details to estimate the likelihood of loan default.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("loan_form", clear_on_submit=False):
    financial, profile = st.columns(2, gap="large")

    with financial:
        st.subheader("Financial profile")
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        income = st.number_input(
            "Annual income (₹)",
            min_value=1.0,
            value=60_000.0,
            step=1_000.0,
        )
        loan_amount = st.number_input(
            "Loan amount (₹)",
            min_value=1.0,
            value=15_000.0,
            step=1_000.0,
        )
        credit_score = st.number_input(
            "Credit score",
            min_value=300,
            max_value=850,
            value=650,
        )
        months_employed = st.number_input("Months employed", min_value=0, value=24)
        num_credit_lines = st.number_input(
            "Number of credit lines",
            min_value=0,
            value=3,
        )
        interest_rate = st.number_input(
            "Interest rate (%)",
            min_value=0.1,
            max_value=100.0,
            value=12.5,
        )
        dti_ratio = st.number_input(
            "Debt-to-income ratio",
            min_value=0.0,
            max_value=1.0,
            value=0.35,
            step=0.01,
        )

    with profile:
        st.subheader("Applicant profile")
        education = st.selectbox(
            "Education",
            ["High School", "Bachelor's", "Master's", "PhD"],
        )
        employment_type = st.selectbox(
            "Employment type",
            ["Full-time", "Part-time", "Self-employed", "Unemployed"],
        )
        marital_status = st.selectbox(
            "Marital status",
            ["Single", "Married", "Divorced"],
        )
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["Auto", "Business", "Education", "Home", "Other"],
        )
        has_mortgage = st.selectbox("Has mortgage", ["Yes", "No"])
        has_dependents = st.selectbox("Has dependents", ["Yes", "No"])
        has_cosigner = st.selectbox("Has co-signer", ["Yes", "No"])

    submitted = st.form_submit_button("Assess default risk", type="primary")


if submitted:
    payload = build_payload(
        age,
        income,
        loan_amount,
        credit_score,
        months_employed,
        num_credit_lines,
        interest_rate,
        dti_ratio,
        education,
        employment_type,
        marital_status,
        has_mortgage,
        has_dependents,
        loan_purpose,
        has_cosigner,
    )

    try:
        with st.spinner("Calculating risk assessment..."):
            response = requests.post(
                f"{API_URL}/predict",
                json=payload,
                timeout=10,
            )

        response.raise_for_status()
        result = response.json()

        risk = result["risk_level"]
        probability = result["probability_of_default"]

        st.subheader("Assessment")
        left, right = st.columns(2)
        left.metric("Probability of default", f"{probability:.2%}")
        right.metric("Prediction", result["prediction"])

        if risk == "Low":
            st.success(f"Risk level: {risk}")
        elif risk == "Medium":
            st.warning(f"Risk level: {risk}")
        else:
            st.error(f"Risk level: {risk}")

    except requests.exceptions.ConnectionError:
        st.error(
            f"Cannot reach the prediction API at {API_URL}. "
            "Start the FastAPI server and try again."
        )
    except requests.exceptions.Timeout:
        st.error("The prediction service took too long to respond. Please try again.")
    except requests.exceptions.HTTPError:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"The prediction service rejected the request: {detail}")
    except (KeyError, ValueError):
        st.error("The prediction service returned an unexpected response. Please try again.")
