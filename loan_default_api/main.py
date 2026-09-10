"""FastAPI service for loan-default predictions."""

from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Loan_Default_model.pkl"
COLUMNS_PATH = BASE_DIR / "model_columns.pkl"

CATEGORICAL_COLUMNS = [
    "Education",
    "EmploymentType",
    "MaritalStatus",
    "HasMortgage",
    "HasDependents",
    "LoanPurpose",
    "HasCoSigner",
]


class ModelStore:
    """Keeps model artifacts in memory after a successful startup."""

    model: Any | None = None
    columns: list[str] = []


store = ModelStore()


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load and validate model artifacts once, before accepting traffic."""
    if not MODEL_PATH.is_file() or not COLUMNS_PATH.is_file():
        raise RuntimeError("Model artifacts are missing from the application directory.")

    store.model = joblib.load(MODEL_PATH)
    store.columns = list(joblib.load(COLUMNS_PATH))

    if not store.columns or not hasattr(store.model, "predict_proba"):
        raise RuntimeError("Model artifacts are invalid or incompatible.")

    yield

    store.model = None
    store.columns = []


app = FastAPI(
    title="Loan Default Prediction API",
    version="1.0.0",
    description="Estimate loan-default risk from applicant information.",
    lifespan=lifespan,
)


class Education(str, Enum):
    high_school = "High School"
    bachelors = "Bachelor's"
    masters = "Master's"
    phd = "PhD"


class EmploymentType(str, Enum):
    full_time = "Full-time"
    part_time = "Part-time"
    self_employed = "Self-employed"
    unemployed = "Unemployed"


class MaritalStatus(str, Enum):
    single = "Single"
    married = "Married"
    divorced = "Divorced"


class YesNo(str, Enum):
    yes = "Yes"
    no = "No"


class LoanPurpose(str, Enum):
    auto = "Auto"
    business = "Business"
    education = "Education"
    home = "Home"
    other = "Other"


class LoanApplicant(BaseModel):
    Age: int = Field(..., ge=18, le=100, examples=[35])
    Income: float = Field(..., gt=0, examples=[60000])
    LoanAmount: float = Field(..., gt=0, examples=[15000])
    CreditScore: int = Field(..., ge=300, le=850, examples=[650])
    MonthsEmployed: int = Field(..., ge=0, examples=[24])
    NumCreditLines: int = Field(..., ge=0, examples=[3])
    InterestRate: float = Field(..., gt=0, le=100, examples=[12.5])
    DTIRatio: float = Field(..., ge=0, le=1, examples=[0.35])
    Education: Education
    EmploymentType: EmploymentType
    MaritalStatus: MaritalStatus
    HasMortgage: YesNo
    HasDependents: YesNo
    LoanPurpose: LoanPurpose
    HasCoSigner: YesNo


class PredictionResponse(BaseModel):
    prediction: str
    probability_of_default: float
    risk_level: str


def prepare_features(applicant: LoanApplicant) -> pd.DataFrame:
    """Apply the same feature engineering and encoding used for training."""
    input_df = pd.DataFrame([applicant.model_dump()])
    input_df["Loan burden"] = input_df["LoanAmount"] / input_df["Income"]

    encoded = pd.get_dummies(
        input_df,
        columns=CATEGORICAL_COLUMNS,
        drop_first=True,
    )

    return encoded.reindex(columns=store.columns, fill_value=0)


@app.get("/", tags=["Health"])
def read_root() -> dict[str, str]:
    """Return a simple service-status message."""
    return {"status": "Loan Default Prediction API is running"}


@app.get("/health", tags=["Health"])
def health_check() -> dict[str, str]:
    """Report whether the trained model is ready to serve predictions."""
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(applicant: LoanApplicant) -> PredictionResponse:
    """Return default classification, probability, and risk band."""
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    try:
        features = prepare_features(applicant)
        probability = float(store.model.predict_proba(features)[0][1])
        prediction = int(store.model.predict(features)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Unable to generate prediction.",
        ) from exc

    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.6:
        risk_level = "Medium"
    else:
        risk_level = "High"

    return PredictionResponse(
        prediction="Default" if prediction else "No Default",
        probability_of_default=round(probability, 4),
        risk_level=risk_level,
    )


@app.get("/feature-importance", tags=["Model"])
def feature_importance() -> list[dict[str, float | str]]:
    """Return the model's global feature importances in descending order."""
    if store.model is None:
        raise HTTPException(status_code=503, detail="Model is not ready.")

    if not hasattr(store.model, "feature_importances_"):
        raise HTTPException(
            status_code=501,
            detail="Feature importance is unavailable for this model.",
        )

    importance = pd.DataFrame(
        {
            "feature": store.columns,
            "importance": store.model.feature_importances_,
        }
    ).sort_values(by="importance", ascending=False)

    return importance.to_dict(orient="records")
