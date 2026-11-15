"""Prediction service: risk of readmission within 30 days for a discharged patient.

Run locally:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696

Endpoints:
    GET  /health   service status
    POST /predict  one patient record -> readmission probability
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

from readmission import normalize, predict_proba


MODEL_PATH = Path(os.getenv("MODEL_PATH", Path(__file__).with_name("model.bin")))

# Patients at or above this probability are flagged for follow-up. With 0.2 the test
# set flags ~9 % of the encounters, of which ~28 % are readmitted (2.4x the base rate).
THRESHOLD = float(os.getenv("RISK_THRESHOLD", "0.2"))

with MODEL_PATH.open("rb") as f_in:
    dv, model = pickle.load(f_in)

app = FastAPI(title="Hospital readmission prediction")


class Patient(BaseModel):
    """One encounter at discharge time, with the raw values of the UCI dataset.

    Categorical fields are optional: a missing value is treated as "not recorded".
    """

    model_config = ConfigDict(populate_by_name=True)

    # numerical features (required)
    time_in_hospital: int = Field(ge=1, examples=[6])
    num_lab_procedures: int = Field(ge=0, examples=[50])
    num_procedures: int = Field(ge=0, examples=[1])
    num_medications: int = Field(ge=0, examples=[20])
    number_outpatient: int = Field(ge=0, examples=[0])
    number_emergency: int = Field(ge=0, examples=[1])
    number_inpatient: int = Field(ge=0, examples=[3])
    number_diagnoses: int = Field(ge=1, examples=[9])

    # demographics and admission
    race: str | None = None
    gender: str | None = Field(default=None, examples=["Female"])
    age: str | None = Field(default=None, examples=["[70-80)"])
    admission_type_id: int | None = Field(default=None, examples=[1])
    discharge_disposition_id: int | None = Field(default=None, examples=[22])
    admission_source_id: int | None = Field(default=None, examples=[7])
    payer_code: str | None = None
    medical_specialty: str | None = None

    # raw ICD-9 diagnosis codes
    diag_1: str | None = Field(default=None, examples=["428"])
    diag_2: str | None = Field(default=None, examples=["250.02"])
    diag_3: str | None = Field(default=None, examples=["414"])

    # lab results ("None" = test not performed)
    max_glu_serum: str | None = None
    A1Cresult: str | None = None

    # medications: "No", "Steady", "Up" or "Down"
    metformin: str | None = None
    repaglinide: str | None = None
    nateglinide: str | None = None
    chlorpropamide: str | None = None
    glimepiride: str | None = None
    acetohexamide: str | None = None
    glipizide: str | None = None
    glyburide: str | None = None
    tolbutamide: str | None = None
    pioglitazone: str | None = None
    rosiglitazone: str | None = None
    acarbose: str | None = None
    miglitol: str | None = None
    troglitazone: str | None = None
    tolazamide: str | None = None
    insulin: str | None = Field(default=None, examples=["Up"])
    glyburide_metformin: str | None = Field(default=None, alias="glyburide-metformin")
    glipizide_metformin: str | None = Field(default=None, alias="glipizide-metformin")
    glimepiride_pioglitazone: str | None = Field(default=None, alias="glimepiride-pioglitazone")
    metformin_rosiglitazone: str | None = Field(default=None, alias="metformin-rosiglitazone")
    metformin_pioglitazone: str | None = Field(default=None, alias="metformin-pioglitazone")
    change: str | None = None
    diabetesMed: str | None = None


class Prediction(BaseModel):
    readmission_probability: float
    high_risk: bool
    threshold: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_PATH.name, "threshold": THRESHOLD}


@app.post("/predict")
def predict(patient: Patient) -> Prediction:
    record = patient.model_dump(by_alias=True)
    probability = float(predict_proba(normalize(pd.DataFrame([record])), dv, model)[0])
    return Prediction(
        readmission_probability=round(probability, 4),
        high_risk=probability >= THRESHOLD,
        threshold=THRESHOLD,
    )
