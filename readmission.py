"""Data cleaning and feature preparation shared by train.py and predict.py.

The steps are the ones explored in notebook.ipynb (sections 3 and 7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GroupShuffleSplit


# Patients who died or were discharged to a hospice cannot be readmitted.
EXPIRED_OR_HOSPICE = [11, 13, 14, 19, 20, 21]

# `weight` is ~97 % missing; `examide` and `citoglipton` have a single value.
DROP_COLUMNS = ["weight", "examide", "citoglipton"]

ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]
DIAG_COLUMNS = ["diag_1", "diag_2", "diag_3"]

# Visit counts are very skewed (median 0), so they are log-transformed.
LOG_COLUMNS = ["number_outpatient", "number_emergency", "number_inpatient"]

NUMERICAL = [
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses",
]
CATEGORICAL = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "diag_1", "diag_2", "diag_3", "max_glu_serum", "A1Cresult",
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
    "change", "diabetesMed",
]
FEATURES = CATEGORICAL + NUMERICAL


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the UCI CSV. `?` is missing; the string "None" means "test not performed"."""
    return pd.read_csv(path, na_values="?", keep_default_na=False, low_memory=False)


def map_icd9(code: str) -> str:
    """Map a raw ICD-9 code to the categories of Strack et al. (2014)."""
    if code == "NA" or code.startswith(("V", "E")):
        return "Other"
    if code.startswith("250"):
        return "Diabetes"

    num = float(code.split(".")[0])
    if 390 <= num <= 459 or num == 785:
        return "Circulatory"
    if 460 <= num <= 519 or num == 786:
        return "Respiratory"
    if 520 <= num <= 579 or num == 787:
        return "Digestive"
    if 580 <= num <= 629 or num == 788:
        return "Genitourinary"
    if 800 <= num <= 999:
        return "Injury"
    if 710 <= num <= 739:
        return "Musculoskeletal"
    if 140 <= num <= 239:
        return "Neoplasms"
    return "Other"


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Column transformations applied to training data and to new patients alike."""
    df = df.copy()
    for col in CATEGORICAL:
        if col not in df.columns:
            df[col] = "NA"
    for col in ID_COLUMNS:
        # codes are integers: 1 -> "1" (not "1.0"), missing -> "NA"
        df[col] = df[col].map(lambda v: "NA" if pd.isna(v) else str(int(v)))
    df[CATEGORICAL] = df[CATEGORICAL].fillna("NA").astype(str)
    for col in DIAG_COLUMNS:
        df[col] = df[col].map(map_icd9)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning of the raw dataset, including the target."""
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_OR_HOSPICE)]
    df = df[df["gender"] != "Unknown/Invalid"].reset_index(drop=True)
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=DROP_COLUMNS + ["encounter_id", "readmitted"])
    return normalize(df)


def split_by_patient(df: pd.DataFrame, test_size: float, seed: int = 1):
    """Split so that all encounters of a patient end up on the same side."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_nbr"]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def prepare_records(df: pd.DataFrame) -> list[dict]:
    """Feature dictionaries for DictVectorizer."""
    data = df[FEATURES].copy()
    data[LOG_COLUMNS] = np.log1p(data[LOG_COLUMNS])
    return data.to_dict(orient="records")


def clean_name(name: str) -> str:
    """XGBoost does not accept [, ] or < in feature names (e.g. age=[0-10))."""
    return name.replace("[", "(").replace("]", ")").replace("<", "lt")


def predict_proba(df: pd.DataFrame, dv, model) -> np.ndarray:
    """Probability of readmission within 30 days for each row of a normalized frame."""
    X = dv.transform(prepare_records(df))
    names = [clean_name(n) for n in dv.get_feature_names_out()]
    return model.predict(xgb.DMatrix(X, feature_names=names))
