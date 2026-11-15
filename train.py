"""Train the final XGBoost model and save it to model.bin.

Usage:
    uv run python data/download_data.py   # once, to extract the CSV
    uv run python train.py

The parameters are the best ones found in notebook.ipynb (section 7.3).
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score

from readmission import (
    clean,
    clean_name,
    load_raw,
    predict_proba,
    prepare_records,
    split_by_patient,
)


XGB_PARAMS = {
    "eta": 0.05,
    "max_depth": 6,
    "min_child_weight": 10,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 0,
}
NUM_ROUNDS = 236


def train(df_train):
    dv = DictVectorizer(sparse=False)
    X = dv.fit_transform(prepare_records(df_train))
    names = [clean_name(n) for n in dv.get_feature_names_out()]

    dtrain = xgb.DMatrix(X, label=df_train["target"].values, feature_names=names)
    model = xgb.train(XGB_PARAMS, dtrain, num_boost_round=NUM_ROUNDS)
    return dv, model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/diabetic_data.csv"))
    parser.add_argument("--output", type=Path, default=Path("model.bin"))
    args = parser.parse_args()

    df = clean(load_raw(args.data))
    df_full_train, df_test = split_by_patient(df, test_size=0.2)
    print(f"full train: {len(df_full_train)} encounters, test: {len(df_test)} encounters")

    dv, model = train(df_full_train)
    test_auc = roc_auc_score(df_test["target"].values, predict_proba(df_test, dv, model))
    print(f"Test ROC AUC: {test_auc:.4f}")

    with args.output.open("wb") as f_out:
        pickle.dump((dv, model), f_out)
    print(f"Model saved to {args.output}")


if __name__ == "__main__":
    main()
