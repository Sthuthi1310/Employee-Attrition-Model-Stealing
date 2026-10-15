"""Reusable preprocessing utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_CANDIDATES = ["attrition", "employee attrition", "left", "churn", "turnover"]
ID_COLUMNS = ["employee_id", "emp_id", "id"]


def identify_target_column(df: pd.DataFrame) -> str:
    """Automatically detect the attrition target column."""
    normalized_cols = {col: col.strip().lower().replace(" ", "_") for col in df.columns}

    for original, normalized in normalized_cols.items():
        if normalized in TARGET_CANDIDATES or "attrition" in normalized:
            return original

    raise ValueError(
        "Could not automatically identify target column. "
        f"Expected one of: {TARGET_CANDIDATES}"
    )


def preprocess_features(df: pd.DataFrame, target_col: str):
    """
    Prepare feature matrix X and encoded target vector y.
    Returns feature names list and column type groups for the pipeline.
    """
    data = df.copy()

    missing_before = data.isnull().sum().sum()
    if missing_before > 0:
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        categorical_cols = data.select_dtypes(include=["object"]).columns

        for col in numeric_cols:
            if col != target_col and data[col].isnull().any():
                data[col].fillna(data[col].median(), inplace=True)

        for col in categorical_cols:
            if col != target_col and data[col].isnull().any():
                data[col].fillna(data[col].mode()[0], inplace=True)

    target_mapping = {"Yes": 1, "No": 0, "yes": 1, "no": 0, 1: 1, 0: 0}
    y = data[target_col].map(target_mapping)

    if y.isnull().any():
        raise ValueError(f"Unrecognized target values in column '{target_col}'.")

    drop_cols = [target_col]
    for col in data.columns:
        if col.strip().lower().replace(" ", "_") in ID_COLUMNS:
            drop_cols.append(col)

    X = data.drop(columns=drop_cols)

    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    return X, y, numeric_features, categorical_features


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """Create preprocessing pipeline: scale numeric, encode categorical."""
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
