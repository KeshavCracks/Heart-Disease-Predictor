"""Load, audit and prepare the official UCI Heart Disease data."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config

REQUIRED_COLUMNS = {
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
    config.SOURCE_COLUMN,
}


def load_raw_data(path: str | Path = config.DATA_PATH) -> pd.DataFrame:
    """Read the merged UCI CSV. Run ``python scripts/download_data.py`` if absent."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, na_values=["?", "", " "])


def missing_value_report(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a sorted per-column missing-value summary."""
    report = pd.DataFrame(
        {
            "missing_count": frame.isna().sum(),
            "missing_percent": (frame.isna().mean() * 100).round(1),
            "observed_count": frame.notna().sum(),
        }
    )
    return report.sort_values("missing_percent", ascending=False)


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize numeric columns, mark impossible zero measurements as missing,
    and map the original 0–4 diagnosis to a binary target (0 vs. >0).

    Missing-value filling, encoding and scaling are deliberately left to the
    model pipeline, which is fitted only on training folds.
    """
    absent = REQUIRED_COLUMNS.difference(raw.columns)
    if absent:
        raise ValueError(f"Dataset is missing required columns: {sorted(absent)}")

    frame = raw.copy()
    numeric_columns = [
        "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
        "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    # In the source data, zero in these two measurements is a missing-value code.
    for column in ("trestbps", "chol"):
        frame.loc[frame[column].eq(0), column] = np.nan

    frame = frame.loc[frame["num"].notna()].copy()
    frame[config.TARGET] = (frame["num"] > 0).astype(int)
    frame[config.SOURCE_COLUMN] = frame[config.SOURCE_COLUMN].fillna("Unknown").astype(str)
    return frame.reset_index(drop=True)


def _category_code(value: Any) -> object:
    """Convert numeric-looking category codes such as 1.0 to the string '1'."""
    if pd.isna(value):
        return np.nan
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def feature_frame(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    """Return exactly the columns for a feature set in pipeline-ready types."""
    if feature_set not in config.FEATURE_SETS:
        raise ValueError(
            f"Unknown feature set {feature_set!r}. Choose from {list(config.FEATURE_SETS)}."
        )
    features = config.FEATURE_SETS[feature_set]
    X = frame[features].copy()
    for column in config.NUMERIC_FEATURES:
        if column in X:
            X[column] = pd.to_numeric(X[column], errors="coerce")
    for column in config.CATEGORICAL_FEATURES:
        if column in X:
            X[column] = pd.to_numeric(X[column], errors="coerce").map(_category_code).astype(object)
    return X


def target_and_groups(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return binary outcome and source-cohort labels for grouped validation."""
    y = frame[config.TARGET].astype(int).rename(config.TARGET)
    groups = frame[config.SOURCE_COLUMN].astype(str).rename(config.SOURCE_COLUMN)
    return y, groups
