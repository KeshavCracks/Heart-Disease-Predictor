"""Leakage-safe preprocessing and small model-search spaces."""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config


def build_preprocessor(feature_set: str) -> ColumnTransformer:
    """Build imputers/encoders/scaler inside a pipeline, fitted per training fold."""
    if feature_set not in config.FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")
    selected = config.FEATURE_SETS[feature_set]
    numeric = [name for name in config.NUMERIC_FEATURES if name in selected]
    categorical = [name for name in config.CATEGORICAL_FEATURES if name in selected]

    numeric_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric),
            ("categorical", categorical_pipeline, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(feature_set: str, estimator) -> Pipeline:
    """Keep the full preprocessing + estimator together for safe inference."""
    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(feature_set)),
            ("model", estimator),
        ]
    )


def model_search_specs(quick: bool = False) -> dict:
    """Return estimator and compact hyperparameter grid for each model family."""
    logistic = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        random_state=config.RANDOM_STATE,
    )
    forest = RandomForestClassifier(
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    boost = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=config.RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )

    if quick:
        return {
            "logistic_regression": (logistic, {"model__C": [1.0]}),
            "random_forest": (
                forest,
                {"model__n_estimators": [200], "model__max_depth": [6]},
            ),
            "xgboost": (
                boost,
                {
                    "model__n_estimators": [150],
                    "model__max_depth": [3],
                    "model__learning_rate": [0.1],
                },
            ),
        }

    return {
        "logistic_regression": (
            logistic,
            {
                "model__C": [0.1, 1.0, 10.0],
                "model__class_weight": [None, "balanced"],
            },
        ),
        "random_forest": (
            forest,
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [2, 5],
            },
        ),
        "xgboost": (
            boost,
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [2, 4],
                "model__learning_rate": [0.03, 0.1],
                "model__subsample": [0.8, 1.0],
            },
        ),
    }
