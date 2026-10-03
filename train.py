"""Train and compare two feature sets and three model families.

From the project root:
    python train.py             # compact hyperparameter search
    python train.py --quick     # one configuration per model for a fast smoke run

Model selection uses GroupKFold by source cohort on the training partition.
The separate stratified test partition is never used to select models.
"""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.model_selection import GroupKFold, GridSearchCV, train_test_split

from src import config
from src.data import clean_data, feature_frame, load_raw_data, missing_value_report, target_and_groups
from src.evaluation import compute_metrics
from src.modeling import build_pipeline, model_search_specs

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def _plot_model_comparison(
    y_test: pd.Series,
    probabilities: dict[str, np.ndarray],
    selected_key: str,
) -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    for configuration, prob in probabilities.items():
        feature_set, model_key = configuration.split("__", maxsplit=1)
        feature_label = "11-field" if feature_set == "core11" else "13-field"
        label = f"{feature_label} · {config.MODEL_LABELS[model_key]}"
        RocCurveDisplay.from_predictions(y_test, prob, name=label, ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", linewidth=1)
    ax.set_title("ROC curves — patient-level held-out test split")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28)
    fig.savefig(config.REPORTS_DIR / "roc_curves.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    selected_prob = probabilities[selected_key]
    selected_pred = (selected_prob >= config.DECISION_THRESHOLD).astype(int)
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        selected_pred,
        labels=[0, 1],
        display_labels=["Negative (0)", "Positive (1)"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set_title("Selected model — held-out test split")
    fig.tight_layout()
    fig.savefig(config.REPORTS_DIR / "selected_confusion_matrix.png", dpi=150)
    plt.close(fig)


def run_training(quick: bool = False) -> dict:
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_data()
    data = clean_data(raw)
    missing_value_report(data[config.FEATURE_SETS["all13"]]).to_csv(
        config.REPORTS_DIR / "missing_values.csv"
    )
    y, groups = target_and_groups(data)

    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    n_groups = groups_train.nunique()
    if n_groups < 2:
        raise ValueError("Grouped cross-validation requires at least two source cohorts.")
    cv = GroupKFold(n_splits=min(4, n_groups))
    specs = model_search_specs(quick=quick)

    rows: list[dict] = []
    probabilities: dict[str, np.ndarray] = {}
    pipelines: dict[str, object] = {}

    for feature_set, feature_names in config.FEATURE_SETS.items():
        X = feature_frame(data, feature_set)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        for model_key, (estimator, parameter_grid) in specs.items():
            configuration = f"{feature_set}__{model_key}"
            print(f"Training {configuration} with grouped CV ...")
            search = GridSearchCV(
                estimator=build_pipeline(feature_set, estimator),
                param_grid=parameter_grid,
                scoring=SCORING,
                refit="roc_auc",
                cv=cv,
                n_jobs=-1,
                error_score="raise",
                return_train_score=False,
            )
            search.fit(X_train, y_train, groups=groups_train)
            best = search.best_estimator_
            test_prob = best.predict_proba(X_test)[:, 1]
            test_metrics = compute_metrics(y_test, test_prob, config.DECISION_THRESHOLD)
            cv_index = search.best_index_
            cv_metrics = {
                metric: float(search.cv_results_[f"mean_test_{metric}"][cv_index])
                for metric in SCORING
            }
            cv_std = {
                metric: float(search.cv_results_[f"std_test_{metric}"][cv_index])
                for metric in SCORING
            }
            label = f"{config.FEATURE_SET_LABELS[feature_set]} · {config.MODEL_LABELS[model_key]}"
            rows.append(
                {
                    "configuration": configuration,
                    "feature_set": feature_set,
                    "feature_set_label": config.FEATURE_SET_LABELS[feature_set],
                    "model": model_key,
                    "model_label": config.MODEL_LABELS[model_key],
                    **{f"cv_{k}": v for k, v in cv_metrics.items()},
                    **{f"cv_{k}_std": v for k, v in cv_std.items()},
                    **{f"test_{k}": v for k, v in test_metrics.items()},
                    "best_params": json.dumps(search.best_params_, default=str, sort_keys=True),
                    "n_features": len(feature_names),
                }
            )
            probabilities[configuration] = test_prob
            pipelines[configuration] = best
            print(
                f"  grouped-CV ROC-AUC={cv_metrics['roc_auc']:.3f} "
                f"(±{cv_std['roc_auc']:.3f}); test ROC-AUC={test_metrics['roc_auc']:.3f}; "
                f"test recall={test_metrics['recall']:.3f}"
            )

    results = pd.DataFrame(rows).sort_values(
        ["cv_roc_auc", "cv_recall", "cv_f1"], ascending=False
    ).reset_index(drop=True)
    # The four cohorts are a small validation sample. If candidates fall within
    # one standard error of the top grouped-CV AUC, prefer the less-missing
    # 11-field input set, then the simpler model family. Recall breaks any
    # remaining tie. This avoids treating tiny CV-score differences as proof.
    top_candidate = results.iloc[0]
    auc_standard_error = float(top_candidate["cv_roc_auc_std"]) / np.sqrt(cv.get_n_splits())
    minimum_eligible_auc = float(top_candidate["cv_roc_auc"]) - auc_standard_error
    eligible = results.loc[results["cv_roc_auc"] >= minimum_eligible_auc].copy()
    eligible["_feature_parsimony"] = eligible["feature_set"].map({"core11": 0, "all13": 1})
    eligible["_model_parsimony"] = eligible["model"].map(
        {"logistic_regression": 0, "random_forest": 1, "xgboost": 2}
    )
    selected = eligible.sort_values(
        ["_feature_parsimony", "_model_parsimony", "cv_recall", "cv_roc_auc"],
        ascending=[True, True, False, False],
    ).iloc[0]
    selected_key = str(selected["configuration"])
    selected_pipeline = pipelines[selected_key]
    joblib.dump(selected_pipeline, config.MODELS_DIR / "best_model.joblib")
    results.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False, float_format="%.5f")
    _plot_model_comparison(y_test, probabilities, selected_key)

    selected_test_metrics = {
        key.removeprefix("test_"): float(selected[key])
        for key in selected.index
        if key.startswith("test_")
    }
    missing = missing_value_report(data)
    missing_percent = {
        feature: float(missing.loc[feature, "missing_percent"])
        for feature in config.EXTRA_FEATURES
    }
    info = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_configuration": selected_key,
        "best_model": str(selected["model"]),
        "best_model_label": str(selected["model_label"]),
        "feature_set": str(selected["feature_set"]),
        "feature_set_label": str(selected["feature_set_label"]),
        "features": config.FEATURE_SETS[str(selected["feature_set"])],
        "selection_rule": "One-standard-error rule on grouped-CV ROC-AUC; within that range prefer the 11-field set, then the simpler model; recall is a tie-breaker.",
        "best_cv_roc_auc": float(top_candidate["cv_roc_auc"]),
        "one_se_auc_cutoff": minimum_eligible_auc,
        "eligible_within_one_se": eligible["configuration"].astype(str).tolist(),
        "decision_threshold": config.DECISION_THRESHOLD,
        "positive_class_definition": "Original UCI num > 0",
        "dataset_rows": int(len(data)),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "source_cohorts": sorted(groups.unique().tolist()),
        "extra_feature_missing_percent": missing_percent,
        "grouped_cv_metrics": {
            metric: float(selected[f"cv_{metric}"]) for metric in SCORING
        },
        "grouped_cv_std": {
            metric: float(selected[f"cv_{metric}_std"]) for metric in SCORING
        },
        "held_out_test_metrics": selected_test_metrics,
        "library_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgboost.__version__,
        },
        "privacy": "Inputs are processed in memory only; this app does not save submissions.",
        "educational_only": True,
    }
    (config.MODELS_DIR / "model_info.json").write_text(json.dumps(info, indent=2))

    print("\nCandidate comparison (sorted by grouped-CV ROC-AUC):")
    print(
        results[[
            "configuration", "cv_roc_auc", "cv_recall", "test_roc_auc", "test_recall",
            "test_specificity", "test_false_negatives", "test_false_positives",
        ]].round(3).to_string(index=False)
    )
    print(f"\nSelected: {selected_key}")
    print(f"Saved model and reports under {config.MODELS_DIR.relative_to(config.ROOT)} and reports/.")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use one small parameter configuration per model for a faster smoke run.",
    )
    args = parser.parse_args()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        run_training(quick=args.quick)


if __name__ == "__main__":
    main()
