"""Project paths, feature definitions and reproducibility settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease_uci.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

RANDOM_STATE = 42
TEST_SIZE = 0.20
DECISION_THRESHOLD = 0.50  # educational default only; not a clinical threshold

# The hospital/cohort label is retained for group-aware validation, but is never
# provided as an input to the prediction model.
SOURCE_COLUMN = "dataset"
RAW_TARGET = "num"
TARGET = "target"

CORE_FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope",
]
EXTRA_FEATURES = ["ca", "thal"]
FEATURE_SETS = {
    "core11": CORE_FEATURES,
    "all13": CORE_FEATURES + EXTRA_FEATURES,
}
FEATURE_SET_LABELS = {
    "core11": "Core 11 fields (omits ca and thal)",
    "all13": "All 13 fields (includes ca and thal)",
}

NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalch", "oldpeak"]
CATEGORICAL_FEATURES = [
    "sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal",
]

FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest pain type",
    "trestbps": "Resting blood pressure (mm Hg)",
    "chol": "Serum cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar > 120 mg/dL",
    "restecg": "Resting ECG result",
    "thalch": "Maximum heart rate achieved",
    "exang": "Exercise-induced angina",
    "oldpeak": "ST depression (oldpeak)",
    "slope": "Peak-exercise ST-segment slope",
    "ca": "Major vessels coloured by fluoroscopy",
    "thal": "Thalassemia test category",
}

# The stored values intentionally match the UCI codes. Display labels explain
# those codes so the app can pass values in the format used during training.
CATEGORY_OPTIONS = {
    "sex": {"0": "Female (0)", "1": "Male (1)"},
    "cp": {
        "1": "Typical angina (1)",
        "2": "Atypical angina (2)",
        "3": "Non-anginal pain (3)",
        "4": "Asymptomatic (4)",
    },
    "fbs": {"0": "No (0)", "1": "Yes (1)"},
    "restecg": {
        "0": "Normal (0)",
        "1": "ST-T abnormality (1)",
        "2": "LV hypertrophy (2)",
    },
    "exang": {"0": "No (0)", "1": "Yes (1)"},
    "slope": {
        "1": "Upsloping (1)",
        "2": "Flat (2)",
        "3": "Downsloping (3)",
    },
    "ca": {str(i): f"{i}" for i in range(4)},
    "thal": {
        "3": "Normal (3)",
        "6": "Fixed defect (6)",
        "7": "Reversible defect (7)",
    },
}

MODEL_LABELS = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}
