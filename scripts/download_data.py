"""Download and combine the official UCI Heart Disease processed files.

Run from the project root:
    python scripts/download_data.py

The UCI archive contains four cohorts (Cleveland, Hungary, Switzerland and
VA Long Beach). The source/dataset label is retained for grouped evaluation,
but is never supplied to the prediction model.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "heart_disease_uci.csv"
UCI_URL = "https://archive.ics.uci.edu/static/public/45/heart+disease.zip"
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalch", "exang", "oldpeak", "slope", "ca", "thal", "num",
]
COHORT_FILES = {
    "Cleveland": "processed.cleveland.data",
    "Hungary": "processed.hungarian.data",
    "Switzerland": "processed.switzerland.data",
    "VA Long Beach": "processed.va.data",
}


def build_dataset() -> pd.DataFrame:
    with urlopen(UCI_URL, timeout=30) as response:
        archive_bytes = response.read()
    combined: list[pd.DataFrame] = []
    with ZipFile(BytesIO(archive_bytes)) as archive:
        for cohort, filename in COHORT_FILES.items():
            with archive.open(filename) as data_file:
                frame = pd.read_csv(
                    data_file,
                    header=None,
                    names=COLUMNS,
                    na_values=["?"],
                    skipinitialspace=True,
                )
            frame.insert(0, "dataset", cohort)
            combined.append(frame)
    data = pd.concat(combined, ignore_index=True)
    data.insert(0, "id", range(1, len(data) + 1))
    # Match the familiar UCI/Kaggle preview order: id, demographics, cohort,
    # clinical measurements, then original diagnosis.
    ordered = ["id", "age", "sex", "dataset"] + COLUMNS[2:]
    return data[ordered]


def main() -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    data = build_dataset()
    data.to_csv(OUTPUT, index=False)
    print(f"Saved {len(data)} rows × {len(data.columns)} columns to {OUTPUT.relative_to(ROOT)}")
    print("Cohorts:")
    print(data["dataset"].value_counts().to_string())


if __name__ == "__main__":
    main()
