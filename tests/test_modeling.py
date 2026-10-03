from sklearn.linear_model import LogisticRegression

from src.data import clean_data, feature_frame, load_raw_data, target_and_groups
from src.modeling import build_pipeline


def test_preprocessing_pipeline_handles_missing_numeric_and_categories():
    clean = clean_data(load_raw_data())
    X = feature_frame(clean, "core11")
    y, _ = target_and_groups(clean)
    sample = X.iloc[:250]
    labels = y.iloc[:250]
    pipeline = build_pipeline("core11", LogisticRegression(max_iter=1000, solver="liblinear"))
    pipeline.fit(sample, labels)
    scores = pipeline.predict_proba(X.iloc[250:260])[:, 1]
    assert len(scores) == 10
    assert ((scores >= 0) & (scores <= 1)).all()
