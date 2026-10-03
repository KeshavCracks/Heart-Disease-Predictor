from src.data import clean_data, feature_frame, load_raw_data, missing_value_report, target_and_groups


def test_official_dataset_shape_and_source_cohorts():
    raw = load_raw_data()
    assert len(raw) == 920
    assert raw["dataset"].nunique() == 4


def test_clean_data_maps_original_target_and_impossible_zeros():
    raw = load_raw_data()
    clean = clean_data(raw)
    assert set(clean["target"].unique()) == {0, 1}
    assert (clean["target"] == (clean["num"] > 0).astype(int)).all()
    assert not clean["trestbps"].eq(0).any()
    assert not clean["chol"].eq(0).any()


def test_feature_sets_and_group_labels_are_separate():
    clean = clean_data(load_raw_data())
    y, groups = target_and_groups(clean)
    core = feature_frame(clean, "core11")
    full = feature_frame(clean, "all13")
    assert core.shape == (len(clean), 11)
    assert full.shape == (len(clean), 13)
    assert "dataset" not in core.columns
    assert "num" not in full.columns
    assert set(y.unique()) == {0, 1}
    assert groups.nunique() == 4
    assert core["sex"].dropna().isin(["0", "1"]).all()


def test_ca_and_thal_missingness_is_reported():
    clean = clean_data(load_raw_data())
    report = missing_value_report(clean)
    assert report.loc["ca", "missing_percent"] > 50
    assert report.loc["thal", "missing_percent"] > 50
