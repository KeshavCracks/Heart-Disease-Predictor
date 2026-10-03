from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def markdown_text(app):
    return "\n".join(element.value for element in app.markdown)


def test_demo_renders_default_output_for_fictional_example_a():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    rendered = markdown_text(app)
    assert "not a personal risk score" in rendered
    assert "Fictional example A" in rendered
    assert "Model score, from 0 to 1" in rendered
    assert "0.027" in rendered  # preset A lands clearly on the negative side
    assert "Negative source-data label" in rendered
    assert "Grouped-CV ROC-AUC, mean ± SD" in rendered
    assert "<caption>" in rendered
    assert "Not affiliated with Vercel" in rendered
    assert "Use fictional values only." in rendered


def test_preset_b_updates_inputs_and_shows_positive_label():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    quick_start = next(w for w in app.selectbox if w.label == "Choose a starting example")

    quick_start.select("Fictional example B").run(timeout=30)

    assert not app.exception
    assert any(w.label == "Age in years" and w.value == 60 for w in app.number_input)
    assert any(w.label == "Cholesterol (mg/dL)" and w.value == 280 for w in app.number_input)
    rendered = markdown_text(app)
    assert "0.944" in rendered
    assert "Positive source-data label" in rendered


def test_custom_edit_is_labelled_custom_and_updates_live():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    age = next(w for w in app.number_input if w.label == "Age in years")

    age.set_value(70).run(timeout=30)

    assert not app.exception
    assert "Custom fictional values" in markdown_text(app)
