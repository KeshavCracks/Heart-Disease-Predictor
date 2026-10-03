"""Independent, public-facing Streamlit UI for the UCI heart-disease learning demo.

Visual direction (2026-10 refresh)
----------------------------------
White-primary, blue-secondary theme inspired by the UI/UX Pro Max skill and the
React Bits aesthetic (soft aurora glow, faint dot-grid, gentle micro-interactions),
implemented with self-contained page CSS only -- no third-party JavaScript, no
component libraries, no copied code. The Streamlit stack, model behaviour, data,
privacy decisions and safety copy are unchanged.

Composition: masthead -> hero (boundary + identity) -> fictional-example tool with a
live output panel -> how-to-read -> evaluation -> data -> limits -> footer, in one
scrollable reading path.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src import config
from src.data import clean_data, load_raw_data, missing_value_report

st.set_page_config(
    page_title="Heart disease model demo",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

MODEL_PATH = config.MODELS_DIR / "best_model.joblib"
INFO_PATH = config.MODELS_DIR / "model_info.json"
COMPARISON_PATH = config.REPORTS_DIR / "model_comparison.csv"
MISSING_PATH = config.REPORTS_DIR / "missing_values.csv"

# ---------------------------------------------------------------------------
# Self-contained white/blue design system (page-owned; no third-party CSS/JS).
# ---------------------------------------------------------------------------
PAGE_CSS = """
/* tokens */
.st-key-hd-page {
  --bg: #ffffff;
  --bg-soft: #f6f9ff;
  --ink: #0f172a;
  --ink-2: #475569;
  --ink-3: #64748b;
  --line: #e3e9f4;
  --blue: #2563eb;
  --blue-strong: #1d4ed8;
  --blue-soft: #eff6ff;
  --blue-100: #dbeafe;
  --ring: rgba(37, 99, 235, .35);
  --radius: 14px;
  --shadow: 0 1px 2px rgba(15,23,42,.05), 0 8px 24px rgba(15,23,42,.06);

  box-sizing: border-box;
  min-height: 100vh;
  background: var(--bg);
  color: var(--ink);
  color-scheme: light;                 /* keep the page white regardless of OS theme */
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  font-size: 16px;
  line-height: 1.6;
  -webkit-font-smoothing: antialiased;
}
/* white canvas everywhere (Streamlit chrome included) */
html, body, #root, [data-testid="stApp"], section[data-testid="stMain"],
[data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"],
[data-testid="stVerticalBlock"], [data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stElementContainer"] { background: #ffffff; }
[data-testid="stHeader"] { display: none; }        /* drop Streamlit's bar for a clean page */
[data-testid="stMainBlockContainer"] { max-width: 72rem; padding-top: 2rem; }

/* one type voice */
.st-key-hd-page :is(p, span, div, label, button, input, select, li, h1, h2, h3, h4,
  [data-testid="stMarkdownContainer"], [data-testid="stWidgetLabel"], [data-baseweb="select"]) {
  font-family: "Inter", ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
  color: var(--ink);
}
.st-key-hd-page code { font-family: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--blue-strong); }
.st-key-hd-page [data-testid="stCaptionContainer"] { color: var(--ink-3); }

/* focus */
.st-key-hd-page :is(button, input, select, [role="combobox"], a, summary):focus-visible {
  outline: 3px solid var(--ring); outline-offset: 2px; border-radius: 8px;
}

/* ---------------- header / hero ---------------- */
.hd-masthead {
  display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap;
  padding-block: .25rem 1.5rem;
}
.hd-brand { display: inline-flex; align-items: center; gap: .65rem; }
.hd-mark {
  width: 38px; height: 38px; border-radius: 10px; flex: 0 0 auto;
  background: linear-gradient(135deg, var(--blue) 0%, var(--blue-strong) 100%);
  display: inline-flex; align-items: center; justify-content: center;
  box-shadow: 0 6px 16px var(--ring);
}
.hd-brand-name { font-weight: 700; font-size: 1.02rem; letter-spacing: -0.01em; }
.hd-masthead-meta { color: var(--ink-3); font-size: .85rem; display: flex; gap: 1.25rem; flex-wrap: wrap; }

.hd-hero {
  position: relative; overflow: hidden;
  border: 1px solid var(--line); border-radius: 20px;
  background:
    radial-gradient(560px 240px at 12% -10%, rgba(37,99,235,.10), transparent 60%),
    radial-gradient(520px 260px at 90% 0%, rgba(37,99,235,.07), transparent 60%),
    radial-gradient(rgba(37,99,235,.10) 1px, transparent 1.5px) 0 0 / 22px 22px,
    var(--bg);
  padding: clamp(1.75rem, 4vw, 3rem);
  margin-bottom: 1.5rem;
}
.hd-pill {
  display: inline-flex; align-items: center; gap: .5rem;
  background: var(--blue-soft); color: var(--blue-strong);
  border: 1px solid var(--blue-100); border-radius: 999px;
  font-size: .8rem; font-weight: 600; padding: .3rem .8rem; margin-bottom: 1.1rem;
}
.hd-pill .hd-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--blue); }
.hd-title { font-size: clamp(1.9rem, 4.2vw, 2.9rem); line-height: 1.12; font-weight: 800; letter-spacing: -0.02em; margin: 0 0 .9rem; }
.hd-title .hd-accent { color: var(--blue); }
.hd-lede { color: var(--ink-2); font-size: 1.06rem; max-width: 62ch; margin: 0; }
.hd-hero-note { margin-top: 1.2rem; color: var(--ink-3); font-size: .9rem; max-width: 70ch; }
.hd-hero-note strong { color: var(--ink); }
.hd-hero-grid { display: grid; grid-template-columns: 1.5fr .8fr; gap: 2.25rem; align-items: center; }
@media (max-width: 900px) { .hd-hero-grid { grid-template-columns: 1fr; } }
.hd-hero-stats { display: flex; flex-direction: column; gap: .8rem; }
.hd-stat { background: rgba(255,255,255,.8); border: 1px solid var(--line); border-radius: 12px; padding: .75rem 1.05rem; }
.hd-stat b { display: block; font-size: 1.55rem; font-weight: 800; color: var(--blue-strong); font-variant-numeric: tabular-nums; line-height: 1.2; }
.hd-stat span { color: var(--ink-3); font-size: .83rem; }

/* ---------------- sections ---------------- */
.hd-section { margin: 3rem 0; }
.hd-overline { color: var(--blue); font-size: .78rem; font-weight: 700; letter-spacing: .09em; text-transform: uppercase; margin: 0 0 .4rem; }
.hd-h2 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; margin: 0 0 .5rem; }
.hd-h3 { font-size: 1.12rem; font-weight: 600; margin: 1.4rem 0 .5rem; }
.hd-p { color: var(--ink-2); max-width: 72ch; margin: 0 0 .8rem; }

/* ---------------- tool card ---------------- */
.hd-card {
  background: var(--bg); border: 1px solid var(--line); border-radius: var(--radius);
  box-shadow: var(--shadow); padding: clamp(1.1rem, 2.6vw, 1.8rem);
}
.hd-tool { display: grid; grid-template-columns: 1.4fr .9fr; gap: 1.75rem; }
@media (max-width: 900px) { .hd-tool { grid-template-columns: 1fr; } }

.hd-output {
  background: var(--blue-soft); border: 1px solid var(--blue-100); border-radius: var(--radius);
  padding: 1.4rem; align-self: start; position: sticky; top: 1rem;
}
.hd-output-label { color: var(--ink-2); font-size: .85rem; font-weight: 600; margin: 0 0 .35rem; }
.hd-score { font-size: 2.6rem; font-weight: 800; color: var(--blue-strong); letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.1; margin: 0 0 .5rem; }
.hd-chip {
  display: inline-flex; align-items: center; gap: .45rem; border-radius: 999px;
  font-size: .85rem; font-weight: 600; padding: .32rem .85rem; margin: .2rem 0 .8rem;
}
.hd-chip.pos { background: var(--blue); color: #fff; }
.hd-chip.neg { background: #fff; color: var(--blue-strong); border: 1px solid var(--blue-100); }
.hd-output-detail { color: var(--ink-2); font-size: .88rem; margin: .3rem 0; }
.hd-output-meta { color: var(--ink-3); font-size: .8rem; margin-top: .8rem; }

/* native widgets, white/blue */
.st-key-hd-page [data-testid="stNumberInput"] input,
.st-key-hd-page [data-testid="stSelectbox"] [role="group"],
.st-key-hd-page [data-baseweb="select"] > div {
  background: #fff !important; color: var(--ink);
  border: 1px solid var(--line) !important; border-radius: 10px; min-height: 2.7rem;
  transition: border-color .15s ease, box-shadow .15s ease;
}
.st-key-hd-page [data-testid="stNumberInput"] input:hover,
.st-key-hd-page [data-testid="stSelectbox"] [role="group"]:hover,
.st-key-hd-page [data-baseweb="select"] > div:hover { border-color: #c7d4ee; }
.st-key-hd-page [data-testid="stNumberInput"] input:focus,
.st-key-hd-page [data-testid="stSelectbox"] [role="group"]:focus-within,
.st-key-hd-page [data-baseweb="select"] > div:focus-within { border-color: var(--blue); box-shadow: 0 0 0 3px var(--ring); }
.st-key-hd-page [data-testid="stSelectbox"] [role="listbox"],
.st-key-hd-page [data-testid="stSelectbox"] ul { background: #fff; border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }
.st-key-hd-page [data-testid="stNumberInputStepUp"], .st-key-hd-page [data-testid="stNumberInputStepDown"] { display: none; }
.st-key-hd-page [data-testid="stNumberInput"] input { padding-inline: .9rem; }
.st-key-hd-page [data-testid="stWidgetLabel"] label { font-weight: 600; font-size: .9rem; color: var(--ink); }
.st-key-hd-page [data-baseweb="select"] [role="listbox"] { background: #fff; border: 1px solid var(--line); border-radius: 10px; box-shadow: var(--shadow); }

/* ---------------- tables ---------------- */
.hd-table-wrap { border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); background: #fff; }
.hd-table-wrap table { width: 100%; border-collapse: collapse; font-size: .94rem; }
.hd-table-wrap caption { text-align: left; color: var(--ink-3); font-size: .85rem; padding: .8rem 1rem 0; caption-side: top; }
.hd-table-wrap th, .hd-table-wrap td { padding: .68rem 1rem; border-bottom: 1px solid var(--line); }
.hd-table-wrap thead th { background: var(--bg-soft); color: var(--ink-2); font-weight: 600; font-size: .85rem; }
.hd-table-wrap tbody tr:last-child td, .hd-table-wrap tbody tr:last-child th { border-bottom: 0; }
.hd-table-wrap tbody tr:hover { background: #f8fafc; }
.hd-table-wrap th[scope="row"] { text-align: left; font-weight: 500; color: var(--ink); }
.hd-table-wrap .num, .hd-table-wrap th.num { text-align: right; font-variant-numeric: tabular-nums; }
.hd-table-wrap tr.sel td, .hd-table-wrap tr.sel th { background: var(--blue-soft); }
@media (max-width: 640px) { .hd-table-wrap { overflow-x: auto; } }

/* buttons (download) */
.st-key-hd-page [data-testid="stDownloadButton"] button {
  background: #fff; color: var(--blue-strong); border: 1px solid var(--blue-100);
  border-radius: 10px; font-weight: 600; min-height: 2.6rem; transition: all .15s ease;
}
.st-key-hd-page [data-testid="stDownloadButton"] button:hover { background: var(--blue-soft); border-color: var(--blue); }

/* expanders */
.st-key-hd-page [data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 12px; background: #fff; }
.st-key-hd-page [data-testid="stExpander"] summary { font-weight: 600; color: var(--ink); }

/* footer */
.hd-footer { border-top: 1px solid var(--line); margin-top: 3.5rem; padding: 1.4rem 0 2.2rem; color: var(--ink-3); font-size: .88rem; display: flex; gap: .5rem; flex-wrap: wrap; }

/* plain-language glossary */
.hd-glossary { display: grid; grid-template-columns: 1fr 1fr; gap: .9rem 2.2rem; margin-top: 1rem; }
@media (max-width: 800px) { .hd-glossary { grid-template-columns: 1fr; } }
.hd-glossary > div { border-left: 3px solid var(--blue-100); padding: .1rem 0 .1rem .9rem; }
.hd-glossary dt { font-weight: 600; color: var(--ink); font-size: .96rem; }
.hd-glossary dd { margin: .1rem 0 0; color: var(--ink-2); font-size: .9rem; }

/* motion (respects reduced motion) */
@keyframes hd-rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: none; } }
.hd-hero, .hd-card { animation: hd-rise .5s ease both; }
.hd-card { animation-delay: .08s; }
@media (prefers-reduced-motion: reduce) {
  .st-key-hd-page *, .st-key-hd-page *::before, .st-key-hd-page *::after {
    animation: none !important; transition: none !important; scroll-behavior: auto !important;
  }
}
"""

FONT_LINKS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&amp;family=JetBrains+Mono:wght@400;600&amp;display=swap" rel="stylesheet" referrerpolicy="no-referrer">
"""

BRAND_SVG = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" aria-hidden="true">'
    '<path d="M3 12h4l2.5-6 4 12 2.5-6H21" stroke="#fff" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round"/></svg>'
)


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


@st.cache_data
def load_public_data_summary():
    clean = clean_data(load_raw_data())
    counts = clean[config.TARGET].value_counts().sort_index()
    cohorts = clean[config.SOURCE_COLUMN].value_counts().sort_values(ascending=False)
    missing = missing_value_report(clean[config.FEATURE_SETS["all13"]])
    return clean, counts, cohorts, missing


def load_artifacts():
    if not MODEL_PATH.exists() or not INFO_PATH.exists():
        st.error("The trained model is missing. From the project folder, run `python train.py`.")
        st.stop()
    try:
        model = load_model(str(MODEL_PATH))
        info = json.loads(INFO_PATH.read_text(encoding="utf-8"))
    except Exception:
        st.error("The demo model could not be loaded. The project maintainer should retrain it with the pinned dependencies, then restart the app.")
        st.stop()
    return model, info


# fictional presets: one clearly negative, one clearly positive
PRESET_A = {"age": 45, "sex": "0", "cp": "2", "trestbps": 120, "chol": 200,
            "fbs": "0", "restecg": "0", "thalch": 170, "exang": "0", "oldpeak": 0.2, "slope": "1"}
PRESET_B = {"age": 60, "sex": "1", "cp": "4", "trestbps": 150, "chol": 280,
            "fbs": "1", "restecg": "1", "thalch": 120, "exang": "1", "oldpeak": 2.5, "slope": "3"}
PRESETS = {"Fictional example A": PRESET_A, "Fictional example B": PRESET_B}
CUSTOM_PRESET = "Enter custom fictional values"
PRESET_KEY = "demo_preset"

NUMERIC_INPUTS = {"age": (28, 77, 45, 1), "trestbps": (80, 200, 120, 1), "chol": (85, 603, 200, 1),
                  "thalch": (60, 202, 170, 1), "oldpeak": (-2.6, 6.2, 0.2, 0.1)}
DISPLAY_LABELS = {
    "age": "Age in years", "sex": "Sex as recorded in the dataset", "cp": "Chest-pain category",
    "trestbps": "Resting blood pressure (mm Hg)", "chol": "Cholesterol (mg/dL)",
    "fbs": "Fasting blood sugar above 120 mg/dL?", "restecg": "Resting ECG category",
    "thalch": "Maximum heart rate in exercise test", "exang": "Chest pain during exercise?",
    "oldpeak": "ST-segment change (oldpeak)", "slope": "Exercise-test ST slope",
}
CATEGORY_LABELS = {
    "sex": {"0": "Female", "1": "Male"},
    "cp": {"1": "Typical angina", "2": "Atypical angina", "3": "Non-anginal pain", "4": "No chest pain reported"},
    "fbs": {"0": "No", "1": "Yes"},
    "restecg": {"0": "Normal", "1": "ST-T wave change", "2": "Possible LV hypertrophy"},
    "exang": {"0": "No", "1": "Yes"},
    "slope": {"1": "Rising", "2": "Flat", "3": "Falling"},
}
FIELD_HELP = {
    "age": "The source dataset includes ages from 28 to 77. Use a fictional value.",
    "sex": "Codes follow the original dataset: female or male. This field can affect outputs and may encode historical dataset bias.",
    "cp": "The source grouped chest pain into four categories. The wording is from the dataset, not a self-assessment guide.",
    "trestbps": "A recorded resting blood-pressure value in the research table, not a target or advice range.",
    "chol": "A recorded cholesterol value in the research table, not a target or advice range.",
    "fbs": "The source dataset uses a yes/no flag for fasting blood sugar above 120 mg/dL.",
    "restecg": "A category recorded from resting ECG in the historical dataset. This app does not interpret ECGs.",
    "thalch": "Maximum heart rate recorded during an exercise test in the source table.",
    "exang": "Whether exercise-related chest pain was recorded in the source dataset.",
    "oldpeak": "An ST-segment measurement in the source table. It requires clinical context and is not interpreted here.",
    "slope": "A categorical slope from the exercise-test data. Labels are simplified for readability.",
}
FORM_GROUPS = [
    ("About the sample", ["age", "sex", "cp"]),
    ("Measurements", ["trestbps", "chol", "fbs"]),
    ("Test results", ["restecg", "thalch", "exang", "oldpeak", "slope"]),
]

# Plain-language, jargon-free explanations of the historical study fields.
GLOSSARY = {
    "age": "How old the person was, in years.",
    "sex": "Recorded as male or female in the old study.",
    "cp": "The kind of chest discomfort that was noted. The study used four buckets, from 'typical angina' to 'no chest pain'.",
    "trestbps": "Blood pressure while resting, in millimetres of mercury (mm Hg).",
    "chol": "The amount of cholesterol in the blood, in mg/dL.",
    "fbs": "Whether blood sugar after not eating for a while was above 120 mg/dL.",
    "restecg": "A tracing of the heart's electrical activity at rest (an ECG), grouped into three simple types.",
    "thalch": "The fastest the heart beat during an exercise test.",
    "exang": "Whether chest pain appeared during exercise.",
    "oldpeak": "How far a part of the exercise ECG tracing dipped (called ST depression).",
    "slope": "The shape of that ECG segment during exercise: rising, flat, or falling.",
}

# One-line, everyday meanings for the evaluation numbers.
PLAIN_METRICS = {
    "Accuracy": "How often the model's guess matched the study's label, overall.",
    "Sensitivity / recall": "Of the people the study called positive, how often the model caught them.",
    "Specificity": "Of the people the study called negative, how often the model also said negative.",
    "Precision": "When the model said positive, how often the study agreed.",
    "F1": "One number that balances 'catching positives' with 'being right when it says positive'.",
    "ROC-AUC": "How well the score separates positives from negatives, no matter which cutoff you pick.",
}


def apply_selected_preset() -> None:
    values = PRESETS.get(st.session_state.get(PRESET_KEY))
    if values:
        for feature, value in values.items():
            st.session_state[f"demo_{feature}"] = value


def add_feature_input(feature: str, container) -> object:
    label = DISPLAY_LABELS[feature]
    key = f"demo_{feature}"
    if feature in config.NUMERIC_FEATURES:
        low, high, fallback, step = NUMERIC_INPUTS[feature]
        st.session_state.setdefault(key, fallback)
        if feature == "oldpeak":
            return container.number_input(label, min_value=float(low), max_value=float(high),
                                          step=float(step), format="%.1f", key=key, help=FIELD_HELP[feature])
        return container.number_input(label, min_value=int(low), max_value=int(high),
                                      step=int(step), format="%d", key=key, help=FIELD_HELP[feature])
    options = CATEGORY_LABELS[feature]
    st.session_state.setdefault(key, next(iter(options)))
    return container.selectbox(label, options=list(options), format_func=lambda c: options[c],
                               key=key, help=FIELD_HELP[feature])


def predict_row(model, features, values, threshold):
    row = pd.DataFrame([{f: values[f] for f in features}], columns=features)
    score = float(model.predict_proba(row)[0, 1])
    return {"score": score, "predicted_class": int(score >= threshold)}


def html_table(caption, headers, rows, *, numeric_columns=None, row_header_column=0, selected_row=None) -> str:
    numeric_columns = numeric_columns or set()
    p = ['<div class="hd-table-wrap"><table>', f'<caption>{html.escape(caption)}</caption>', "<thead><tr>"]
    for i, h in enumerate(headers):
        cls = ' class="num"' if i in numeric_columns else ""
        p.append(f"<th scope=\"col\"{cls}>{html.escape(h)}</th>")
    p.append("</tr></thead><tbody>")
    for r_i, row in enumerate(rows):
        sel = ' class="sel"' if selected_row is not None and r_i == selected_row else ""
        p.append(f"<tr{sel}>")
        for i, value in enumerate(row):
            cell = html.escape(str(value))
            if i == row_header_column:
                p.append(f'<th scope="row">{cell}</th>')
            else:
                cls = ' class="num"' if i in numeric_columns else ""
                p.append(f"<td{cls}>{cell}</td>")
        p.append("</tr>")
    p.append("</tbody></table></div>")
    return "".join(p)


def main() -> None:
    st.markdown(f"<style>{PAGE_CSS}</style>", unsafe_allow_html=True)
    st.markdown(FONT_LINKS, unsafe_allow_html=True)
    model, info = load_artifacts()
    clean, target_counts, cohorts, missing = load_public_data_summary()
    features = info["features"]
    threshold = float(info["decision_threshold"])

    with st.container(key="hd-page"):
        st.markdown(
            f"""
            <a class="hd-skip" href="#hd-tool" style="position:absolute;left:-9999px;">Skip to the example tool</a>
            <div class="hd-masthead">
              <span class="hd-brand"><span class="hd-mark">{BRAND_SVG}</span><span class="hd-brand-name">Heart disease model demo</span></span>
              <span class="hd-masthead-meta"><span>UCI Heart Disease dataset</span><span>Independent educational project</span></span>
            </div>
            <div class="hd-hero">
              <div class="hd-hero-grid">
                <div>
                  <span class="hd-pill"><span class="hd-dot"></span>Educational demo · fictional values only</span>
                  <h1 class="hd-title">Explore a historical model,<br><span class="hd-accent">not a personal risk score</span></h1>
                  <p class="hd-lede">Try fictional values against a model trained on 920 historical records from four source cohorts. Its output is a dataset label, not a health assessment.</p>
                  <p class="hd-hero-note"><strong>Use fictional values only.</strong> The app does not save form entries or prediction history. Selected model: {html.escape(info['best_model_label'])}. The score is not a calibrated probability for a person.</p>
                </div>
                <div class="hd-hero-stats" aria-label="Dataset at a glance">
                  <div class="hd-stat"><b>920</b><span>historical records</span></div>
                  <div class="hd-stat"><b>4</b><span>source cohorts</span></div>
                  <div class="hd-stat"><b>{len(features)}</b><span>input fields in the selected model</span></div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- tool ----
        st.session_state.setdefault(PRESET_KEY, next(iter(PRESETS)))
        for feature in features:
            st.session_state.setdefault(f"demo_{feature}", PRESET_A[feature])

        st.markdown('<p class="hd-overline">Try it</p><h2 class="hd-h2" id="hd-tool">Try a fictional example</h2>'
                    '<p class="hd-p">Inputs are limited to values observed in the source data. Those bounds are dataset limits, not medical reference ranges. The output updates live as you change values.</p>',
                    unsafe_allow_html=True)
        st.selectbox("Choose a starting example", options=[*PRESETS.keys(), CUSTOM_PRESET],
                     key=PRESET_KEY, on_change=apply_selected_preset,
                     help="Both examples are fictional. Selecting one fills the form and updates the output.")

        tool_left, tool_right = st.columns([1.4, 0.9], gap="large")
        values = {}
        with tool_left:
            for group_name, group_features in FORM_GROUPS:
                st.markdown(f'<h3 class="hd-h3">{html.escape(group_name)}</h3>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, feature in enumerate(group_features):
                    values[feature] = add_feature_input(feature, cols[i % 2])

        result = predict_row(model, features, values, threshold)
        selected_preset = st.session_state.get(PRESET_KEY)
        preset_values = PRESETS.get(selected_preset)
        sample_label = selected_preset if (preset_values and all(values[f] == preset_values[f] for f in features)) else "Custom fictional values"
        score = float(result["score"]); predicted = int(result["predicted_class"])
        chip_cls = "pos" if predicted else "neg"
        chip_text = "Positive source-data label" if predicted else "Negative source-data label"
        with tool_right:
            st.markdown(
                f"""
                <div class="hd-output" role="status" aria-live="polite">
                  <p class="hd-output-label">Model score, from 0 to 1</p>
                  <p class="hd-score">{score:.3f}</p>
                  <span class="hd-chip {chip_cls}">{chip_text}</span>
                  <p class="hd-output-detail">Uncalibrated model output. It is not a percentage chance or an individual risk estimate.</p>
                  <p class="hd-output-detail">Class at the software cutoff of {threshold:.2f}. Positive means the original table recorded <code>num &gt; 0</code>; this is not a diagnosis.</p>
                  <p class="hd-output-meta">{html.escape(sample_label)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        # ---- how to read ----
        st.markdown(
            """
            <div class="hd-section">
              <p class="hd-overline">Interpretation</p>
              <h2 class="hd-h2">How to read the output</h2>
              <p class="hd-p">Think of the score as a similarity dial: close to 1 means your fictional row looks like the positive examples in the old research table; close to 0 means it looks like the negative ones. It is not a person's chance of having a condition.</p>
              <p class="hd-p">The model simply compares the numbers you enter with patterns it saw in the historical data, then says which side of the 0.50 line the row falls on.</p>
              <p class="hd-p">Chest-pain and ECG-related fields are categories from the source study that need clinical context. The app does not interpret symptoms, ECGs, or exercise tests.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- evaluation ----
        test = info["held_out_test_metrics"]
        comparison = pd.read_csv(COMPARISON_PATH)
        rows, selected_row = [], None
        for i, item in comparison.iterrows():
            n = "11 fields" if item["feature_set"] == "core11" else "13 fields"
            name = f"{item['model_label']}, {n}"
            if item["configuration"] == info["best_configuration"]:
                name += " — selected"; selected_row = i
            rows.append([name, f"{item['cv_roc_auc']:.3f} ± {item['cv_roc_auc_std']:.3f}"])
        st.markdown(
            """
            <div class="hd-section">
              <p class="hd-overline">Evaluation</p>
              <h2 class="hd-h2">How the candidates compared</h2>
              <p class="hd-p">Six combinations were tested: three model families on 11 or 13 fields. Selection used a one-standard-error rule on grouped cross-validation by source cohort; the patient-level test split did not choose the model.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(html_table("Grouped cross-validation ROC-AUC across source-cohort folds. Values show mean ± standard deviation.",
                               ["Candidate", "Grouped-CV ROC-AUC, mean ± SD"], rows,
                               numeric_columns={1}, selected_row=selected_row), unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="hd-section">
              <h3 class="hd-h3">Selected model on the held-out examples</h3>
              <p class="hd-p">The patient-level test split contained {int(info['test_rows'])} records from the same four historical cohorts. It is a software check, not external validation.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(html_table(
            "Held-out test metrics at a 0.50 cutoff, except ROC-AUC which summarizes ranking over thresholds.",
            ["Measure", "Value", "What it means, in plain words"],
            [["Accuracy", f"{float(test['accuracy']):.3f}", PLAIN_METRICS["Accuracy"]],
             ["Sensitivity / recall", f"{float(test['recall']):.3f}", PLAIN_METRICS["Sensitivity / recall"]],
             ["Specificity", f"{float(test['specificity']):.3f}", PLAIN_METRICS["Specificity"]],
             ["Precision", f"{float(test['precision']):.3f}", PLAIN_METRICS["Precision"]],
             ["F1", f"{float(test['f1']):.3f}", PLAIN_METRICS["F1"]],
             ["ROC-AUC", f"{float(test['roc_auc']):.3f}", PLAIN_METRICS["ROC-AUC"]]],
            numeric_columns={1}), unsafe_allow_html=True)
        with st.expander("Show the selected model's confusion counts"):
            st.markdown(html_table("Counts against the source-data labels in the 184-record held-out split.",
                                   ["Test outcome", "Count"],
                                   [["True negative", int(test["true_negatives"])], ["False positive", int(test["false_positives"])],
                                    ["False negative", int(test["false_negatives"])], ["True positive", int(test["true_positives"])]],
                                   numeric_columns={1}), unsafe_allow_html=True)
            st.caption("These are counts from one split of a small historical dataset, not expected error rates for a person or a current population.")
        if COMPARISON_PATH.exists():
            st.download_button("Download the full candidate metrics CSV", data=COMPARISON_PATH.read_bytes(),
                               file_name="heart-model-comparison.csv", mime="text/csv",
                               help="The file includes grouped-CV and held-out test metrics. Test metrics were not used to select the model.")

        # ---- data ----
        ca_missing = float(info["extra_feature_missing_percent"]["ca"])
        thal_missing = float(info["extra_feature_missing_percent"]["thal"])
        st.markdown(
            """
            <div class="hd-section">
              <p class="hd-overline">Data</p>
              <h2 class="hd-h2">What is in the source data?</h2>
              <p class="hd-p">The demo combines four historical UCI cohorts. The original <code>num</code> field is mapped to a binary research label: zero stays negative; any value above zero becomes positive. Cohort names are used for grouped validation and never as model inputs.</p>
              <p class="hd-p">The experiment compared 11 fields with a 13-field option that adds <code>ca</code> and <code>thal</code>. More rows were not added from other sources because differing definitions, populations, permissions, or measurement practices need review before combining data.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(html_table(
            f"The combined dataset contains {len(clean)} records: {int(target_counts.get(0,0))} negative and {int(target_counts.get(1,0))} positive.",
            ["Source cohort", "Records"], [[str(n), int(c)] for n, c in cohorts.items()], numeric_columns={1}),
            unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="hd-section">
              <h3 class="hd-h3">Why the selected model uses 11 fields</h3>
              <p class="hd-p"><code>ca</code> is missing in {ca_missing:.1f}% of rows and <code>thal</code> in {thal_missing:.1f}%. The 11-field model was chosen by the documented cross-validation rule; this is a data-quality decision, not evidence of clinical superiority.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("View the full missing-value audit"):
            mrows = [[str(f), int(r["missing_count"]), f"{float(r['missing_percent']):.1f}%", int(r["observed_count"])]
                     for f, r in missing.sort_values("missing_percent", ascending=False).iterrows()]
            st.markdown(html_table("Missing and observed values across all 13 candidate input fields.",
                                   ["Field", "Missing values", "Missing", "Observed values"], mrows,
                                   numeric_columns={1, 2, 3}), unsafe_allow_html=True)

        # ---- plain-language glossary ----
        gloss_items = "".join(
            f"<div><dt>{html.escape(DISPLAY_LABELS[f].rstrip('?'))}</dt><dd>{html.escape(GLOSSARY[f])}</dd></div>"
            for _, feats in FORM_GROUPS for f in feats
        )
        st.markdown(
            f"""
            <div class="hd-section">
              <p class="hd-overline">In everyday words</p>
              <h2 class="hd-h2">What the words mean</h2>
              <p class="hd-p">The form uses terms from a historical study. Here is what each one means in plain language, so you do not need any medical background to explore the demo.</p>
              <dl class="hd-glossary">{gloss_items}</dl>
              <p class="hd-p" style="margin-top:1.1rem;">These are simple explanations of fields recorded in an old research table. They are not medical advice, and you should not measure or interpret your own health from them.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ---- limits ----
        st.markdown(
            """
            <div class="hd-section">
              <p class="hd-overline">Limits</p>
              <h2 class="hd-h2">What the results do not establish</h2>
              <ul style="color:var(--ink-2); max-width:72ch; line-height:1.7;">
                <li>The dataset is historical, modest in size, and not representative of every current or local population.</li>
                <li>Grouped validation uses four source cohorts; it cannot replace independent external validation.</li>
                <li>No prospective study, calibration study, subgroup fairness audit, or clinical review has been completed.</li>
                <li>The 0.50 cutoff is a software demo setting, not a clinically chosen threshold.</li>
                <li>Do not use the output for diagnosis, screening, treatment, or reassurance. Do not enter real health or identifying information.</li>
              </ul>
              <p class="hd-p">Data source: UCI Machine Learning Repository, <a href="https://archive.ics.uci.edu/dataset/45/heart+disease" rel="noreferrer" style="color:var(--blue-strong);">Heart Disease dataset</a>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="hd-footer"><span>Independent educational project.</span><span>Not affiliated with Vercel.</span><span>Not for clinical use.</span></div>',
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
