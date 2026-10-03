# Shareable demo deployment

This project is prepared for a public **educational demo URL**, but it is not published automatically. Publishing requires a hosting account and a GitHub repository that you control. “Production-ready” here means a tested, documented educational demo—not a clinically validated product.

## Safety and privacy boundary

- The app has no database, accounts, prediction-history feature or application-level input persistence. Submitted values are used in memory for the active session.
- Hosting infrastructure may retain operational logs under its own policies. Do not enter real patient, health or identifying information; use the fictional presets only.
- No secrets are required for this demo. Never commit API keys, access tokens or private data.
- This model is not for diagnosis, screening, treatment or care decisions. See [MODEL_CARD.md](MODEL_CARD.md) before sharing.

## Design asset and hosting notes

The app includes the byte-identical published stylesheet at `assets/vercel-brand.css`; the page identifies itself as independent and does not use Vercel logos. Geist web fonts are requested from Google Fonts, with a system-font fallback when that request is blocked.

## Pre-publish checklist

1. Run `python -m pip install -r requirements-dev.txt` and `pytest -q` locally.
2. Confirm `models/best_model.joblib`, `models/model_info.json`, the dataset and required report assets are present.
3. Run `streamlit run app.py`; try both fictional presets and the custom-value flow, then read the evaluation and data sections below the tool. Confirm the first example output renders automatically.
4. Check the app at a narrow/mobile viewport and with keyboard-only navigation. Keep the educational warning visible.
5. Review the host's privacy, logging, resource and access policies. Share only if they fit the intended public-demo use.

## Typical Streamlit-compatible hosting flow

1. Push this project to a GitHub repository you control. Include the fitted model artifact and `requirements.txt` so the app loads without running a lengthy training job at startup. The checked-in artifact is Logistic Regression; XGBoost is needed only if a later training run selects an XGBoost model.
2. In your chosen Streamlit-compatible host, create an app from that repository, select `app.py` as the entry point and use the project root as the app directory.
3. Let the host install `requirements.txt`, then open the generated URL and repeat the pre-publish checks.
4. If you retrain or replace the artifact, update the model card and metrics, run the full tests, and verify the hosted artifact and dependency versions together before publishing.

Cloud platform names, account requirements and deployment settings change; follow the host's current instructions. A future real-world health product would require a distinct, legally and clinically governed validation, privacy, security and monitoring program. This demo is not a starting authorization for clinical use.

## Vercel (static React frontend, no backend)

A self-contained React frontend lives in `web/`. It runs the same Logistic Regression
entirely in the browser from an exported model (`web/src/model.json`), so it needs no
server, no database, and never transmits visitor inputs. This is the recommended path
for a public Vercel URL.

1. Push this repository to GitHub.
2. In Vercel, import the repo and set the **Root Directory** to `web` (Vite is
   auto-detected; build `npm run build`, output `dist`).
   Alternatively, run `npm run build` locally and drag the `web/dist` folder into
   Vercel for an instant static deploy.
3. No environment variables or secrets are required.

The Streamlit app in this repo (`app.py`) remains available for local, in-person demos;
it is not deployable to Vercel because Vercel is serverless and Streamlit needs a
long-running process.
