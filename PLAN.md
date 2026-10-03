# Build plan — Heart Disease Prediction educational demo

## Decisions locked in
- Build a clean implementation in this folder; use the public GitHub project only as an architecture reference, not as copied code.
- Use the official UCI combined Heart Disease data (920 rows across four source cohorts).
- Compare a **core 11-field** model with a **13-field experiment** that adds `ca` and `thal`. Those two fields have substantial missingness, so we will not assume that adding them improves the model.
- Compare Logistic Regression, Random Forest and XGBoost. Select using grouped cross-validation by source cohort; keep a separate held-out test set for the final report.
- Create a Streamlit demo with no user-input or prediction-history persistence. Prepare instructions for a shareable cloud app URL; actual publication requires the user's hosting account/repository connection.

## Implementation sequence
1. Fetch and normalize the four official UCI cohorts; preserve the source-cohort label for validation but never use it as a model input.
2. Audit missing values and convert the original `num` target to binary `num > 0`.
3. Fit imputers, categorical encoding and numeric scaling only inside model pipelines. Compare both feature sets and all three model families.
4. Report accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC and false-positive/false-negative counts. Select among models with a one-standard-error rule on grouped-CV ROC-AUC (prefer fewer/more complete inputs if scores are indistinguishable); use the untouched test set only for the final check.
5. Build a scrollable Streamlit demo using the user-provided design.md and published VBG foundation: Geist typography, a tool-first editorial layout, semantic evidence tables, light/dark-aware tokens, keyboard focus and no input storage. Keep the project clearly independent and present a **model score**, not a personal risk estimate.
6. Add data/pipeline/UI smoke tests, a README, a model card, and deployment instructions. The first cloud version is public/demo-only, has no application-level input storage, and warns visitors to use fictional values only.

## Out of scope for version one
- Clinical diagnosis, treatment advice, or claims of clinical validity.
- Identifiable patient accounts or cloud prediction-history storage.
- Automatic publication to a third-party cloud account without the user's authorization.
