# Model card — Heart Disease Educational Demo

## Summary

A small educational binary-classification demo using the UCI Heart Disease dataset. The selected artifact is **Logistic Regression on the 11 core fields**. It is designed to teach data cleaning, cohort-aware evaluation and threshold trade-offs—not to assess a person's health.

## Intended use

**Allowed purpose:** classroom demonstrations, portfolio review, and experiments on fictional values.

**Not intended for:** diagnosis, screening, treatment, triage, reassurance, clinical decision support, insurance, employment, or any decision affecting a real person. The app is not a medical device and its model score is not a calibrated probability.

## Data

- Source: UCI Machine Learning Repository, Heart Disease dataset (historical cohorts assembled from Cleveland, Hungary, Switzerland and VA Long Beach).
- 920 rows and 13 candidate clinical inputs plus a cohort label and original outcome; the cohort label is used for grouped validation only.
- Original `num` outcome values are mapped to `0` if `num == 0`, otherwise `1`.
- `ca` and `thal` have substantial missingness (66.4% and 52.8% after cleaning) and were included only in the 13-field experiment. The selected model uses the more complete 11-field set.
- Additional datasets were not mixed in: joining data with different definitions, measurement practices, permissions or populations without review could reduce validity rather than improve it.

## Model and evaluation

- Candidate families: Logistic Regression, Random Forest and XGBoost, compared on 11- and 13-field feature sets.
- Preprocessing is part of each scikit-learn pipeline: median imputation and scaling for numeric fields; most-frequent imputation and one-hot encoding for categories.
- Source cohort is excluded from inputs. GroupKFold by source cohort is used for model selection; an 80/20 stratified patient split is held out for a final software check.
- Selection applies a one-standard-error rule to grouped-CV ROC-AUC, then prefers the 11-field set and the simpler model family; recall is a tie-breaker. The test split is not used for selection.
- Current seed-42 run: selected model grouped-CV ROC-AUC **0.816 ± 0.053**. On the 184-row patient-level test split: accuracy **0.826**, ROC-AUC **0.905**, recall **0.882**, specificity **0.756**, 12 false negatives and 20 false positives at the demonstration cutoff of 0.50.
- This test split contains records from the same historical source cohorts; it is not an external validation study. Results can vary with the split and must not be presented as clinical performance.

## Known limitations and risks

- The data are historical, modest in size and not representative of every current or local population.
- The four cohorts differ in case mix; grouped cross-validation is included but cannot replace new-site external validation.
- No prospective study, external validation, probability calibration, clinical utility analysis, fairness/subgroup audit or clinician review has been completed.
- The original fields include sex and other clinical measurements; the model may reproduce historical biases or correlations.
- A 0.50 cutoff is a software default, not a clinically chosen threshold. Recall/specificity trade-offs are shown to make errors visible.
- Performance metrics are statistical comparisons with source labels, not guarantees about any person.

## Privacy and security

- The app has no account, database, prediction history or input logging feature.
- Form values exist only in the active app session and are not written to disk.
- Public visitors are instructed to use fictional values only. Do not upload patient data to a public demo.

## Requirements before any clinical use

This artifact is **not cleared for clinical use**. A separate regulated product effort would need, at minimum: lawful and consented representative data; independent prospective validation across sites and subgroups; calibration and clinically justified thresholds; review by qualified clinicians and governance bodies; documented intended-use boundaries; privacy/security controls; monitoring, drift management, incident response and a validated human workflow. High scores on this dataset alone are not sufficient.

## Reproduction

From the project directory, install `requirements-dev.txt`, run `python train.py`, and inspect `reports/model_comparison.csv` and `models/model_info.json`. Versions for the checked-in artifact are recorded in `models/model_info.json`.
