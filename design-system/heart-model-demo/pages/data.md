# Data and limitations section

- Identify the UCI dataset, 920 records, and four source cohorts. Explain `num > 0` as a positive label in the original table, not a diagnosis.
- Compare 11 and 13 fields empirically. Explain that `ca` and `thal` are frequently missing and that the selected 11-field choice is a data-quality/model-selection decision, not evidence of clinical superiority.
- Keep cohort counts visible and place the full missing-value audit in a disclosure with a semantic table.
- Explain that four cohort folds and one patient-level split are not external validation. Note the absent prospective, calibration, fairness, and clinical reviews.
- Do not add data from other sources without checking definitions, permissions, populations, and measurement practices.
- End with the official UCI citation and the public-demo privacy caveat. Visitors must use fictional values only.
