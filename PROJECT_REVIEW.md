# Repository review and Marcus's contribution

Reviewed the complete tracked repository at starting commit `9159c2a`, including both notebooks (code, text, saved outputs), all four CSV datasets, feature configuration, all eight EDA figures, README files, dependencies, CI, ignore rules, and editor configuration. The supplied project plan was reviewed as project context; its embedded instructions and links are not separate authorization to edit the team's shared report.

## Your assigned work

The plan assigns **Marcus Reese** regularized linear modeling: **Ridge and Lasso**, hyperparameter tuning, diagnostics, and documented interpretations. Implemented locally in `src/linear_models.py` and `results/marcus_linear_models.ipynb`. Other teammates' model families and their shared evaluation module remain their responsibilities.

## Existing files and findings

| Files | Findings |
| --- | --- |
| `README.md` | Describes the project, team environment, and notebook CI. |
| `requirements.txt` | Pins the full team's packages, including models outside Marcus's scope. A smaller, pinned `requirements-linear.txt` supports this contribution. |
| `bike_eda_pipeline.ipynb` | Contains loading, quality checks, cleaning, eight visualizations, feature construction, temporal splitting, and preprocessing. Its advertised import from `bike_eda_pipeline` fails because no `.py` module exists. It also advertises `evaluate` and `to_counts`, which it does not define. |
| `data/hour.csv`, `data/day.csv`, `data/Readme.txt` | 17,379 hourly observations and 731 daily rows across 2011–2012; daily rental targets agree with aggregated hourly targets. Raw hourly data has no nulls/duplicate rows, but 165 calendar hours are absent. Targets sum exactly. |
| `data/processed/train.csv`, `test.csv` | 15,211 training rows through September 2012 and 2,168 test rows in October–December 2012. Audited against the notebook's transformations. |
| `data/processed/feature_config.json` | Defines 15 linear input columns, three targets, log transformation, and exclusions. The fitted one-hot representation has 67 columns on the full training set. |
| `figures/01`–`08` PNGs | Target skew, weekday/hour commuting patterns, growth, rain and temperature associations, correlations, and suspicious feels-like temperatures. These already support the team's feature choices. |
| `results/sample_model_run.ipynb` | Empty code cell and a test note; no baseline or trained model. |
| `.github/workflows/notebooks-ci.yml` | Lints and executes notebooks on Python 3.12. The new contribution was validated locally on the version recorded in its manifest; remote CI still needs to run after a push. |
| `.gitignore`, `.idea/*` | Ignore rules and editor settings; no modeling logic. |

## Decisions for the linear models

- Use committed processed data and configuration so scores remain comparable to teammates' models.
- Fit Ridge and Lasso separately for casual, registered, and total demand: six tuned models.
- Retain the existing features: scaled weather; one-hot season, weather, and hour/working-day interaction; binary and cyclic predictors. Scaling and encoding are fitted **inside each training fold**.
- Fit `log1p(count)` and return `max(expm1(prediction), 0)` as the team specifies. The inverse is nonlinear, so summed rider-group predictions need not equal the direct-total model.
- Tune with five expanding training windows, never shuffled folds. Choose alpha by mean validation **RMSE in rental counts**, matching the plan's primary metric. Record MAE and RMSLE as secondary metrics.
- Search Ridge alpha from 0.001 to 10,000 (15 log-spaced values); Lasso from 0.00001 to 1 (16 values). Their optimization objectives scale differently, so alpha magnitudes should not be compared across families. Boundary selections are flagged in the run manifest.
- Evaluate fixed selections on the same test hours and compare casual + registered against direct total within each family. Save every candidate's fold scores and timings, selected model coefficients, test predictions, plots, versions, and data hashes.

## Limitations and team coordination

1. The plan's “2011 and 2013” is a typo: the actual data is 2011–2012.
2. The existing EDA explicitly reports using registered **test RMSLE** to choose the hour/working-day interaction. Consequently, this test period has already influenced feature selection; describe it as the team's fixed comparison set rather than a completely untouched estimate. This contribution does not tune alpha on it. A more independent estimate would require a redesigned training-only feature-selection procedure with an unused evaluation period.
3. Humidity zeros are interpolated using future weather readings in the existing preprocessing. This is suitable only when those readings are available for historical cleaning; revisit causal imputation if claiming a strictly prospective forecast. Actual weather predictors also assume weather is available at prediction time. Missing hours remain absent rather than filled with invented rentals.
4. Full one-hot coding plus cyclic and binary terms includes redundant predictors. Regularization handles prediction, but individual coefficients are conditional associations, not causal effects or unique feature importance. In particular, redundant terms can divide the same effect across coefficients; zero Lasso coefficients do not prove irrelevance. Weather is standardized while other columns are not, so absolute coefficients are not universally comparable.
5. Exponentiating a log-space regression does not generally recover the conditional mean of counts. Peak-demand underprediction is a possible consequence, to inspect in the residual plots. No test-derived correction factor is used.
6. Malachi's planned `src/evaluate.py` is absent. The contribution contains its own small metric function and namespaced output files so it runs now; coordinate the eventual shared metric/logger interface.
7. Saved **test** predictions support final comparison only. They are not training data for Ben's stacked ensemble. Proper stacking needs training-period predictions generated without fitting or tuning on the predicted rows, preferably with nested chronological validation.
8. No GitHub changes or shared Overleaf edits have been made. The contribution is local and reviewable.

## Run

From this repository's root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-linear.txt
python -m src.linear_models
python -m pytest -q
```

Alternatively, run `results/marcus_linear_models.ipynb` from top to bottom. It retrains the six models and displays the saved results and figures. The first run may take several minutes. Trained `.joblib` files are regenerated locally and ignored by Git.

## Validation completed locally

- Three focused automated tests passed: temporal/target separation, count inversion/metrics, and training-only scaling.
- All three repository notebooks executed successfully from top to bottom. Original notebooks were checked in a temporary copy to preserve their tracked outputs.
- Both processed splits reproduced the existing EDA transformations within numerical tolerance; daily targets matched hourly aggregations.
- All eight saved score rows independently recomputed from aligned predictions, including both group sums.
- All twelve generated figures were visually inspected. Git whitespace checks passed.
- Local execution used Python 3.14.7 with core package versions matching the team pins; Python 3.12 and remote GitHub Actions have not been verified in this session.
