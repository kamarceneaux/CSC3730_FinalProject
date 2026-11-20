# Bike Rental Demand

CSC 3730 final project. We're using the UCI Bike Sharing dataset to model casual and registered riders separately and compare what drives demand for each group.

## Data

`data/` has the two CSVs from UCI:

- `day.csv` - daily totals (731 rows)
- `hour.csv` - hourly totals (17,379 rows)

`data/Readme.txt` explains the columns. `cnt` is always `casual + registered`.

Source: Fanaee-T, H. & Gama, J. (2013), [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset).

## Setup

Use Python 3.12 if you can, since that's what CI runs.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter lab
```

If you add a library, put it in `requirements.txt` with a pinned version so everyone's environment stays the same.

## Notebooks

Every push and pull request to `main` runs all the notebooks from top to bottom on GitHub Actions. If a cell errors, the check fails. Before you push:

- Restart the kernel and run all cells to make sure it works from a clean start.
- Load data with relative paths (`../data/hour.csv`), not paths on your own computer.
- Don't put spaces in notebook file names.

## Layout

```
data/        raw dataset
results/     saved figures and model outputs
.github/     CI workflow
```

## Marcus: Ridge and Lasso

See [the repository review](PROJECT_REVIEW.md) for findings, methodology, caveats,
and setup. Run `python -m src.linear_models` from the project root, or open
`results/marcus_linear_models.ipynb`. Results, tuning curves, diagnostics, and
aligned predictions are written to `results/linear_models/`.


## Danielle: Decision Tree and Random Forest Models

This section contains the tree-based regression models for predicting casual, registered, and total hourly bike rentals. The notebook compares baseline and hyperparameter-tuned Decision Tree and Random Forest models, evaluates feature importance, and compares direct total-rental predictions with the sum of separately predicted casual and registered rentals.

Open `results/danielle_tree_models.ipynb` to view the complete analysis, model training, tuning results, evaluation metrics, and visualizations. Results and saved model files are available in `results/tree_models/`.

**Methodology:** Models use the team's fixed chronological training and testing split, five-fold expanding time-series cross-validation for hyperparameter tuning, and log-transformed rental targets. Performance is evaluated using RMSE, MAE, and RMSLE on the October–December 2012 test period.

## Kiersten: KNN and SVR Models

This section contains the distance-based regression models for predicting casual, registered, and total hourly bike rentals: K-Nearest Neighbors (KNN) and Support Vector Regression (SVR, RBF kernel). Each model is tuned separately for each target, and the casual and registered predictions are summed and compared with the direct total-rental model.

Open `results/kiersten_distance_models.ipynb` to view the results, tuning curves, team comparison, and interpretation. The notebook only loads saved results, so it runs in seconds. To retrain, run these from the project root:

```
python -m src.distance_models knn
python -m src.distance_models svr
```

Metrics, tuning results, diagnostic plots, and aligned test predictions are written to `results/distance_models/`. `test_predictions.csv` has `knn_*` and `svr_*` columns for each target, plus summed totals, on the same October–December 2012 hours as the other models.

**Methodology:** Models use the team's fixed chronological split, log-transformed targets, and five-fold expanding time-series cross-validation, with features scaled inside each fold. Settings were chosen on validation RMSE only, never on the test set. KNN tried `k` from 1 to 80 with uniform and distance weighting; SVR tried `C` of 0.1, 1, 3, and 10. `C=100` was left out because of run time.

**Results:** SVR had the lowest test RMSE of the saved models for casual (19.0), registered (56.8), and total rentals (63.0 direct, 62.6 summed). KNN was much weaker on registered and total demand (total RMSE 111.9). The small difference between summed and direct totals should not be treated as significant.

## Malachi: Shared evaluation and interpretation

Use `src.evaluate` for shared metrics, count inversion, timestamp alignment,
JSONL results logging, and standard plots. Run `python -m src.evaluation_report`
to compare the committed linear and tree predictions and generate residual and
driver figures. See [EVALUATION.md](EVALUATION.md) for the API and evaluation rules,
and `results/evaluation/INTERPRETATION.md` for measured findings.
