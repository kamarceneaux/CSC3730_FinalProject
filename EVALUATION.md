# Evaluation and interpretation — Malachi

The shared API is `src/evaluate.py`. It scores original rental counts, validates
inputs, aligns saved predictions by datetime, logs results, summarizes residuals,
and produces standard diagnostic and rider-group driver plots.

## Team integration

```python
from src.evaluate import metrics, to_counts, log_result, plot_diagnostics

# Only invert if your model returns log1p predictions.
# TransformedTargetRegressor.predict already returns counts: do not invert twice.
predicted = model.predict(X_test)
scores = metrics(y_test, predicted)  # RMSE, MAE, RMSLE
log_result("results/my_model/evaluation.jsonl", model="My model",
           target="casual", actual=y_test, predicted=predicted,
           run_id="experiment-001", metadata={"target_scale": "counts"})
fig = plot_diagnostics(test.datetime, y_test, predicted, "My model: casual")
fig.savefig("results/my_model/residuals.png", dpi=150)
```

Close figures with `matplotlib.pyplot.close(fig)` when done. Each worker should
write its own log file. The logger appends JSONL and retains repeated run IDs;
include hyperparameters, data hashes, and the split description in metadata.
`linear_models.py` now imports the common metrics and inverse transformation,
retaining its existing public imports and score column names.

For multi-model evaluation, save a CSV with one unique `datetime` and count-scale
predictions per hour. `align_predictions(truth, predictions)` sorts timestamps
and rejects extra/missing hours, duplicates, and conflicting actual targets.
It does not silently drop inconvenient observations. Pass aligned vectors to
`compare_totals(actual_total, casual_prediction, registered_prediction, direct_prediction)`.
Sum the two predictions after inverse transformation. Never sum logarithms.
The comparison requires models evaluated on identical timestamps and targets.

## Reproduce the consolidated evaluation

From the project root, using the team's installed dependencies:

```bash
python -m src.evaluation_report
python -m pytest -q
```

The report reads existing committed predictions and importance CSVs; it does not
retrain models or change tuning. Outputs go to `results/evaluation/`:

- `metrics.csv`: 24 rows across six model variants and four targets/approaches.
- `total_comparison.csv`: direct versus summed RMSE/MAE and group-error correlation.
- `residual_strata.csv`: error by hour, working-day status, and weather, with sample counts.
- `results.jsonl`: shared result records, rebuilt for this report run.
- Residual panels, hourly bias, total comparison, and two driver comparison figures.
- `manifest.json`: input hashes and local runtime versions.

## Evaluation rules and interpretation

Use January 2011–September 2012 for training and October–December 2012 for the
fixed comparison. Tune only within training using chronological validation. RMSE
is the primary criterion; MAE is complementary and RMSLE is secondary. Residuals
are **actual minus predicted**: positive bias means underprediction. NRMSE and
NMAE divide errors by that group's mean observed count; they are undefined when
that mean is zero. These help compare demand scales but do not establish intrinsic
predictability. Avoid MAPE because zero-demand hours are valid.

Driver figures reproduce the tuned trees' impurity-based importance on log targets.
They describe how these fitted models use features, not causal effects or effect
direction. Correlated hour/cyclic features can split importance, and impurity
importance favors predictors with many split points. Do not compare these values
numerically against linear coefficients. Future model artifacts can support
permutation importance on the same validation hours, preferably grouping related
cyclic/calendar variables. Do not tune models from final interpretation plots.

The summed model's squared error includes a cross-product of casual and registered
errors. Positively correlated errors can reinforce each other, so separate models
are not guaranteed to improve total demand. Report improvements separately within
each model family; a point difference is not a statistical significance result.
A future uncertainty analysis should resample contiguous days/weeks, not independent
hours, and acknowledge the short single-season evaluation period.

The dataset covers **2011–2012**, not 2011 and 2013. The repository's earlier feature
selection used test performance, so this is not an untouched test estimate.
Existing weather interpolation uses future readings, and observed weather may
not be known for prospective forecasts. Missing calendar hours must not be treated
as zero rentals. These limitations constrain operational forecasting claims.

Source: https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset
