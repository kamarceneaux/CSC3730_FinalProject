# Marcus Reese — Ridge and Lasso results

## Method

Six models were fitted using the team's 15 linear input columns: one Ridge and one Lasso for each of casual, registered, and total demand. The full training design matrix contains 67 encoded columns. Each model fits log1p rentals, with predictions returned to nonnegative counts. Scaling and encoding are learned within the training fold.

Training contains 15,211 hours (January 2011–September 2012); the fixed comparison set contains 2,168 observed hours (October–December 2012). Five expanding training folds evaluated 15 Ridge alpha values and 16 Lasso values per target: 465 candidate/fold fits, followed by six refits on the full training period. Selection used mean validation RMSE in rental counts. No convergence warning was accepted. The model grids were specified before model evaluation; neither model family nor feature set was changed based on these test scores.

## Selected parameters and measured errors

| Model | Target | Alpha | CV RMSE | Test RMSE | Test MAE | Test RMSLE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ridge | Casual | 0.001 | 34.598 | 26.908 | 12.744 | 0.595 |
| Ridge | Registered | 10 | 84.535 | 62.155 | 36.088 | 0.444 |
| Ridge | Direct total | 10 | 104.039 | 70.019 | 42.970 | 0.426 |
| Ridge | Summed groups | — | — | 70.625 | 43.434 | 0.426 |
| Lasso | Casual | 0.001 | 34.586 | 27.889 | 13.074 | 0.597 |
| Lasso | Registered | 0.002154435 | 79.618 | 62.719 | 38.079 | 0.448 |
| Lasso | Direct total | 0.002154435 | 99.257 | 74.255 | 46.624 | 0.433 |
| Lasso | Summed groups | — | — | 72.845 | 45.422 | 0.430 |

RMSE and MAE are in rentals/hour. CV scores are means across distinct historical validation periods, not uncertainty bounds for the test scores. Summed-group rows combine the separately selected models; their alpha and CV score are not a single direct-model parameter or score.

## Interpretation

**Total demand:** Ridge's direct-total prediction gives the lowest observed total-demand RMSE among these four approaches. Summing group models worsens Ridge RMSE by about 0.87% (70.625 versus 70.019), but improves Lasso RMSE by about 1.90% (72.845 versus 74.255). Thus this experiment gives mixed evidence for separate rider modeling; it does not support a universal improvement from summing predictions. This comparison covers only Marcus's model families, not the team's tree, boosting, distance, or ensemble models. The small score differences have not been assessed for statistical significance.

**Rider groups:** Casual models have lower absolute count error, but higher RMSLE than registered models. The smaller casual target scale makes its lower RMSE insufficient evidence that casual behavior is easier to predict. Casual scatterplots show substantial underprediction of the largest rental peaks. Mean actual-minus-predicted error is +8.08 rentals/hour for Ridge casual and +8.60 for Lasso casual, consistent with that pattern. Log-target fitting can favor relative fit at the expense of large count peaks.

**Regularization:** Moderate Ridge shrinkage (alpha 10) is selected for registered and direct total. Ridge casual selects the smallest searched value (0.001), at the boundary of the grid, and its low-alpha validation curve is nearly flat. Treat this as a near-unregularized fit and a boundary limitation, not proof that 0.001 is globally optimal. Strong shrinkage substantially worsens validation error in both families. An optional follow-up is a predeclared training-only comparison with still smaller Ridge alphas or unregularized regression; the current test scores must not guide that experiment.

**Sparsity:** Lasso retains 56 of 67 coefficients for casual demand, 50 for registered demand, and 52 for direct total (absolute-value threshold 1e-10). Ridge retains all 67. The encoded feature set contains redundant one-hot, cyclic, and binary terms, so these zero coefficients should not be interpreted as proof a real-world factor is irrelevant. Coefficients operate on log1p rentals, and weather columns are standardized; other columns use different scales. These are conditional predictive associations, not causal effects.

**Residuals and time:** The prediction scatter and residual plots reveal larger errors at high demand. Casual daily mean curves miss several high-demand days. Registered and total models follow the broad temporal decline, but individual-hour errors remain. Positive residuals denote underprediction. Days missing recorded hours are summarized over available observations; the curves do not represent complete-day totals for those days.

## Evaluation caveats

The earlier team EDA notebook explicitly used test RMSLE to justify the hour/working-day interaction. The fixed comparison set therefore already influenced feature selection, although alpha selection here uses training data only. Do not describe it as fully untouched. The preprocessing also interpolates humidity in both temporal directions, and the models use observed weather; these assumptions limit claims about a prospective forecast using only information available ahead of time. See `../../PROJECT_REVIEW.md` for the complete audit.

## Deliverables and reproducibility

- `cv_results.csv`: every candidate's train/validation scores, fold scores, ranks, and timings.
- `cv_folds.csv`: exact chronological training/validation boundaries.
- `metrics.csv`: final comparisons and selected-model training RMSE.
- `test_predictions.csv`: timestamps, actual targets, and six individual plus two summed predictions.
- `coefficients.csv`: encoded feature coefficients for each fitted model.
- Twelve PNGs: six tuning curves and six diagnostic panels.
- `run_manifest.json`: versions, data SHA-256 hashes, choices, and boundary flags.
- Six `.joblib` files: fitted preprocessing, target transformation, and regressors; locally regenerated and ignored by Git.

Only load trusted model files. For a team ensemble, generate separate training-period predictions with a chronological protocol that also excludes each predicted row from hyperparameter selection; do not train an ensemble on these test predictions.
