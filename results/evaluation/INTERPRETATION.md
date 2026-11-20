# Evaluation results and interpretation

These results were recomputed from the committed predictions at repository commit
`40361cabfa89ca7ce70ffdff9a04bafe09256955`. Every model was aligned to the same
2,168 observed hours from October 1 through December 31, 2012. Predictions were
scored in rental counts, with no additional clipping or retraining.

## Does adding group predictions improve total demand?

| Model | Direct RMSE | Summed RMSE | RMSE improvement | Direct MAE | Summed MAE |
|---|---:|---:|---:|---:|---:|
| Ridge | 70.02 | 70.63 | −0.87% | 42.97 | 43.43 |
| Lasso | 74.25 | 72.85 | 1.90% | 46.62 | 45.42 |
| Decision Tree baseline | 95.50 | 98.12 | −2.75% | 59.51 | 63.55 |
| Decision Tree tuned | 99.98 | 91.45 | 8.53% | 63.29 | 59.42 |
| Random Forest baseline | 88.16 | 82.98 | 5.88% | 58.57 | 56.45 |
| Random Forest tuned | 88.16 | 82.98 | 5.88% | 58.57 | 56.45 |

Positive improvement means lower error after summing. Separate modeling helped
Lasso and both tuned tree families, but did not help Ridge or the baseline decision
tree. The best total prediction among these saved candidates was **direct Ridge**.
Therefore the initial expectation that tree models would outperform linear models
is not supported for total demand on this period. Tuned Random Forest and its
baseline have effectively identical scores; they are not independent confirmations.
The tuned direct Decision Tree performs worse than its baseline on this period,
illustrating that training-CV tuning does not guarantee test improvement.

## How predictable are the two groups?

Random Forest achieved the lowest casual RMSE (24.10) and MAE (12.31). Ridge achieved
the lowest registered RMSE (62.16) and MAE (36.09). Registered counts are larger, so
its larger raw RMSE does not establish poorer predictability. For Random Forest,
RMSE divided by mean demand is 0.760 for casual riders versus 0.385 for registered
riders. For Ridge, these ratios are 0.848 and 0.332. Both comparisons support lower
relative error for registered riders in this evaluation period.

## What do residuals reveal?

Residuals are actual minus predicted. Casual demand is underestimated during the
day in all four main model families. For example, Ridge underpredicts by 27.32
rentals/hour on average at 14:00 and 27.10 at 15:00. Its overall casual bias is +8.08.
The Random Forest registered model has overall bias +27.32 rentals/hour, with
pronounced morning and evening peak underprediction. The diagnostic scatter also
shows error spread increasing with predicted demand.

Ridge's registered residuals change sign: afternoon underprediction gives way to
evening overprediction. Consequently, a small aggregate bias can conceal systematic
hour-specific errors. The direct Ridge model's overall bias is only +0.21, which
should be read alongside its RMSE of 70.02 and the hourly diagnostic patterns.
See `hourly_bias.png` and the individual residual panels; `residual_strata.csv`
provides sample counts and errors by hour, working-day status, and weather.

## Which predictors distinguish the groups?

The tuned Random Forest assigns hour-of-day importance of 0.475 to casual demand
and 0.609 to registered demand. Temperature importance is 0.245 for casual demand
versus 0.075 for registered demand; working-day importance is 0.053 versus 0.077.
These patterns are consistent with stronger weather-related variation in casual
riding and stronger time/calendar structure in registered riding.

These are impurity importances from models fitted to log targets, not causal
estimates or evidence of positive/negative effects. The model also includes cyclic
versions of time variables, so individual importances divide overlapping signals.
Temperature's higher importance does not by itself quantify how many extra rentals
warmer weather causes. The two driver figures compare the same features side by
side within each model family.

## Limits on the conclusions

The dataset spans 2011–2012, not 2011 and 2013. Prior feature engineering used this
test period, as documented in PROJECT_REVIEW.md; report it as a fixed comparison
set rather than an untouched generalization estimate. The existing weather
interpolation uses future readings, and observed weather is not necessarily
available at forecast issuance. This evaluates the committed prediction pipeline,
not a fully validated prospective forecasting system. Missing calendar hours are
excluded, not imputed to zero. There are no confidence intervals or significance
claims here. The small Ridge and Lasso differences particularly merit temporal
uncertainty analysis before strong claims. The comparison describes complete
pipelines with different feature representations, not a controlled isolation of
algorithm choice.

## Suggested report paragraph

On the common October–December 2012 comparison set, the benefit of modeling rider
groups separately depended on model family. Summing casual and registered
predictions reduced total RMSE by 8.53% for the tuned Decision Tree, 5.88% for the
Random Forest, and 1.90% for Lasso, while Ridge's error increased by 0.87%. Direct
Ridge produced the lowest total RMSE (70.02 rentals/hour). Random Forest performed
best for casual rentals, whereas Ridge performed best for registered rentals.
Relative errors were larger for casual riders. Driver comparisons emphasized
hour-of-day for both groups, with greater temperature importance for casual demand
and greater working-day importance for registered demand. Residual analysis showed
systematic daytime casual underprediction and peak-period registered underprediction
in the tree models. These findings are descriptive of the fixed comparison set,
which had already influenced feature selection, and do not establish causal effects
or statistically significant differences.

## Verification

All 14 automated tests passed, including the three existing linear-model tests and
11 new evaluation cases. Timestamp alignment, invalid inputs, count inversion,
known metric values, logger append behavior, grouped residuals, and plot creation
were checked. The report regenerated successfully; representative total, hourly,
driver, and residual figures were visually inspected. Git whitespace checks passed.
Local validation used Python 3.9 with numpy 2.0.2, pandas 2.3.3, scikit-learn 1.6.1,
and matplotlib 3.9.4. This differs from the team's pinned environment; CI with the
team pins remains to be run. Tests emitted dependency deprecation warnings,
a single-day plot axis warning, and numerical warnings in the existing linear
smoke test; its returned predictions nevertheless passed finite-value assertions.
No models were retrained for the report and no notebooks were re-executed.
