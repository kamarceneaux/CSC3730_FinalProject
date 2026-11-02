# Tree-Based Models: Results and Reproducibility

## Overview
This folder contains the Decision Tree and Random Forest models developed for the Bike Sharing Demand Prediction project.

The models predict hourly bike rental demand for casual users, registered users, and total rentals using the processed UCI Bike Sharing dataset.

## Modeling Approach
- Used the project's chronological training and testing split.
- Applied a log transformation to rental count targets to help manage skewness.
- Used TimeSeriesSplit cross-validation with five folds for hyperparameter tuning.
- Evaluated models using RMSE, MAE, and RMSLE.
- Compared predictions for casual and registered rentals against predictions made directly for total rentals.

## Model Results

Random Forest achieved the strongest overall performance among the tree-based models.

| Target | Random Forest RMSE | Random Forest MAE |
|---|---:|---:|
| Casual | 24.10 | 12.31 |
| Registered | 72.11 | 47.88 |
| Total | 88.16 | 58.57 |

## Key Findings
- Hour of day was the most influential feature across all three prediction targets.
- Temperature was more influential when predicting casual rentals than registered rentals.
- Working day status was relatively more influential for registered rental predictions.
- Predicting casual and registered rentals separately and summing their predictions performed better than directly predicting total rentals.

## Files
This folder contains model evaluation metrics, prediction outputs, feature importance results, comparison figures, and the tree model notebook.

The notebook is located at `results/danielle_tree_models.ipynb`.