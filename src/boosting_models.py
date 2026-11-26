"""Boosting experiments for Bike Sharing hour.csv.

All hyperparameter selection uses the 2011-01 through 2012-09 training period.
October-December 2012 is held out for reporting, not model selection.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit, ParameterGrid
from xgboost import XGBRegressor

FEATURES = ['hr', 'mnth', 'weekday', 'season', 'weathersit',
            'yr', 'holiday', 'workingday', 'temp', 'hum', 'windspeed',
            'hr_sin', 'hr_cos', 'mnth_sin', 'mnth_cos',
            'weekday_sin', 'weekday_cos']
TARGETS = ['casual', 'registered', 'cnt']
GRIDS = {
    'GradientBoosting': {
        'learning_rate': [0.03, 0.1],
        'max_depth': [2, 3],
        'n_estimators': [150, 350],
    },
    'XGBoost': {
        'learning_rate': [0.03, 0.1],
        'max_depth': [3, 5],
        'n_estimators': [800],  # upper bound: early stopping selects the actual tree count
    },
}


def load_data(path):
    df = pd.read_csv(path)
    df['datetime'] = pd.to_datetime(df['dteday']) + pd.to_timedelta(df['hr'], unit='h')
    df = df.sort_values('datetime').reset_index(drop=True)
    df['weathersit'] = df['weathersit'].clip(upper=3)
    df['hum'] = df['hum'].replace(0, np.nan).ffill()
    df['hum'] = df['hum'].fillna(df['hum'].median())
    for col, period in {'hr': 24, 'mnth': 12, 'weekday': 7}.items():
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    assert not set(FEATURES).intersection(TARGETS)
    train = df.loc[df['datetime'] < '2012-10-01'].copy()
    test = df.loc[df['datetime'] >= '2012-10-01'].copy()
    return train, test


def to_counts(log_predictions):
    return np.clip(np.expm1(np.asarray(log_predictions)), 0, None)


def metrics(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.clip(np.asarray(predicted), 0, None)
    return {'RMSE': float(np.sqrt(mean_squared_error(actual, predicted))),
            'MAE': float(mean_absolute_error(actual, predicted)),
            'RMSLE': float(np.sqrt(mean_squared_log_error(actual, predicted)))}


def make_model(family, params, early_stopping=False):
    if family == 'GradientBoosting':
        return GradientBoostingRegressor(random_state=42, loss='squared_error', **params)
    return XGBRegressor(objective='reg:squarederror', random_state=42,
                        n_jobs=2, tree_method='hist',
                        **({'early_stopping_rounds': 35} if early_stopping else {}),
                        **params)


def tune(family, X, y, target, results_dir, quick=False):
    cv = TimeSeriesSplit(n_splits=2 if quick else 5)
    logs = []
    for params in ParameterGrid(({k: [v[0]] for k, v in GRIDS[family].items()} if quick else GRIDS[family])):
        fold_scores, best_rounds = [], []
        for fold, (train_idx, val_idx) in enumerate(cv.split(X), 1):
            model = make_model(family, params, early_stopping=(family == 'XGBoost'))
            if family == 'XGBoost':
                model.fit(X.iloc[train_idx], np.log1p(y.iloc[train_idx]),
                          eval_set=[(X.iloc[val_idx], np.log1p(y.iloc[val_idx]))],
                          verbose=False)
                best_rounds.append(int(model.best_iteration + 1))
            else:
                model.fit(X.iloc[train_idx], np.log1p(y.iloc[train_idx]))
            pred = to_counts(model.predict(X.iloc[val_idx]))
            score = metrics(y.iloc[val_idx], pred)
            fold_scores.append(score['RMSE'])
            logs.append({'model': family, 'target': target, 'fold': fold,
                         **params, **score, 'best_rounds': best_rounds[-1] if best_rounds else None})
        # Explicit selection on count-scale CV RMSE, as in the team report.
    log_df = pd.DataFrame(logs)
    log_df.to_csv(results_dir / f'{family}_{target}_folds.csv', index=False)
    keys = list(GRIDS[family])
    summary = log_df.groupby(keys, dropna=False, as_index=False).agg(
        CV_RMSE=('RMSE', 'mean'), CV_RMSE_std=('RMSE', 'std'),
        CV_MAE=('MAE', 'mean'), CV_RMSLE=('RMSLE', 'mean'))
    summary = summary.sort_values('CV_RMSE').reset_index(drop=True)
    summary.to_csv(results_dir / f'{family}_{target}_tuning.csv', index=False)
    best = {k: (int(summary.loc[0, k]) if k in ('max_depth', 'n_estimators') else float(summary.loc[0, k])) for k in keys}
    if family == 'XGBoost':
        match = np.ones(len(log_df), dtype=bool)
        for k in keys:
            match &= np.isclose(log_df[k].astype(float).to_numpy(), float(best[k]))
        best['n_estimators'] = max(1, int(np.median(log_df.loc[match, 'best_rounds'])))
    return best, summary.loc[0].to_dict()


def plot_diagnostics(actual, predicted, dates, family, target, out):
    residual = np.asarray(actual) - np.asarray(predicted)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(predicted, residual, s=7, alpha=.3)
    ax.axhline(0, linestyle='--', color='black')
    ax.set(xlabel='Predicted rentals/hour', ylabel='Actual - predicted',
           title=f'{family}: {target} residuals')
    fig.tight_layout()
    fig.savefig(out / f'{family}_{target}_residuals.png', dpi=160)
    plt.close(fig)
    by_day = pd.DataFrame({'day': pd.to_datetime(dates).dt.date,
                           'actual': np.asarray(actual), 'predicted': np.asarray(predicted)})
    by_day = by_day.groupby('day')[['actual', 'predicted']].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    by_day.plot(ax=ax)
    ax.set(xlabel='Test date', ylabel='Mean rentals per recorded hour',
           title=f'{family}: {target} daily actual vs predicted')
    fig.tight_layout()
    fig.savefig(out / f'{family}_{target}_daily.png', dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='hour.csv')
    parser.add_argument('--out', default='boosting_results')
    parser.add_argument('--quick', action='store_true', help='Small test run to check installation and code')
    args = parser.parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        data_path = Path(__file__).resolve().parent / args.data
    if not data_path.exists():
        parser.error(f'Cannot find dataset: {args.data}. Put hour.csv next to this script.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_data(data_path)
    if args.quick:
        train = train.iloc[:2500].copy()
        test = test.iloc[:150].copy()
        print('Quick check: using a reduced dataset and one setting per model')
    X_train, X_test = train[FEATURES], test[FEATURES]
    final_rows = []
    predictions = pd.DataFrame({'datetime': test['datetime'],
                                **{f'actual_{t}': test[t] for t in TARGETS}})
    for family in GRIDS:
        for target in TARGETS:
            print(f'Tuning {family} for {target} ...', flush=True)
            y_train = train[target]
            best_params, cv_result = tune(family, X_train, y_train, target, out, quick=args.quick)
            model = make_model(family, best_params)
            model.fit(X_train, np.log1p(y_train))
            pred = to_counts(model.predict(X_test))
            predictions[f'{family}_{target}'] = pred
            final_rows.append({'Model': family, 'Target': target,
                               'CV_RMSE': cv_result['CV_RMSE'],
                               **metrics(test[target], pred),
                               'Parameters': json.dumps(best_params)})
            plot_diagnostics(test[target], pred, test['datetime'], family, target, out)
            importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values().tail(12)
            fig, ax = plt.subplots(figsize=(8, 5))
            importances.plot.barh(ax=ax)
            ax.set(title=f'{family}: {target} feature importance', xlabel='Model feature importance')
            fig.tight_layout()
            fig.savefig(out / f'{family}_{target}_importance.png', dpi=160)
            plt.close(fig)
        summed = predictions[f'{family}_casual'] + predictions[f'{family}_registered']
        predictions[f'{family}_summed'] = summed
        final_rows.append({'Model': family, 'Target': 'Summed groups',
                           'CV_RMSE': np.nan, **metrics(test['cnt'], summed),
                           'Parameters': 'separate tuned casual + registered'})
    pd.DataFrame(final_rows).to_csv(out / 'boosting_comparison.csv', index=False)
    predictions.to_csv(out / 'boosting_test_predictions.csv', index=False)
    print(pd.DataFrame(final_rows)[['Model', 'Target', 'RMSE', 'MAE', 'RMSLE']].to_string(index=False))
    print(f'All experiment logs and plots saved to {out.resolve()}')


if __name__ == '__main__':
    main()
