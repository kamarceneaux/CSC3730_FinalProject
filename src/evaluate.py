"""Shared count-scale evaluation. Residual = actual - predicted (positive: underprediction).

Inputs must contain finite, nonnegative counts, not log predictions. No implicit
clipping, rounding, row dropping, or timestamp intersection is performed.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _vector(values):
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not len(result) or not np.isfinite(result).all():
        raise ValueError('Expected a nonempty, finite, one-dimensional vector')
    if (result < 0).any():
        raise ValueError('Expected nonnegative rental counts')
    return result


def metrics(actual, predicted):
    """RMSE, MAE, RMSLE; compatible with src.linear_models.metrics."""
    if isinstance(actual, pd.Series) and isinstance(predicted, pd.Series):
        if not actual.index.equals(predicted.index):
            raise ValueError('Series indexes differ')
    y, p = _vector(actual), _vector(predicted)
    if y.shape != p.shape:
        raise ValueError('Actual and predicted lengths differ')
    return {'RMSE': float(np.sqrt(np.mean((y-p)**2))),
            'MAE': float(np.mean(np.abs(y-p))),
            'RMSLE': float(np.sqrt(np.mean((np.log1p(y)-np.log1p(p))**2)))}


def to_counts(log_predictions):
    """Explicit log1p inversion with the team's nonnegative clipping policy."""
    values = np.asarray(log_predictions, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite log prediction')
    with np.errstate(over='ignore'):
        counts = np.maximum(np.expm1(values), 0)
    if not np.isfinite(counts).all():
        raise ValueError('Inverse transform overflow')
    return counts


def align_predictions(truth, predictions):
    """Align to truth by unique datetime; reject missing/extra hours or target drift."""
    frames = []
    for source in (truth, predictions):
        frame = source.copy()
        frame['datetime'] = pd.to_datetime(frame['datetime'], errors='raise')
        if frame.datetime.isna().any() or frame.datetime.duplicated().any():
            raise ValueError('Missing or duplicate timestamps')
        frames.append(frame.set_index('datetime').sort_index())
    actual, predicted = frames
    if not actual.index.equals(predicted.index):
        raise ValueError('Prediction hours must exactly match evaluation hours')
    for column in actual.columns.intersection(predicted.columns):
        if not np.allclose(actual[column], predicted[column], rtol=0, atol=1e-8):
            raise ValueError(f'Prediction file disagrees with truth: {column}')
    return actual.join(predicted.drop(columns=actual.columns.intersection(predicted.columns)))


def evaluate(actual, predicted):
    """Metrics plus sample size, bias and mean-normalized errors for group comparison."""
    result = metrics(actual, predicted)
    y, p = _vector(actual), _vector(predicted)
    mean = float(y.mean())
    return {**result, 'n': len(y), 'bias': float((y-p).mean()),
            'NRMSE': result['RMSE']/mean if mean else None,
            'NMAE': result['MAE']/mean if mean else None}


def compare_totals(actual, casual, registered, direct):
    """Arrays must already represent identical hours; use align_predictions first."""
    metrics(actual, casual)
    metrics(actual, registered)
    summed = _vector(casual) + _vector(registered)
    a, b = evaluate(actual, direct), evaluate(actual, summed)
    return pd.DataFrame([{'approach': 'direct', **a}, {'approach': 'summed', **b}])


def log_result(path, *, model, target, actual, predicted, run_id, split='test', metadata=None):
    """Append a JSONL record. Use one file per worker; concurrent writers unsupported.

    Caller supplies run_id and provenance (data hashes, settings) in metadata.
    Repeated run IDs are retained, never silently overwritten.
    """
    record = {'run_id': str(run_id), 'model': str(model), 'target': str(target),
              'split': str(split), **evaluate(actual, predicted), 'metadata': metadata or {}}
    line = json.dumps(record, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(line + '\n')
    return record


def residual_summary(frame, actual, predicted, by):
    """Stratified count, RMSE, MAE and signed bias; retains small groups explicitly."""
    rows = []
    for value, group in frame.groupby(by, observed=True, dropna=False):
        rows.append({by: value, **evaluate(group[actual], group[predicted])})
    return pd.DataFrame(rows)


def plot_diagnostics(dates, actual, predicted, title='Residual diagnostics'):
    """Return Figure; caller owns saving and closing. Daily curves use observed hours."""
    metrics(actual, predicted)
    y, p = _vector(actual), _vector(predicted)
    dates = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(dates) != len(y) or dates.hasnans or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError('Dates must be unique, sorted and aligned')
    residual = y-p
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    ax = axes[0, 0]
    ax.scatter(y, p, s=6, alpha=.25)
    limit = max(y.max(), p.max(), 1)
    ax.plot([0, limit], [0, limit], 'k--')
    ax.set(xlabel='Actual rentals/hour', ylabel='Predicted rentals/hour')
    ax = axes[0, 1]
    ax.scatter(p, residual, s=6, alpha=.25)
    ax.axhline(0, color='black', linestyle='--')
    ax.set(xlabel='Predicted rentals/hour', ylabel='Actual minus predicted')
    ax = axes[1, 0]
    ax.hist(residual, bins=40)
    ax.axvline(0, color='black', linestyle='--')
    ax.set(xlabel='Actual minus predicted', ylabel='Hours')
    daily = pd.DataFrame({'actual': y, 'predicted': p}, index=dates).resample('D').mean()
    daily.plot(ax=axes[1, 1])
    axes[1, 1].set(xlabel='Date', ylabel='Mean rentals per observed hour')
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_driver_comparison(importance, title='Model feature importance'):
    """Plot existing impurity importances; associations, not causal effects.

    Requires feature, target, importance columns from the same model family.
    """
    data = importance[importance.target.isin(['casual', 'registered'])]
    table = data.pivot(index='feature', columns='target', values='importance')
    if set(table.columns) != {'casual', 'registered'} or table.isna().any().any():
        raise ValueError('Both groups must have identical feature sets')
    table = table.loc[table.max(axis=1).sort_values().index]
    ax = table.plot.barh(figsize=(9, 7))
    ax.set(title=title, xlabel='Impurity importance (within-model share)', ylabel='')
    ax.figure.tight_layout()
    return ax.figure
