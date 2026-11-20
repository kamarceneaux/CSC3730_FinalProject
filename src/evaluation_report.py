"""Recompute shared evaluation from committed predictions: python -m src.evaluation_report."""
from pathlib import Path
import hashlib
import json
import platform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .evaluate import (align_predictions, evaluate, log_result, plot_diagnostics,
                       plot_driver_comparison, residual_summary)

ROOT = Path(__file__).resolve().parents[1]


def run(root=ROOT, output=None):
    root = Path(root)
    out = Path(output) if output else root/'results/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    truth = pd.read_csv(root/'data/processed/test.csv')[['datetime', 'casual', 'registered', 'cnt', 'hr', 'workingday', 'weathersit']]
    if not (truth.casual + truth.registered == truth.cnt).all():
        raise ValueError('Targets do not sum')
    specs = [('Ridge', 'linear_models/test_predictions.csv', 'ridge'),
             ('Lasso', 'linear_models/test_predictions.csv', 'lasso'),
             ('Decision Tree baseline', 'tree_models/decision_tree_baseline_predictions.csv', 'dt_baseline'),
             ('Decision Tree tuned', 'tree_models/decision_tree_tuned_predictions.csv', 'dt_tuned'),
             ('Random Forest baseline', 'tree_models/random_forest_baseline_predictions.csv', 'rf_baseline'),
             ('Random Forest tuned', 'tree_models/random_forest_tuned_predictions.csv', 'rf_tuned')]
    rows, stratified, comparisons, inputs = [], [], [], {}
    log = out/'results.jsonl'
    # This report owns its log; the shared logger itself always appends.
    log.write_text('')
    for name, filename, prefix in specs:
        path = root/'results'/filename
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = align_predictions(truth, pd.read_csv(path))
        data[f'{prefix}_summed_total'] = data[f'{prefix}_casual'] + data[f'{prefix}_registered']
        scores = {}
        for target in ('casual', 'registered', 'cnt', 'summed_total'):
            actual = 'cnt' if target == 'summed_total' else target
            prediction = f'{prefix}_{target}'
            score = log_result(log, model=name, target=target, actual=data[actual], predicted=data[prediction],
                               run_id='committed-predictions', metadata={'source': filename, 'sha256': inputs[str(path.relative_to(root))]})
            scores[target] = score
            rows.append({k: v for k, v in score.items() if k != 'metadata'})
            for column in ('hr', 'workingday', 'weathersit'):
                summary = residual_summary(data, actual, prediction, column).rename(columns={column: 'value'})
                summary['stratum'], summary['model'], summary['target'] = column, name, target
                stratified.append(summary)
            if 'baseline' not in name:
                fig = plot_diagnostics(data.index, data[actual], data[prediction], f'{name}: {target}')
                fig.savefig(out/f'{prefix}_{target}_residuals.png', dpi=140)
                plt.close(fig)
        direct, summed = scores['cnt'], scores['summed_total']
        comparisons.append({'model': name, 'direct_RMSE': direct['RMSE'], 'summed_RMSE': summed['RMSE'],
                            'RMSE_improvement_pct': 100*(direct['RMSE']-summed['RMSE'])/direct['RMSE'],
                            'direct_MAE': direct['MAE'], 'summed_MAE': summed['MAE'],
                            'group_error_correlation': float(np.corrcoef(data.casual-data[f'{prefix}_casual'], data.registered-data[f'{prefix}_registered'])[0, 1])})
    scores = pd.DataFrame(rows)
    scores.to_csv(out/'metrics.csv', index=False)
    strata = pd.concat(stratified)
    strata.to_csv(out/'residual_strata.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, target in zip(axes, ('casual', 'registered')):
        hourly = strata[(strata.stratum == 'hr') & (strata.target == target) & ~strata.model.str.contains('baseline')]
        hourly.pivot(index='value', columns='model', values='bias').plot(ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        ax.set(title=target.title(), xlabel='Hour of day', ylabel='Mean actual minus predicted')
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'hourly_bias.png', dpi=150); plt.close(fig)
    comparison = pd.DataFrame(comparisons)
    comparison.to_csv(out/'total_comparison.csv', index=False)
    ax = comparison.set_index('model')[['direct_RMSE', 'summed_RMSE']].plot.barh(figsize=(10, 5))
    ax.set(xlabel='RMSE (rentals/hour)', ylabel='', title='Same 2,168 evaluation hours: direct versus summed')
    ax.figure.tight_layout(); ax.figure.savefig(out/'total_comparison.png', dpi=150); plt.close(ax.figure)
    for family in ('random_forest', 'decision_tree'):
        path = root/f'results/tree_models/{family}_feature_importance.csv'
        inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        fig = plot_driver_comparison(pd.read_csv(path), family.replace('_', ' ').title() + ': rider-group comparison')
        fig.savefig(out/f'{family}_drivers.png', dpi=150); plt.close(fig)
    path = root/'data/processed/test.csv'
    inputs[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    (out/'manifest.json').write_text(json.dumps({'inputs_sha256': inputs, 'python': platform.python_version(),
        'numpy': np.__version__, 'pandas': pd.__version__, 'matplotlib': matplotlib.__version__,
        'period': [str(data.index.min()), str(data.index.max())], 'n': len(data),
        'residual_sign': 'actual minus predicted', 'source': 'Saved predictions; no retraining'}, indent=2))
    print(comparison.to_string(index=False))
    return scores


if __name__ == '__main__':
    run()
