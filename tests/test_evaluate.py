import json
import numpy as np
import pandas as pd
import pytest
from src.evaluate import metrics, to_counts, align_predictions, compare_totals, log_result


def test_known_errors():
    result = metrics([0, 10], [0, 12])
    assert result['RMSE'] == pytest.approx(np.sqrt(2))
    assert result['MAE'] == 1
    assert result['RMSLE'] == pytest.approx(abs(np.log(11)-np.log(13))/np.sqrt(2))


@pytest.mark.parametrize('bad', [[], [np.nan], [np.inf], [-1], [[1]]])
def test_reject_invalid_counts(bad):
    with pytest.raises(ValueError):
        metrics(bad, bad)


def test_length_and_index_mismatch():
    with pytest.raises(ValueError):
        metrics([1, 2], [1])
    with pytest.raises(ValueError):
        metrics(pd.Series([1], index=[1]), pd.Series([1], index=[2]))


def test_alignment_and_rejection():
    truth = pd.DataFrame({'datetime': ['2012-10-01', '2012-10-02'], 'cnt': [3, 6]})
    pred = pd.DataFrame({'datetime': ['2012-10-02', '2012-10-01'], 'p': [6, 3]})
    aligned = align_predictions(truth, pred)
    assert metrics(aligned.cnt, aligned.p)['RMSE'] == 0
    for invalid in (pred.iloc[:1], pd.concat([pred, pred.iloc[:1]]), pred.assign(cnt=[7, 3])):
        with pytest.raises(ValueError):
            align_predictions(truth, invalid)


def test_sum_and_explicit_transform():
    result = compare_totals([5, 10], [2, 3], [3, 7], [4, 9])
    assert result.iloc[1].RMSE == 0
    np.testing.assert_allclose(to_counts(np.log1p([0, 12])), [0, 12])
    assert to_counts([-2])[0] == 0
    with pytest.raises(ValueError):
        to_counts([1000])


def test_logger_appends_and_zero_normalization(tmp_path):
    path = tmp_path/'nested/results.jsonl'
    for _ in range(2):
        log_result(path, model='test', target='casual', actual=[0], predicted=[0], run_id='a')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]['NRMSE'] is None


def test_plots_and_strata():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from src.evaluate import plot_diagnostics, plot_driver_comparison, residual_summary
    dates = pd.date_range('2012-10-01', periods=3, freq='h')
    fig = plot_diagnostics(dates, [0, 2, 4], [1, 1, 2])
    assert len(fig.axes) == 4
    plt.close(fig)
    with pytest.raises(ValueError):
        plot_diagnostics(dates[::-1], [0, 2, 4], [1, 1, 2])
    frame = pd.DataFrame({'hour': [0, 0, 1], 'actual': [0, 2, 4], 'pred': [1, 1, 2]})
    summary = residual_summary(frame, 'actual', 'pred', 'hour')
    assert summary.iloc[0]['n'] == 2 and summary.iloc[1]['bias'] == 2
    imp = pd.DataFrame({'feature': ['hr', 'hr'], 'target': ['casual', 'registered'], 'importance': [.4, .6]})
    fig = plot_driver_comparison(imp)
    plt.close(fig)
