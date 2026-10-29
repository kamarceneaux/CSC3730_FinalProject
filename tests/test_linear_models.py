"""Protect the temporal split, target separation, and count-scale predictions."""
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from src.linear_models import TARGETS, build_model, load_data, metrics, to_counts


def test_temporal_split_and_target_exclusion():
    train, test, config = load_data()
    features = sum(config['linear_features'].values(), [])
    assert len(train) == 15211 and len(test) == 2168
    assert not set(features) & set(TARGETS)
    for fit, val in TimeSeriesSplit(5).split(train):
        assert train.datetime.iloc[fit[-1]] < train.datetime.iloc[val[0]]
    assert train.datetime.max() < test.datetime.min()


def test_metrics_and_inverse_use_counts():
    np.testing.assert_allclose(to_counts(np.log1p([0, 5, 20])), [0, 5, 20])
    assert to_counts([-2])[0] == 0
    result = metrics([0, 10], [0, 12])
    assert np.isclose(result['RMSE'], np.sqrt(2))
    assert result['MAE'] == 1


def test_scaler_is_fitted_on_training_only():
    train, test, config = load_data()
    features = sum(config['linear_features'].values(), [])
    subset = train.iloc[:1000]
    model = build_model('Ridge', config).fit(subset[features], subset.casual)
    scaler = model.regressor_.named_steps['preprocess'].named_transformers_['numeric']
    np.testing.assert_allclose(scaler.mean_, subset[config['linear_features']['numeric_scaled']].mean())
    predicted = model.predict(test[features].iloc[:50])
    assert np.isfinite(predicted).all() and (predicted >= 0).all()
