"""Protect the baseline models and their chronological out-of-fold scoring."""
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from src.baselines import HourProfile, baseline_candidates, chronological_oof
from src.linear_models import load_data


def test_hour_profile_means_and_recent_window():
    times = pd.date_range("2012-01-01", periods=4, freq="7D")
    X = pd.DataFrame({"datetime": times, "hour_workday": [8, 8, 8, 32]})
    y = [10, 20, 30, 100]
    assert HourProfile().fit(X, y).predict(pd.DataFrame({"hour_workday": [8, 32, 99]})).tolist() == [20, 100, 40]
    recent = HourProfile(weeks=2).fit(X, y)  # keeps only the last two weeks
    assert recent.predict(pd.DataFrame({"hour_workday": [8]}))[0] == 30


def test_out_of_fold_predictions_never_see_their_own_rows():
    train, _, config = load_data()
    subset = train.iloc[:3000]
    cv = TimeSeriesSplit(5)
    profile = next(c for c in baseline_candidates(config) if c.name == "Hour profile")
    oof, fold_rmse = chronological_oof(profile, subset, "registered", cv)
    first_val = next(iter(cv.split(subset)))[1][0]
    assert np.isnan(oof[:first_val]).all() and np.isfinite(oof[first_val:]).all()
    assert len(fold_rmse) == 5
    # Changing a validation-block target must not change that block's prediction.
    changed = subset.copy()
    changed.loc[changed.index[-1], "registered"] = 10_000
    oof_changed, _ = chronological_oof(profile, changed, "registered", cv)
    assert oof_changed[-1] == oof[-1]
