"""Ben Gutowski's baseline models. Run: python -m src.baselines.

Baselines set the floor every tuned model has to beat. They are deliberately
simple, have no tuned hyperparameters, and use the same chronological split,
the same five expanding-window folds, and the same metrics as the rest of the team.
"""
from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from src.linear_models import TARGETS, build_model, load_data, metrics

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "baselines"
N_SPLITS = 5
RECENT_WEEKS = 8  # fixed in advance, not tuned: about two months of recent demand
# Categorical slots from the team's figures: blue = registered, orange = casual.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300", "#e34948"]
TARGET_LABELS = {"casual": "Casual", "registered": "Registered", "cnt": "Direct total",
                 "summed_total": "Casual + registered"}


class HourProfile(BaseEstimator, RegressorMixin):
    """Predict the mean training count for each hour x working-day cell.

    With ``weeks`` set, only the most recent ``weeks`` of the training window are
    averaged, which tracks the ridership growth level at the cost of more noise.
    Means are taken on raw counts because the mean is what minimizes squared error.
    """

    def __init__(self, weeks=None):
        self.weeks = weeks

    def fit(self, X, y):
        frame = pd.DataFrame({"key": X["hour_workday"].to_numpy(), "time": X["datetime"].to_numpy(),
                              "y": np.asarray(y, dtype=float)})
        if self.weeks is not None:
            frame = frame[frame.time > frame.time.max() - pd.Timedelta(weeks=self.weeks)]
        self.profile_ = frame.groupby("key").y.mean()
        self.fallback_ = float(frame.y.mean())
        return self

    def predict(self, X):
        return X["hour_workday"].map(self.profile_).fillna(self.fallback_).to_numpy(dtype=float)


@dataclass
class Candidate:
    """A model recipe: rebuildable from scratch for every fold and every target."""
    name: str
    owner: str
    features: list
    make: Callable[[str], object]


def baseline_candidates(config):
    linear = sum(config["linear_features"].values(), [])
    profile = ["datetime", "hour_workday"]
    return [
        Candidate("Training mean", "Ben", linear, lambda target: DummyRegressor(strategy="mean")),
        Candidate("Hour profile", "Ben", profile, lambda target: HourProfile()),
        Candidate(f"Recent hour profile ({RECENT_WEEKS} wk)", "Ben", profile,
                  lambda target: HourProfile(weeks=RECENT_WEEKS)),
        # Marcus's exact pipeline with the penalty removed: the unregularized reference
        # that Ridge and Lasso must improve on. Full one-hot coding is rank-deficient;
        # lstsq returns the minimum-norm solution, so predictions are still well defined.
        Candidate("Linear regression (OLS)", "Ben", linear,
                  lambda target: build_model("Ridge", config).set_params(regressor__model=LinearRegression())),
    ]


def chronological_oof(candidate, train, target, cv):
    """Out-of-fold predictions from expanding windows; the first block stays NaN."""
    oof = np.full(len(train), np.nan)
    fold_rmse = []
    for fit, val in cv.split(train):
        model = candidate.make(target).fit(train[candidate.features].iloc[fit], train[target].iloc[fit])
        oof[val] = model.predict(train[candidate.features].iloc[val])
        fold_rmse.append(metrics(train[target].iloc[val], oof[val])["RMSE"])
    return oof, fold_rmse


def fit_predict(candidate, train, test, target):
    model = candidate.make(target).fit(train[candidate.features], train[target])
    return model, model.predict(test[candidate.features])


def data_hashes():
    files = [ROOT / "data/processed" / name for name in ("train.csv", "test.csv", "feature_config.json")]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "sklearn": sklearn.__version__}


def plot_rmse_bars(scores, title, path):
    """Grouped bars: one group per target, one bar per model, test RMSE."""
    models = list(dict.fromkeys(scores.model))
    targets = [t for t in TARGET_LABELS if t in set(scores.target)]
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(12, 4.5))
    for i, model in enumerate(models):
        rows = scores[scores.model == model].set_index("target").reindex(targets)
        x = np.arange(len(targets)) + (i - (len(models) - 1) / 2) * width
        ax.bar(x, rows.RMSE, width * 0.92, color=COLORS[i % len(COLORS)], label=model)
    ax.set_xticks(np.arange(len(targets)), [TARGET_LABELS[t] for t in targets])
    ax.set(ylabel="Test RMSE (rentals/hour)", title=title)
    ax.grid(axis="y", alpha=.3); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_daily(predictions, target, columns, labels, title, path):
    """Daily mean of hourly actual vs predicted demand over the test period."""
    daily = predictions.set_index("datetime")[[target, *columns]].resample("D").mean()
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(daily.index, daily[target], color="#0b0b0b", linewidth=2, label="Actual")
    for i, (column, label) in enumerate(zip(columns, labels)):
        ax.plot(daily.index, daily[column], color=COLORS[i % len(COLORS)], linewidth=1.5, label=label)
    ax.set(ylabel="Mean rentals/hour", xlabel="Test date", title=title)
    ax.grid(alpha=.3); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, frameon=False, ncol=3)
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def run(output_dir=OUTPUT):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train, test, config = load_data()
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows, folds = [], []
    predictions = test[["datetime", *TARGETS]].copy()
    for candidate in baseline_candidates(config):
        slug = candidate.name.split(" (")[0].lower().replace(" ", "_")
        for target in TARGETS:
            print(f"Baseline {candidate.name} for {target}", flush=True)
            _, fold_rmse = chronological_oof(candidate, train, target, cv)
            model, predicted = fit_predict(candidate, train, test, target)
            predictions[f"{slug}_{target}"] = predicted
            folds.extend({"model": candidate.name, "target": target, "fold": k, "RMSE": r}
                         for k, r in enumerate(fold_rmse, start=1))
            rows.append({"model": candidate.name, "target": target, "CV_RMSE": float(np.mean(fold_rmse)),
                         **metrics(test[target], predicted),
                         "train_RMSE": metrics(train[target], model.predict(train[candidate.features]))["RMSE"]})
        summed = predictions[f"{slug}_casual"] + predictions[f"{slug}_registered"]
        predictions[f"{slug}_summed_total"] = summed
        rows.append({"model": candidate.name, "target": "summed_total", **metrics(test.cnt, summed)})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(folds).to_csv(out / "cv_fold_scores.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)

    plot_rmse_bars(scores, "Baselines: test RMSE (Oct-Dec 2012)", out / "baseline_test_rmse.png")
    names = [c.name for c in baseline_candidates(config)]
    slugs = [n.split(" (")[0].lower().replace(" ", "_") for n in names]
    for target in TARGETS:
        plot_daily(predictions, target, [f"{s}_{target}" for s in slugs], names,
                   f"Baselines vs actual: {TARGET_LABELS[target].lower()} (daily mean of hourly values)",
                   out / f"baseline_daily_{target}.png")
    manifest = {**versions(), "folds": N_SPLITS, "recent_profile_weeks": RECENT_WEEKS,
                "selection_metric": "none (baselines are not tuned)", "data_sha256": data_hashes()}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(scores.to_string(index=False), flush=True)
    return scores


if __name__ == "__main__":
    run()
