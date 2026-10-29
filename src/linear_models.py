"""Marcus Reese's regularized linear models. Run: python -m src.linear_models."""
from __future__ import annotations

import hashlib
import json
import platform
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_squared_log_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "linear_models"
TARGETS = ("casual", "registered", "cnt")
# Separate ranges: sklearn's Ridge and Lasso objectives normalize loss differently.
ALPHAS = {"Ridge": np.logspace(-3, 4, 15), "Lasso": np.logspace(-5, 0, 16)}


def to_counts(log_predictions):
    """Invert the team's log target convention and enforce nonnegative counts."""
    return np.maximum(np.expm1(log_predictions), 0.0)


def metrics(actual, predicted):
    """All errors use original rental counts; RMSLE additionally logs counts."""
    return {"RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
            "MAE": float(mean_absolute_error(actual, predicted)),
            "RMSLE": float(np.sqrt(mean_squared_log_error(actual, predicted)))}


def load_data(root=ROOT):
    """Use the team's committed split and reject leakage or misaligned targets."""
    root = Path(root)
    config = json.loads((root / "data/processed/feature_config.json").read_text())
    train, test = [pd.read_csv(root / f"data/processed/{name}.csv", parse_dates=["datetime"])
                   for name in ("train", "test")]
    features = sum(config["linear_features"].values(), [])
    if set(features) & set(config["excluded"]):
        raise ValueError("Excluded/target column appears in predictors")
    for frame in (train, test):
        if not frame.datetime.is_monotonic_increasing or frame.datetime.duplicated().any():
            raise ValueError("Timestamps must be unique and chronological")
        if frame[features + list(TARGETS)].isna().any().any():
            raise ValueError("Missing modeling values")
        if not (frame.casual + frame.registered == frame.cnt).all():
            raise ValueError("Rental totals do not match rider groups")
    if train.datetime.max() >= test.datetime.min():
        raise ValueError("Training overlaps test period")
    if train.datetime.max() != pd.Timestamp(config["train_period"][1]):
        raise ValueError("Training end differs from team configuration")
    if test.datetime.min() != pd.Timestamp(config["test_period"][0]):
        raise ValueError("Test start differs from team configuration")
    return train, test, config


def build_model(kind, config):
    """Fit all learned preprocessing inside each CV training fold."""
    columns = config["linear_features"]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), columns["numeric_scaled"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columns["one_hot"]),
        ("other", "passthrough", columns["passthrough"]),
    ], verbose_feature_names_out=False)
    regressor = Ridge() if kind == "Ridge" else Lasso(max_iter=100000, tol=1e-4)
    pipeline = Pipeline([("preprocess", preprocessor), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p,
                                      inverse_func=to_counts, check_inverse=False)


def save_diagnostics(kind, target, actual, predicted, dates, search, out):
    cv = pd.DataFrame(search.cv_results_)
    alpha = cv["param_regressor__model__alpha"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 4))
    for fold in range(search.n_splits_):
        ax.semilogx(alpha, -cv[f"split{fold}_test_score"], alpha=.35, label=f"Fold {fold + 1}")
    ax.semilogx(alpha, -cv.mean_test_score, color="black", linewidth=2, label="Mean validation RMSE")
    ax.axvline(search.best_params_["regressor__model__alpha"], color="red", linestyle=":")
    ax.set(xlabel="Regularization strength (alpha)", ylabel="Validation RMSE (rentals/hour)", title=f"{kind}: {target}")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / f"{kind.lower()}_{target}_tuning.png", dpi=140); plt.close(fig)

    residual = np.asarray(actual) - predicted
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].scatter(actual, predicted, s=4, alpha=.25)
    limit = max(np.max(actual), np.max(predicted))
    axes[0].plot([0, limit], [0, limit], color="black", linestyle="--")
    axes[0].set(xlabel="Actual rentals/hour", ylabel="Predicted rentals/hour")
    axes[1].scatter(predicted, residual, s=4, alpha=.25)
    axes[1].axhline(0, color="black", linestyle="--")
    axes[1].set(xlabel="Predicted rentals/hour", ylabel="Actual minus predicted")
    daily = pd.DataFrame({"date": dates, "actual": np.asarray(actual), "predicted": predicted}).set_index("date").resample("D").mean()
    daily.plot(ax=axes[2]); axes[2].set(ylabel="Mean rentals/hour", xlabel="Test date")
    axes[2].tick_params(axis="x", rotation=45)
    fig.suptitle(f"{kind}: {target} - held-out diagnostics")
    fig.tight_layout(); fig.savefig(out / f"{kind.lower()}_{target}_diagnostics.png", dpi=140); plt.close(fig)


def run(output_dir=OUTPUT):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train, test, config = load_data()
    features = sum(config["linear_features"].values(), [])
    x_train, x_test = train[features], test[features]
    cv = TimeSeriesSplit(n_splits=5)
    folds = []
    for fold, (fit, val) in enumerate(cv.split(train), start=1):
        folds.append({"fold": fold, "fit_rows": len(fit), "validation_rows": len(val),
                      "fit_start": str(train.datetime.iloc[fit[0]]), "fit_end": str(train.datetime.iloc[fit[-1]]),
                      "validation_start": str(train.datetime.iloc[val[0]]), "validation_end": str(train.datetime.iloc[val[-1]])})
    pd.DataFrame(folds).to_csv(out / "cv_folds.csv", index=False)
    rows, tuning, coefficients, selected = [], [], [], []
    predictions = test[["datetime", *TARGETS]].copy()
    for kind, alphas in ALPHAS.items():
        for target in TARGETS:
            print(f"Tuning {kind} for {target} ({len(alphas)} alphas, 5 chronological folds)", flush=True)
            search = GridSearchCV(build_model(kind, config),
                {"regressor__model__alpha": alphas}, cv=cv,
                scoring="neg_root_mean_squared_error", n_jobs=1, error_score="raise", return_train_score=True)
            # Do not silently accept a nonconverged Lasso solution.
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                search.fit(x_train, train[target])
            model = search.best_estimator_
            predicted = model.predict(x_test)
            predictions[f"{kind.lower()}_{target}"] = predicted
            alpha = float(search.best_params_["regressor__model__alpha"])
            row = {"model": kind, "target": target, "alpha": alpha,
                   "CV_RMSE": -search.best_score_, **metrics(test[target], predicted)}
            row["train_RMSE"] = metrics(train[target], model.predict(x_train))["RMSE"]
            rows.append(row)
            selected.append({"model": kind, "target": target, "alpha": alpha,
                             "at_grid_boundary": alpha in (float(alphas[0]), float(alphas[-1]))})
            results = pd.DataFrame(search.cv_results_)
            results.insert(0, "target", target); results.insert(0, "model", kind)
            tuning.append(results.drop(columns="params"))
            pipe = model.regressor_
            names = pipe.named_steps["preprocess"].get_feature_names_out()
            coef = pipe.named_steps["model"].coef_
            coefficients.extend({"model": kind, "target": target, "feature": name, "coefficient_log_scale": float(value)}
                                for name, value in zip(names, coef))
            joblib.dump(model, out / f"{kind.lower()}_{target}.joblib")
            save_diagnostics(kind, target, test[target], predicted, test.datetime, search, out)
        summed = predictions[f"{kind.lower()}_casual"] + predictions[f"{kind.lower()}_registered"]
        predictions[f"{kind.lower()}_summed_total"] = summed
        rows.append({"model": kind, "target": "summed_total", **metrics(test.cnt, summed)})
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "metrics.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(out / "cv_results.csv", index=False)
    pd.DataFrame(coefficients).to_csv(out / "coefficients.csv", index=False)
    predictions.to_csv(out / "test_predictions.csv", index=False)
    files = [ROOT / "data/processed" / name for name in ("train.csv", "test.csv", "feature_config.json")]
    manifest = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "sklearn": sklearn.__version__, "selection_metric": "mean expanding-window validation RMSE in original counts",
                "target_transform": config["target_transform"], "models": selected,
                "data_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(scores.to_string(index=False), flush=True)
    return scores


if __name__ == "__main__":
    run()
