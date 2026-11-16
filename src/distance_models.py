"""Kiersten Cheathams's distance models: KNN and  SVR
    Run one model at a time from the project root:
    python -m src.distance_models knn
    python -m src.distance_models svr

Each run tunes the model for casual, registered and total (cnt) rentals, scores it on
the October-December 2012 test hours, and writes results to results/distance_models/.
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from .evaluate import log_result, metrics, plot_diagnostics, to_counts
from .linear_models import TARGETS, load_data  

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "distance_models"

GRIDS = {
    "knn": {"n_neighbors": [1, 2, 3, 5, 10, 20, 40, 80], "weights": ["uniform", "distance"]},
    "svr": {"C": [0.1, 1, 3, 10]},
}

NAMES = {"knn": "KNN", "svr": "SVR"}


def build_model(kind, config):
    """Fit all learned preprocessing inside each CV training fold."""   #CHANGE ** DO I NEED CV ??
    columns = config["linear_features"]
    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), columns["numeric_scaled"]),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), columns["one_hot"]),
        ("other", "passthrough", columns["passthrough"]),
    ], verbose_feature_names_out=False)
    regressor = KNeighborsRegressor() if kind == "knn" else SVR(kernel="rbf")
    pipeline = Pipeline([("preprocess", preprocessor), ("model", regressor)])
    return TransformedTargetRegressor(regressor=pipeline, func=np.log1p,
                                      inverse_func=to_counts, check_inverse=False)


def save_tuning_plot(kind, target, search, out):
    """Validation RMSE for every setting tried, so the report can justify the choice."""
    cv = pd.DataFrame(search.cv_results_)
    names = list(GRIDS[kind])
    x_col = f"param_regressor__model__{names[0]}"
    fig, ax = plt.subplots(figsize=(8, 4))
    if len(names) > 1:
        group_col = f"param_regressor__model__{names[1]}"
        for value, group in cv.groupby(group_col):
            ax.plot(group[x_col].astype(float), -group.mean_test_score, marker="o", label=f"{names[1]}={value}")
        ax.legend(fontsize=8)
    else:
        ax.plot(cv[x_col].astype(float), -cv.mean_test_score, marker="o", color="black")
    ax.set_xscale("log")
    ax.set(xlabel=names[0], ylabel="Mean validation RMSE (rentals/hour)",
           title=f"{NAMES[kind]}: {target}. Best: {search.best_params_}")
    fig.tight_layout()
    fig.savefig(out / f"{kind}_{target}_tuning.png", dpi=140)
    plt.close(fig)

def run(kind, output_dir=OUTPUT):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    train, test, config = load_data()
    features = sum(config["linear_features"].values(), [])     
    x_train, x_test = train[features], test[features]
    cv = TimeSeriesSplit(n_splits=5)
    grid = {f"regressor__model__{name}": values for name, values in GRIDS[kind].items()}

    # Keep other models' columns if the file already exists (e.g. KNN, then SVR).
    pred_path = out / "test_predictions.csv"
    if pred_path.exists():
        predictions = pd.read_csv(pred_path)
        predictions = predictions[["datetime", *TARGETS, *[c for c in predictions.columns if not c.startswith(f"{kind}_") and c not in ("datetime", *TARGETS)]]]
    else:
        predictions = test[["datetime", *TARGETS]].copy()
    predictions["datetime"] = pd.to_datetime(predictions["datetime"])

    rows, tuning = [], []
    for target in TARGETS:
        start = time.time()
        print(f"Tuning {NAMES[kind]} for {target}...", flush=True)
        search = GridSearchCV(build_model(kind, config), grid, cv=cv,
                              scoring="neg_root_mean_squared_error",  # RMSE in rental counts
                              n_jobs=-1, error_score="raise", return_train_score=True)
        search.fit(x_train, train[target])
        predicted = search.best_estimator_.predict(x_test)  # already in counts
        predictions[f"{kind}_{target}"] = predicted

        best = {k.replace("regressor__model__", ""): v for k, v in search.best_params_.items()}
        rows.append({"model": NAMES[kind], "target": target, **best,
                     "CV_RMSE": -search.best_score_, **metrics(test[target], predicted),
                     "train_RMSE": metrics(train[target], search.best_estimator_.predict(x_train))["RMSE"],
                     "seconds": round(time.time() - start, 1)})
        results = pd.DataFrame(search.cv_results_)
        results.insert(0, "target", target)
        results.insert(0, "model", NAMES[kind])
        tuning.append(results.drop(columns="params"))

        save_tuning_plot(kind, target, search, out)
        fig = plot_diagnostics(test.datetime, test[target], predicted, f"{NAMES[kind]}: {target}")
        fig.savefig(out / f"{kind}_{target}_diagnostics.png", dpi=140)
        plt.close(fig)
        # Shared team log (Malachi's format): one line per model and target.
        log_result(out / "results.jsonl", model=NAMES[kind], target=target, actual=test[target],
                   predicted=predicted, run_id=f"{kind}-{target}-{time.strftime('%Y%m%d-%H%M%S')}",
                   metadata={"best_params": best, "target_scale": "counts", "cv": "TimeSeriesSplit(5)"})

    # Does adding casual + registered predictions beat predicting total directly?
    summed = predictions[f"{kind}_casual"] + predictions[f"{kind}_registered"]
    predictions[f"{kind}_summed_total"] = summed
    rows.append({"model": NAMES[kind], "target": "summed_total", **metrics(test.cnt, summed)})

    scores = pd.DataFrame(rows)
    scores.to_csv(out / f"{kind}_metrics.csv", index=False)
    pd.concat(tuning, ignore_index=True).to_csv(out / f"{kind}_cv_results.csv", index=False)
    predictions.to_csv(pred_path, index=False)
    print(scores.round(3).to_string(index=False), flush=True)
    return scores


if __name__ == "__main__":
    choice = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if choice not in GRIDS:
        sys.exit(f"Usage: python -m src.distance_models [{'|'.join(GRIDS)}]")
    warnings.filterwarnings("ignore", category=UserWarning)
    run(choice)