import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neural_network import MLPClassifier

from src import config, evaluation


def build_mlp(**params):
    return MLPClassifier(**{**config.MLP_FIXED_PARAMS, **params})


def build_rf(**params):
    return RandomForestClassifier(**{**config.RF_FIXED_PARAMS, **params})


def tune_hyperparameters(builder, param_grid, X_fit, y_fit, X_val, y_val):
    metric = config.SELECTION_METRIC
    grid = list(ParameterGrid(param_grid))
    rows = []

    for i, params in enumerate(grid, start=1):
        model = builder(**params)

        start = time.perf_counter()
        model.fit(X_fit, y_fit)
        fit_time = time.perf_counter() - start

        val_metrics = evaluation.compute_metrics(y_val, model.predict(X_val))
        row = {
            **params,
            "train_accuracy": model.score(X_fit, y_fit),
            **{f"val_{k}": v for k, v in val_metrics.items()},
            "fit_time_s": fit_time,
        }
        if hasattr(model, "n_iter_"):          # MLP: epochs until early stop
            row["n_iter"] = model.n_iter_
        rows.append(row)

        print(
            f"  [{i}/{len(grid)}] {params} -> "
            f"val_{metric}={val_metrics[metric]:.4f} ({fit_time:.1f}s)"
        )

    best_index = max(range(len(rows)), key=lambda k: rows[k][f"val_{metric}"])
    best_params = grid[best_index]

    results = pd.DataFrame(rows).sort_values(f"val_{metric}", ascending=False)
    return best_params, results.reset_index(drop=True)


def train_and_evaluate(builder, params, X_train, y_train, X_test, y_test):
    model = builder(**params)

    start = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - start

    y_pred = model.predict(X_test)
    metrics = evaluation.compute_metrics(y_test, y_pred)
    metrics["fit_time_s"] = fit_time
    cm = evaluation.get_confusion_matrix(y_test, y_pred)
    return model, metrics, cm


if __name__ == "__main__":
    # Smoke test with a tiny grid (LoL draft_only): python -m src.models
    from src import preprocessing
    print("testing")
    X_train, X_test, y_train, y_test = preprocessing.prepare_lol("draft_only")
    X_fit, X_val, y_fit, y_val = preprocessing.make_validation_split(X_train, y_train)

    print("Majority baseline:", evaluation.majority_baseline(X_train, y_train, X_test, y_test))

    small_mlp_grid = {"hidden_layer_sizes": [(32,), (64, 32)], "learning_rate_init": [0.001]}
    small_rf_grid = {"n_estimators": [50], "max_depth": [10, None]}

    for name, builder, grid in (
        ("MLP", build_mlp, small_mlp_grid),
        ("Random Forest", build_rf, small_rf_grid),
    ):
        print(f"\n--- {name} ---")
        best, table = tune_hyperparameters(builder, grid, X_fit, y_fit, X_val, y_val)
        _, metrics, cm = train_and_evaluate(builder, best, X_train, y_train, X_test, y_test)
        print("Best params:", best)
        print("Test metrics:", {k: round(v, 4) for k, v in metrics.items()})
        print("Confusion matrix:\n", cm)