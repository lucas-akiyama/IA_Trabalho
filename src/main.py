import argparse
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid

from src import config, evaluation, models, plots, preprocessing

SCENARIOS = {
    "dota2": preprocessing.prepare_dota2,
    "lol_draft_only": lambda: preprocessing.prepare_lol(preprocessing.SCENARIO_DRAFT_ONLY),
    "lol_draft_ingame": lambda: preprocessing.prepare_lol(preprocessing.SCENARIO_DRAFT_INGAME),
}

METRIC_KEYS = ["accuracy", "precision", "recall", "f1", "fit_time_s"]


def accuracy_ci95(accuracy, n_samples):
    return 1.96 * np.sqrt(accuracy * (1.0 - accuracy) / n_samples)


def _summary_row(scenario, model, best_params, per_seed, n_test):
    row = {"scenario": scenario, "model": model, "best_params": str(best_params)}
    for key in METRIC_KEYS:
        row[f"{key}_mean"] = per_seed[key].mean()
        row[f"{key}_std"] = per_seed[key].std(ddof=0)
    row["accuracy_ci95"] = accuracy_ci95(row["accuracy_mean"], n_test)
    row["n_test"] = n_test
    return row


def run_scenario(name, quick=False):
    print(f"\n{'=' * 70}\nScenario: {name}\n{'=' * 70}")
    start = time.perf_counter()

    X_train, X_test, y_train, y_test = SCENARIOS[name]()
    X_fit, X_val, y_fit, y_val = preprocessing.make_validation_split(X_train, y_train)
    print(f"X_train={X_train.shape} | X_fit={X_fit.shape} | "
          f"X_val={X_val.shape} | X_test={X_test.shape}")

    seeds = config.QUICK_SEEDS if quick else config.SEEDS
    setups = {
        "mlp": (
            models.build_mlp,
            config.QUICK_MLP_PARAM_GRID if quick else config.MLP_PARAM_GRID,
        ),
        "rf": (
            models.build_rf,
            config.QUICK_RF_PARAM_GRID if quick else config.RF_PARAM_GRID,
        ),
    }

    summary_rows = []

    # Majority-class baseline (deterministic -> std = 0)
    base = evaluation.majority_baseline(X_train, y_train, X_test, y_test)
    base_df = pd.DataFrame([{**base, "fit_time_s": 0.0}])
    summary_rows.append(_summary_row(name, "baseline", "-", base_df, len(y_test)))
    print(f"Baseline accuracy: {base['accuracy']:.4f}")

    for model_key, (builder, grid) in setups.items():
        print(f"\n--- {model_key.upper()}: tuning ({len(ParameterGrid(grid))} combinations) ---")
        best_params, table = models.tune_hyperparameters(
            builder, grid, X_fit, y_fit, X_val, y_val
        )
        print(f"Best params: {best_params}")
        table.to_csv(config.TABLES_DIR / f"{name}_{model_key}_tuning.csv", index=False)

        if model_key == "mlp":
            plots.plot_mlp_hyperparameters(
                table, f"MLP hyperparameters - {name}",
                config.FIGURES_DIR / f"{name}_mlp_hyperparameters.png",
            )
        else:
            plots.plot_rf_hyperparameters(
                table, f"Random Forest hyperparameters - {name}",
                config.FIGURES_DIR / f"{name}_rf_hyperparameters.png",
            )

        # Final evaluation: refit on the full training set, once per seed
        per_seed_rows = []
        for i, seed in enumerate(seeds):
            params = {**best_params, "random_state": seed}
            _, metrics, cm = models.train_and_evaluate(
                builder, params, X_train, y_train, X_test, y_test
            )
            per_seed_rows.append({"seed": seed, **metrics})
            print(f"  seed={seed}: acc={metrics['accuracy']:.4f} "
                  f"f1={metrics['f1']:.4f} ({metrics['fit_time_s']:.1f}s)")

            if i == 0:  # confusion matrix of the reference seed
                np.savetxt(config.TABLES_DIR / f"{name}_{model_key}_confusion.csv",
                           cm, fmt="%d", delimiter=",")
                evaluation.plot_confusion_matrix(
                    cm, f"{model_key.upper()} - {name} (seed {seed})",
                    config.FIGURES_DIR / f"{name}_{model_key}_confusion.png",
                    class_names=config.CLASS_NAMES[name],
                )

        per_seed = pd.DataFrame(per_seed_rows)
        per_seed.to_csv(config.TABLES_DIR / f"{name}_{model_key}_seeds.csv", index=False)
        summary_rows.append(
            _summary_row(name, model_key, best_params, per_seed, len(y_test))
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(config.TABLES_DIR / f"{name}_summary.csv", index=False)
    print(f"\n{summary[['model', 'accuracy_mean', 'accuracy_std', 'f1_mean']].round(4)}")
    print(f"Scenario '{name}' finished in {(time.perf_counter() - start) / 60:.1f} min")


def combine_summaries():
    frames = []
    for name in SCENARIOS:  # fixed order
        path = config.TABLES_DIR / f"{name}_summary.csv"
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        return
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(config.TABLES_DIR / "summary.csv", index=False)
    plots.plot_model_comparison(summary, config.FIGURES_DIR / "model_comparison.png")
    print(f"\nCombined summary saved to {config.TABLES_DIR / 'summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="MLP vs Random Forest experiments")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=list(SCENARIOS), help="scenarios to run")
    parser.add_argument("--quick", action="store_true",
                        help="reduced grids and seeds for a fast check")
    args = parser.parse_args()

    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for name in args.scenarios:
        run_scenario(name, quick=args.quick)
    combine_summaries()


if __name__ == "__main__":
    main()