import argparse
import ast
import math

import numpy as np
import pandas as pd

from src import config, models
from src.main import SCENARIOS

N_BOOTSTRAP = 2000
EXACT_THRESHOLD = 25   # use the exact binomial test when b + c is below this


def mcnemar_test(correct_a, correct_b):
    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))
    n = b + c
    if n == 0:
        return b, c, 1.0
    if n < EXACT_THRESHOLD:
        k = min(b, c)
        tail = sum(math.comb(n, i) for i in range(k + 1))
        return b, c, min(1.0, 2.0 * tail / 2 ** n)
    chi2 = (abs(b - c) - 1) ** 2 / n
    return b, c, math.erfc(math.sqrt(chi2 / 2.0))


def paired_bootstrap_ci(correct_a, correct_b, n_boot=N_BOOTSTRAP, seed=config.RANDOM_STATE):
    diff = correct_a.astype(float) - correct_b.astype(float)
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _load_best_params(scenario):
    path = config.TABLES_DIR / f"{scenario}_summary.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.main` first.")
    summary = pd.read_csv(path).set_index("model")
    return {
        "mlp": ast.literal_eval(summary.loc["mlp", "best_params"]),
        "rf": ast.literal_eval(summary.loc["rf", "best_params"]),
    }


def run_scenario(scenario):
    print(f"\n=== {scenario} ===")
    X_train, X_test, y_train, y_test = SCENARIOS[scenario]()
    best = _load_best_params(scenario)
    y_true = y_test.to_numpy()

    rows = []
    for seed in config.SEEDS:
        correct = {}
        for key, builder in (("mlp", models.build_mlp), ("rf", models.build_rf)):
            model = builder(**{**best[key], "random_state": seed})
            model.fit(X_train, y_train)
            correct[key] = model.predict(X_test) == y_true

        b, c, p_value = mcnemar_test(correct["mlp"], correct["rf"])
        low, high = paired_bootstrap_ci(correct["mlp"], correct["rf"])
        rows.append({
            "scenario": scenario,
            "seed": seed,
            "acc_mlp": correct["mlp"].mean(),
            "acc_rf": correct["rf"].mean(),
            "diff_mlp_minus_rf": correct["mlp"].mean() - correct["rf"].mean(),
            "ci95_low": low,
            "ci95_high": high,
            "mlp_only_correct_b": b,
            "rf_only_correct_c": c,
            "mcnemar_p": p_value,
        })
        print(f"  seed={seed}: MLP={rows[-1]['acc_mlp']:.4f} RF={rows[-1]['acc_rf']:.4f} "
              f"diff={rows[-1]['diff_mlp_minus_rf'] * 100:+.2f} p.p. "
              f"CI95=[{low * 100:+.2f}, {high * 100:+.2f}] b={b} c={c} p={p_value:.4g}")
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Paired tests: MLP vs Random Forest")
    parser.add_argument("--scenarios", nargs="+", choices=list(SCENARIOS),
                        default=list(SCENARIOS))
    args = parser.parse_args()

    results = pd.concat([run_scenario(s) for s in args.scenarios], ignore_index=True)
    config.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out = config.TABLES_DIR / "significance.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
