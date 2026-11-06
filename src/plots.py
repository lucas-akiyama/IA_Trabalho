from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: figures are only saved to disk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _save(fig, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_mlp_hyperparameters(results, title, output_path):
    df = results.copy()
    df["gap"] = df["train_accuracy"] - df["val_accuracy"]

    architectures = sorted(
        dict.fromkeys(df["hidden_layer_sizes"]), key=lambda t: (len(t), sum(t))
    )
    labels = ["-".join(map(str, a)) for a in architectures]
    df["architecture"] = df["hidden_layer_sizes"].map(lambda t: "-".join(map(str, t)))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, column, name, cmap in (
        (axes[0], "val_accuracy", "Validation accuracy", "viridis"),
        (axes[1], "gap", "Train - validation accuracy", "magma"),
    ):
        table = df.pivot_table(
            index="architecture",
            columns="learning_rate_init",
            values=column,
            aggfunc="mean",
        ).reindex(labels)
        sns.heatmap(table, annot=True, fmt=".3f", cmap=cmap, ax=ax)
        ax.set_title(name)
        ax.set_xlabel("Initial learning rate")
        ax.set_ylabel("Hidden layers")
    fig.suptitle(title)
    _save(fig, output_path)


def _depth_label(value):
    return "None" if pd.isna(value) else str(int(value))


def plot_rf_hyperparameters(results, title, output_path):
    df = results.copy()
    df["depth_label"] = df["max_depth"].map(_depth_label)
    depths = sorted(df["depth_label"].unique(), key=lambda s: 10**6 if s == "None" else int(s))

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for depth in depths:
        sub = df[df["depth_label"] == depth].sort_values("n_estimators")
        (line,) = ax.plot(
            sub["n_estimators"], sub["val_accuracy"], marker="o", label=f"depth={depth}"
        )
        ax.plot(
            sub["n_estimators"], sub["train_accuracy"],
            linestyle="--", color=line.get_color(), alpha=0.6,
        )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Accuracy (solid: validation, dashed: train)")
    ax.set_title(title)
    ax.legend()
    _save(fig, output_path)


def plot_model_comparison(summary, output_path):
    scenarios = list(dict.fromkeys(summary["scenario"]))
    model_names = ["baseline", "mlp", "rf"]
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k, model in enumerate(model_names):
        means, stds = [], []
        for scenario in scenarios:
            row = summary[(summary["scenario"] == scenario) & (summary["model"] == model)]
            means.append(row["accuracy_mean"].iloc[0] if len(row) else np.nan)
            stds.append(row["accuracy_std"].iloc[0] if len(row) else 0.0)
        positions = np.arange(len(scenarios)) + (k - 1) * width
        bars = ax.bar(positions, means, width, yerr=stds, capsize=3, label=model.upper())
        for bar, value in zip(bars, means):
            if not np.isnan(value):
                ax.text(bar.get_x() + bar.get_width() / 2, value + 0.005,
                        f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels(scenarios)
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy: baseline vs. MLP vs. Random Forest")
    ax.legend()
    _save(fig, output_path)