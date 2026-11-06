from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

CLASS_LABELS = [0, 1]


def compute_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=CLASS_LABELS)


def majority_baseline(X_train, y_train, X_test, y_test):
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    return compute_metrics(y_test, dummy.predict(X_test))


def plot_confusion_matrix(cm, title, output_path, class_names=("Class 0", "Class 1")):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    row_pct = cm / cm.sum(axis=1, keepdims=True) * 100
    annotations = [
        [f"{cm[i, j]}\n({row_pct[i, j]:.1f}%)" for j in range(2)] for i in range(2)
    ]

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=annotations,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)