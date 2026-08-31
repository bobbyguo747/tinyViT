from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision.datasets import FashionMNIST

from dataset import FASHION_MNIST_CLASSES
from utils import resolve_project_path


EXPERIMENTS = {
    "cnn_seed42": {"label": "CNN", "color": "#2563EB"},
    "tinyvit_seed42": {"label": "TinyViT", "color": "#059669"},
    "tinyvit_no_pos_seed42": {
        "label": "TinyViT without position encoding",
        "color": "#D97706",
    },
}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def set_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "#FAFAFA",
            "axes.titleweight": "bold",
            "axes.labelcolor": "#1F2937",
            "text.color": "#1F2937",
            "savefig.bbox": "tight",
        }
    )


def plot_training_curves(output_dir: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    for run_name, style in EXPERIMENTS.items():
        rows = read_csv_rows(resolve_project_path("outputs") / run_name / "history.csv")
        epochs = [int(row["epoch"]) for row in rows]
        train_loss = [float(row["train_loss"]) for row in rows]
        val_loss = [float(row["val_loss"]) for row in rows]
        train_accuracy = [float(row["train_accuracy"]) for row in rows]
        val_accuracy = [float(row["val_accuracy"]) for row in rows]

        axes[0].plot(
            epochs,
            train_loss,
            color=style["color"],
            linewidth=2,
            label=f"{style['label']} train",
        )
        axes[0].plot(
            epochs,
            val_loss,
            color=style["color"],
            linewidth=1.8,
            linestyle="--",
            label=f"{style['label']} validation",
        )
        axes[1].plot(
            epochs,
            train_accuracy,
            color=style["color"],
            linewidth=2,
            label=f"{style['label']} train",
        )
        axes[1].plot(
            epochs,
            val_accuracy,
            color=style["color"],
            linewidth=1.8,
            linestyle="--",
            label=f"{style['label']} validation",
        )

    axes[0].set_title("Training and validation loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_xticks(range(1, 21, 2))
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_title("Training and validation accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_xticks(range(1, 21, 2))
    axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[1].legend(fontsize=8, ncol=2)

    figure.suptitle("Fashion-MNIST experiment curves (seed=42)", fontsize=15, weight="bold")
    figure.tight_layout()
    figure.savefig(output_dir / "training_curves.png", dpi=180)
    plt.close(figure)


def collect_comparison_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_name, style in EXPERIMENTS.items():
        run_dir = resolve_project_path("outputs") / run_name
        training = read_json(run_dir / "training_summary.json")
        test = read_json(run_dir / "test_metrics.json")
        rows.append(
            {
                "model": style["label"],
                "run_name": run_name,
                "parameter_count": int(test["parameter_count"]),
                "best_epoch": int(training["best_epoch"]),
                "validation_accuracy": float(training["best_validation_accuracy"]),
                "validation_macro_f1": float(training["best_validation_macro_f1"]),
                "test_accuracy": float(test["test_accuracy"]),
                "test_macro_f1": float(test["test_macro_f1"]),
                "training_seconds": float(training["elapsed_seconds"]),
                "evaluation_seconds": float(test["evaluation_seconds"]),
            }
        )
    return rows


def save_comparison_csv(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    with (output_dir / "model_comparison.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot_model_comparison(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    labels = [row["model"] for row in rows]
    accuracy = [row["test_accuracy"] for row in rows]
    macro_f1 = [row["test_macro_f1"] for row in rows]
    colors = [EXPERIMENTS[row["run_name"]]["color"] for row in rows]
    positions = np.arange(len(rows))
    width = 0.34

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    accuracy_bars = axis.bar(
        positions - width / 2,
        accuracy,
        width,
        color=colors,
        alpha=0.95,
        label="Test accuracy",
    )
    f1_bars = axis.bar(
        positions + width / 2,
        macro_f1,
        width,
        color=colors,
        alpha=0.48,
        hatch="//",
        label="Test Macro-F1",
    )
    axis.set_title("Model comparison on the independent test set")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 1.0)
    axis.set_xticks(positions, labels)
    axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axis.legend(loc="lower left")
    axis.grid(axis="x", visible=False)

    for bars in (accuracy_bars, f1_bars):
        for bar in bars:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{bar.get_height():.2%}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(output_dir / "model_comparison.png", dpi=180)
    plt.close(figure)


def read_confusion_matrix(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.reader(file))
    class_names = rows[0][1:]
    matrix = np.asarray([[int(value) for value in row[1:]] for row in rows[1:]])
    return class_names, matrix


def plot_confusion_matrices(output_dir: Path) -> None:
    for run_name, style in EXPERIMENTS.items():
        class_names, matrix = read_confusion_matrix(
            resolve_project_path("outputs") / run_name / "confusion_matrix.csv"
        )
        figure, axis = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar_kws={"label": "Sample count"},
            xticklabels=class_names,
            yticklabels=class_names,
            square=True,
            linewidths=0.3,
            linecolor="white",
            ax=axis,
        )
        axis.set_title(f"Confusion matrix: {style['label']}")
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        axis.tick_params(axis="x", rotation=40)
        axis.tick_params(axis="y", rotation=0)
        figure.tight_layout()
        figure.savefig(output_dir / f"confusion_matrix_{run_name}.png", dpi=180)
        plt.close(figure)


def choose_one_per_true_class(
    rows: list[dict[str, str]], correct: bool
) -> list[dict[str, str]]:
    """Choose the first matching item for every true class in dataset order."""
    selected: list[dict[str, str]] = []
    for class_id in range(len(FASHION_MNIST_CLASSES)):
        match = next(
            row
            for row in rows
            if int(row["true_label"]) == class_id
            and (row["correct"].lower() == "true") == correct
        )
        selected.append(match)
    return selected


def plot_sample_grid(
    output_path: Path,
    title: str,
    selected_rows: list[dict[str, str]],
    test_dataset: FashionMNIST,
) -> None:
    figure, axes = plt.subplots(2, 5, figsize=(13, 7.4))
    for axis, row in zip(axes.flat, selected_rows, strict=True):
        sample_index = int(row["sample_index"])
        image, _ = test_dataset[sample_index]
        axis.imshow(np.asarray(image), cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"True: {row['true_class']}\nPred: {row['predicted_class']}",
            fontsize=9,
            color="#166534" if row["correct"].lower() == "true" else "#B91C1C",
        )
        axis.axis("off")

    figure.suptitle(title, fontsize=15, weight="bold")
    figure.subplots_adjust(top=0.86, bottom=0.04, hspace=0.42, wspace=0.08)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_tinyvit_samples(output_dir: Path) -> None:
    predictions = read_csv_rows(
        resolve_project_path("outputs") / "tinyvit_seed42" / "test_predictions.csv"
    )
    test_dataset = FashionMNIST(
        resolve_project_path("data"), train=False, download=False
    )
    correct_rows = choose_one_per_true_class(predictions, correct=True)
    incorrect_rows = choose_one_per_true_class(predictions, correct=False)
    plot_sample_grid(
        output_dir / "tinyvit_correct_samples.png",
        "TinyViT: one correct prediction per true class",
        correct_rows,
        test_dataset,
    )
    plot_sample_grid(
        output_dir / "tinyvit_misclassified_samples.png",
        "TinyViT: one misclassification per true class",
        incorrect_rows,
        test_dataset,
    )


def main() -> None:
    set_plot_style()
    output_dir = resolve_project_path("outputs") / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparison_rows = collect_comparison_rows()
    save_comparison_csv(output_dir, comparison_rows)
    plot_training_curves(output_dir)
    plot_model_comparison(output_dir, comparison_rows)
    plot_confusion_matrices(output_dir)
    plot_tinyvit_samples(output_dir)

    created = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    print("created files:")
    for filename in created:
        print(f"  {filename}")


if __name__ == "__main__":
    main()
