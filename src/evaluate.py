from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from dataset import build_test_loader
from model import build_model_from_saved_config
from utils import resolve_project_path, select_device, set_seed


def load_formal_checkpoint(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load and validate a formal best.pt checkpoint."""
    path = Path(checkpoint_path)
    if not path.is_absolute():
        path = resolve_project_path(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    expected_purpose = "formal Fashion-MNIST classification experiment"
    if checkpoint.get("purpose") != expected_purpose:
        raise ValueError(
            "evaluate.py accepts only a formal best.pt checkpoint; "
            f"got purpose={checkpoint.get('purpose')!r}"
        )

    return checkpoint


@torch.no_grad()
def predict_test_set(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int], list[list[float]]]:
    """Return labels, predictions and raw logits in stable dataset order."""
    all_labels: list[int] = []
    all_predictions: list[int] = []
    all_logits: list[list[float]] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predictions = logits.argmax(dim=1)
        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.cpu().tolist())
        all_logits.extend(logits.cpu().tolist())

    return all_labels, all_predictions, all_logits


def save_predictions(
    path: Path,
    labels: list[int],
    predictions: list[int],
    logits: list[list[float]],
    class_names: list[str],
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "true_class",
                "predicted_label",
                "predicted_class",
                "correct",
                *[f"logit_{index}" for index in range(len(class_names))],
            ]
        )
        for index, (label, prediction, sample_logits) in enumerate(
            zip(labels, predictions, logits, strict=True)
        ):
            writer.writerow(
                [
                    index,
                    label,
                    class_names[label],
                    prediction,
                    class_names[prediction],
                    label == prediction,
                    *sample_logits,
                ]
            )


def save_confusion_matrix(
    path: Path, matrix: list[list[int]], class_names: list[str]
) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["true\\predicted", *class_names])
        for class_name, row in zip(class_names, matrix, strict=True):
            writer.writerow([class_name, *row])


def evaluate_checkpoint(checkpoint_path: str | Path) -> None:
    # Device and data settings come from best.pt; no external config is needed.
    checkpoint = load_formal_checkpoint(checkpoint_path)
    project_config = checkpoint["project_config"]
    set_seed(int(project_config["seed"]))
    device = select_device(project_config.get("device", "auto"))
    model_name = checkpoint.get("model_name", "tinyvit")
    model = build_model_from_saved_config(model_name, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()
    test_loader = build_test_loader(project_config, download=True)
    class_names = list(checkpoint["class_names"])

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    labels, predictions, logits = predict_test_set(model, test_loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start

    accuracy = float(accuracy_score(labels, predictions))
    macro_f1 = float(f1_score(labels, predictions, average="macro"))
    matrix = confusion_matrix(labels, predictions, labels=range(len(class_names))).tolist()
    per_class = classification_report(
        labels,
        predictions,
        labels=range(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "purpose": "independent evaluation of validation-selected best.pt",
        "run_name": checkpoint["run_name"],
        "model_name": model_name,
        "checkpoint_epoch": checkpoint["epoch"],
        "checkpoint_validation_accuracy": checkpoint["validation_accuracy"],
        "checkpoint_validation_macro_f1": checkpoint["validation_macro_f1"],
        "test_samples": len(labels),
        "test_accuracy": accuracy,
        "test_macro_f1": macro_f1,
        "evaluation_seconds": elapsed_seconds,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "device": str(device),
        "per_class_report": per_class,
    }

    output_dir = resolve_project_path("outputs") / checkpoint["run_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    save_predictions(
        output_dir / "test_predictions.csv",
        labels,
        predictions,
        logits,
        class_names,
    )
    save_confusion_matrix(output_dir / "confusion_matrix.csv", matrix, class_names)

    print(f"checkpoint: {checkpoint_path}")
    print(f"model: {model_name}")
    print(f"selected at epoch: {checkpoint['epoch']}")
    print(f"validation accuracy: {checkpoint['validation_accuracy']:.2%}")
    print(f"validation Macro-F1: {checkpoint['validation_macro_f1']:.4f}")
    print(f"test samples: {len(labels)}")
    print(f"test accuracy: {accuracy:.2%}")
    print(f"test Macro-F1: {macro_f1:.4f}")
    print(f"evaluation time: {elapsed_seconds:.2f}s")
    print(f"saved evaluation files to: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a validation-selected TinyViT best.pt on the test set"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to the formal best.pt; all model/data config is loaded from it",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate_checkpoint(parse_args().checkpoint)
