from __future__ import annotations

import argparse
import copy
import csv
import json
import tempfile
from pathlib import Path

import matplotlib
import sklearn
import torch
import torchvision
import yaml
from PIL import Image

from dataset import build_dataloaders
from evaluate import load_formal_checkpoint
from model import build_model, build_model_from_saved_config
from utils import PROJECT_ROOT, load_config


REQUIRED_FILES = (
    "README.md",
    "EXPERIMENT_ANALYSIS.md",
    "requirements.txt",
    "config.yaml",
    "src/model.py",
    "src/dataset.py",
    "src/train.py",
    "src/evaluate.py",
    "src/predict.py",
    "src/inspect_model.py",
    "src/plot_results.py",
    "src/utils.py",
)
RUN_NAMES = ("tinyvit_seed42", "tinyvit_no_pos_seed42", "cnn_seed42")
EXPECTED_PARAMETERS = {
    "tinyvit": 105_098,
    "tinyvit_no_pos": 101_898,
    "cnn": 96_362,
}


def pass_check(message: str) -> None:
    print(f"[PASS] {message}")


def verify_required_files() -> None:
    missing = [path for path in REQUIRED_FILES if not (PROJECT_ROOT / path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    pass_check(f"{len(REQUIRED_FILES)} required project files exist")


def verify_data(config: dict, fresh_download: bool) -> None:
    if fresh_download:
        with tempfile.TemporaryDirectory(prefix="tinyvit_fashion_mnist_") as temp_dir:
            temporary_config = copy.deepcopy(config)
            temporary_config["data"]["root"] = temp_dir
            bundle = build_dataloaders(temporary_config, download=True)
            verify_bundle(bundle)
        pass_check("Fashion-MNIST fresh download and temporary cleanup")
    else:
        bundle = build_dataloaders(config, download=True)
        verify_bundle(bundle)


def verify_bundle(bundle) -> None:
    sizes = (
        len(bundle.train_loader.dataset),
        len(bundle.val_loader.dataset),
        len(bundle.test_loader.dataset),
    )
    if sizes != (54_000, 6_000, 10_000):
        raise AssertionError(f"Unexpected data split sizes: {sizes}")
    if set(bundle.train_indices).intersection(bundle.val_indices):
        raise AssertionError("Train and validation indices overlap")
    images, labels = next(iter(bundle.train_loader))
    if images.shape != (128, 1, 28, 28) or labels.shape != (128,):
        raise AssertionError(f"Unexpected batch shapes: {images.shape}, {labels.shape}")
    pass_check("data split 54000/6000/10000, zero overlap, batch [128,1,28,28]")


def verify_models(config: dict) -> None:
    test_input = torch.zeros(2, 1, 28, 28)
    for model_name, expected_count in EXPECTED_PARAMETERS.items():
        model = build_model(model_name, config["model"])
        model.eval()
        with torch.no_grad():
            logits = model(test_input)
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if logits.shape != (2, 10):
            raise AssertionError(f"{model_name} logits shape is {logits.shape}")
        if parameter_count != expected_count:
            raise AssertionError(
                f"{model_name} parameter count {parameter_count} != {expected_count}"
            )
        pass_check(f"{model_name}: logits [2,10], parameters {parameter_count:,}")


def verify_checkpoints_and_results() -> None:
    for run_name in RUN_NAMES:
        checkpoint_path = PROJECT_ROOT / "checkpoints" / run_name / "best.pt"
        checkpoint = load_formal_checkpoint(checkpoint_path)
        model_name = checkpoint.get("model_name", "tinyvit")
        model = build_model_from_saved_config(model_name, checkpoint["model_config"])
        model.load_state_dict(checkpoint["model_state"], strict=True)

        output_dir = PROJECT_ROOT / "outputs" / run_name
        metrics_path = output_dir / "test_metrics.json"
        predictions_path = output_dir / "test_predictions.csv"
        if not metrics_path.is_file() or not predictions_path.is_file():
            raise FileNotFoundError(f"Missing evaluation files for {run_name}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        with predictions_path.open("r", encoding="utf-8-sig", newline="") as file:
            prediction_count = sum(1 for _ in csv.DictReader(file))
        if prediction_count != 10_000:
            raise AssertionError(f"{run_name} predictions={prediction_count}")
        pass_check(
            f"{run_name}: checkpoint strict-load, 10,000 predictions, "
            f"test accuracy {metrics['test_accuracy']:.2%}"
        )


def verify_figures() -> None:
    comparison_dir = PROJECT_ROOT / "outputs" / "comparison"
    expected_figures = (
        "training_curves.png",
        "model_comparison.png",
        "confusion_matrix_cnn_seed42.png",
        "confusion_matrix_tinyvit_seed42.png",
        "confusion_matrix_tinyvit_no_pos_seed42.png",
        "tinyvit_correct_samples.png",
        "tinyvit_misclassified_samples.png",
    )
    for filename in expected_figures:
        path = comparison_dir / filename
        with Image.open(path) as image:
            image.verify()
    pass_check(f"{len(expected_figures)} result figures open successfully")


def print_environment() -> None:
    print("environment:")
    print(f"  Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    print(f"  torch: {torch.__version__}")
    print(f"  torchvision: {torchvision.__version__}")
    print(f"  CUDA runtime: {torch.version.cuda}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  capability: {torch.cuda.get_device_capability(0)}")
    print(f"  scikit-learn: {sklearn.__version__}")
    print(f"  matplotlib: {matplotlib.__version__}")
    print(f"  PyYAML: {yaml.__version__}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run non-training project verification")
    parser.add_argument(
        "--fresh-download",
        action="store_true",
        help="Download Fashion-MNIST into a temporary empty directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print_environment()
    config = load_config()
    verify_required_files()
    verify_data(config, fresh_download=args.fresh_download)
    verify_models(config)
    verify_checkpoints_and_results()
    verify_figures()
    print("PROJECT VERIFICATION PASSED")


if __name__ == "__main__":
    main()
