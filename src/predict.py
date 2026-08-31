from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision.datasets import FashionMNIST

from dataset import build_transform
from evaluate import load_formal_checkpoint
from model import build_model_from_saved_config
from utils import resolve_project_path, select_device, set_seed


def load_image(
    args: argparse.Namespace, checkpoint: dict[str, Any]
) -> tuple[Image.Image, torch.Tensor, str | None, str]:
    """Load either one external image or a reproducible test-set sample."""
    project_config = checkpoint["project_config"]
    data_config = project_config["data"]
    transform = build_transform(float(data_config["mean"]), float(data_config["std"]))
    class_names = list(checkpoint["class_names"])

    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_absolute():
            image_path = resolve_project_path(image_path)
        image = Image.open(image_path).convert("L").resize((28, 28))
        return image, transform(image), None, str(image_path)

    dataset = FashionMNIST(
        resolve_project_path(data_config["root"]),
        train=False,
        download=True,
    )
    if not 0 <= args.index < len(dataset):
        raise ValueError(f"--index must be between 0 and {len(dataset) - 1}")
    image, label = dataset[args.index]
    return image, transform(image), class_names[label], f"Fashion-MNIST test index {args.index}"


def save_prediction_figure(
    output_path: Path,
    image: Image.Image,
    source: str,
    true_class: str | None,
    predicted_class: str,
    confidence: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5.2, 5.4))
    axis.imshow(image, cmap="gray", vmin=0, vmax=255)
    axis.axis("off")
    true_text = true_class if true_class is not None else "unknown"
    axis.set_title(
        f"Source: {source}\nTrue: {true_text}\n"
        f"Prediction: {predicted_class} ({confidence:.2%})",
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def predict(args: argparse.Namespace) -> None:
    checkpoint = load_formal_checkpoint(args.checkpoint)
    project_config = checkpoint["project_config"]
    set_seed(int(project_config["seed"]))
    device = select_device(project_config.get("device", "auto"))
    model_name = checkpoint.get("model_name", "tinyvit")
    model = build_model_from_saved_config(model_name, checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device)
    model.eval()

    image, image_tensor, true_class, source = load_image(args, checkpoint)
    with torch.inference_mode():
        logits = model(image_tensor.unsqueeze(0).to(device))
        probabilities = logits.softmax(dim=1)[0]

    class_names = list(checkpoint["class_names"])
    top_count = min(args.top_k, len(class_names))
    top_probabilities, top_indices = probabilities.topk(top_count)
    predicted_index = int(top_indices[0].item())
    predicted_class = class_names[predicted_index]
    confidence = float(top_probabilities[0].item())

    print(f"checkpoint: {args.checkpoint}")
    print(f"model: {model_name}")
    print(f"source: {source}")
    print(f"true class: {true_class if true_class is not None else 'unknown'}")
    print(f"predicted class: {predicted_class}")
    print(f"confidence: {confidence:.2%}")
    print("top predictions:")
    for rank, (probability, class_index) in enumerate(
        zip(top_probabilities.tolist(), top_indices.tolist(), strict=True), start=1
    ):
        print(f"  {rank}. {class_names[class_index]}: {probability:.2%}")
    print("note: Softmax is applied here only to present probabilities; the model returns logits")

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = resolve_project_path(output_path)
    save_prediction_figure(
        output_path,
        image,
        source,
        true_class,
        predicted_class,
        confidence,
    )
    print(f"saved figure: {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one reproducible image prediction")
    parser.add_argument("--checkpoint", required=True, help="Formal best.pt checkpoint")
    source_group = parser.add_mutually_exclusive_group()
    source_group.add_argument("--image", help="Optional external image path")
    source_group.add_argument(
        "--index", type=int, default=0, help="Fashion-MNIST test index (default: 0)"
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", default="outputs/prediction_demo.png")
    args = parser.parse_args()
    if args.top_k <= 0:
        parser.error("--top-k must be positive")
    return args


if __name__ == "__main__":
    predict(parse_args())

