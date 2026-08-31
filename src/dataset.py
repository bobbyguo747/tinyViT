from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import FashionMNIST

from utils import load_config, resolve_project_path, seed_worker, set_seed


FASHION_MNIST_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


@dataclass(frozen=True)
class DataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    class_names: tuple[str, ...] = FASHION_MNIST_CLASSES


def build_transform(mean: float, std: float) -> transforms.Compose:
    """Convert a grayscale image to a normalized [1, 28, 28] tensor."""
    if std <= 0:
        raise ValueError("data.std must be positive")
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((mean,), (std,)),
        ]
    )


def stratified_split_indices(
    targets: torch.Tensor | list[int], val_size: int, seed: int
) -> tuple[list[int], list[int]]:
    """Create a deterministic class-balanced train/validation split."""
    labels = np.asarray(targets, dtype=np.int64)
    if not 0 < val_size < len(labels):
        raise ValueError(f"val_size must be between 1 and {len(labels) - 1}")

    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(val_size, len(classes))
    train_indices: list[int] = []
    val_indices: list[int] = []

    for position, class_id in enumerate(classes):
        class_indices = np.flatnonzero(labels == class_id)
        rng.shuffle(class_indices)
        class_val_count = base_count + int(position < remainder)
        if class_val_count >= len(class_indices):
            raise ValueError(f"Validation split consumes class {class_id}")
        val_indices.extend(class_indices[:class_val_count].tolist())
        train_indices.extend(class_indices[class_val_count:].tolist())

    # Shuffle the combined lists so samples are not grouped by class.
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    if set(train_indices).intersection(val_indices):
        raise RuntimeError("Train and validation indices overlap")
    if len(train_indices) + len(val_indices) != len(labels):
        raise RuntimeError("Split does not cover the complete training set")
    return train_indices, val_indices


def stratified_subset_indices(
    targets: torch.Tensor | list[int],
    candidate_indices: tuple[int, ...] | list[int],
    sample_count: int,
    seed: int,
) -> list[int]:
    """Select a deterministic, class-balanced subset from candidate indices."""
    labels = np.asarray(targets, dtype=np.int64)
    candidates = np.asarray(candidate_indices, dtype=np.int64)
    if not 0 < sample_count <= len(candidates):
        raise ValueError(f"sample_count must be between 1 and {len(candidates)}")

    candidate_labels = labels[candidates]
    classes = np.unique(candidate_labels)
    rng = np.random.default_rng(seed)
    base_count, remainder = divmod(sample_count, len(classes))
    selected: list[int] = []

    for position, class_id in enumerate(classes):
        class_candidates = candidates[candidate_labels == class_id].copy()
        rng.shuffle(class_candidates)
        class_count = base_count + int(position < remainder)
        if class_count > len(class_candidates):
            raise ValueError(f"Not enough candidates for class {class_id}")
        selected.extend(class_candidates[:class_count].tolist())

    rng.shuffle(selected)
    if len(selected) != sample_count or len(set(selected)) != sample_count:
        raise RuntimeError("Stratified subset selection produced invalid indices")
    return selected


def build_dataloaders(config: dict[str, Any], download: bool = True) -> DataBundle:
    """Build strict train/validation/test DataLoaders from one config."""
    seed = int(config["seed"])
    data_config = config["data"]
    set_seed(seed)

    data_root = resolve_project_path(data_config["root"])
    data_root.mkdir(parents=True, exist_ok=True)
    transform = build_transform(
        mean=float(data_config["mean"]),
        std=float(data_config["std"]),
    )

    # Two training-set objects let train/validation transforms diverge later
    # without changing the split. At present both intentionally use no augmentation.
    train_full = FashionMNIST(data_root, train=True, transform=transform, download=download)
    val_full = FashionMNIST(data_root, train=True, transform=transform, download=download)
    test_dataset = FashionMNIST(data_root, train=False, transform=transform, download=download)

    train_indices, val_indices = stratified_split_indices(
        train_full.targets,
        val_size=int(data_config["val_size"]),
        seed=seed,
    )
    train_dataset = Subset(train_full, train_indices)
    val_dataset = Subset(val_full, val_indices)

    batch_size = int(data_config["batch_size"])
    num_workers = int(data_config["num_workers"])
    pin_memory = bool(data_config["pin_memory"] and torch.cuda.is_available())
    generator = torch.Generator().manual_seed(seed)
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    test_loader = DataLoader(test_dataset, shuffle=False, **common_loader_args)

    return DataBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        train_indices=tuple(train_indices),
        val_indices=tuple(val_indices),
    )


def build_test_loader(config: dict[str, Any], download: bool = True) -> DataLoader:
    """Build only the untouched official test loader for independent evaluation."""
    seed = int(config["seed"])
    data_config = config["data"]
    set_seed(seed)
    transform = build_transform(
        mean=float(data_config["mean"]),
        std=float(data_config["std"]),
    )
    test_dataset = FashionMNIST(
        resolve_project_path(data_config["root"]),
        train=False,
        transform=transform,
        download=download,
    )
    num_workers = int(data_config["num_workers"])
    return DataLoader(
        test_dataset,
        batch_size=int(data_config["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(data_config["pin_memory"] and torch.cuda.is_available()),
        worker_init_fn=seed_worker,
        persistent_workers=num_workers > 0,
    )


def label_counts(dataset: FashionMNIST, indices: tuple[int, ...]) -> dict[int, int]:
    """Count labels for a subset without loading image tensors."""
    labels = dataset.targets[list(indices)].tolist()
    return dict(sorted(Counter(labels).items()))


def run_data_check(config_path: str | Path) -> None:
    """Download data and print split/batch facts used by later stages."""
    config = load_config(config_path)
    bundle = build_dataloaders(config, download=True)
    images, labels = next(iter(bundle.train_loader))
    train_base = bundle.train_loader.dataset.dataset
    val_base = bundle.val_loader.dataset.dataset

    print(f"train samples: {len(bundle.train_loader.dataset)}")
    print(f"validation samples: {len(bundle.val_loader.dataset)}")
    print(f"test samples: {len(bundle.test_loader.dataset)}")
    print(f"train label counts: {label_counts(train_base, bundle.train_indices)}")
    print(f"validation label counts: {label_counts(val_base, bundle.val_indices)}")
    print(f"batch images: {tuple(images.shape)} dtype={images.dtype}")
    print(f"batch labels: {tuple(labels.shape)} dtype={labels.dtype}")
    print(f"pixel range after normalization: [{images.min():.3f}, {images.max():.3f}]")
    print(f"train/validation overlap: {len(set(bundle.train_indices) & set(bundle.val_indices))}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Fashion-MNIST data loading and split")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_data_check(args.config)
