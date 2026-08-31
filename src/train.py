from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import f1_score

from dataset import FASHION_MNIST_CLASSES, build_dataloaders, stratified_subset_indices
from model import MODEL_NAMES, build_model, build_tinyvit
from utils import load_config, resolve_project_path, seed_worker, select_device, set_seed


def make_overfit_loaders(
    config: dict[str, Any], sample_count: int
) -> tuple[DataLoader, DataLoader, list[int], dict[int, int]]:
    """Build train/evaluation loaders over the same small training-only subset."""
    bundle = build_dataloaders(config, download=True)
    full_training_dataset = bundle.train_loader.dataset.dataset
    seed = int(config["seed"])
    selected_indices = stratified_subset_indices(
        targets=full_training_dataset.targets,
        candidate_indices=bundle.train_indices,
        sample_count=sample_count,
        seed=seed,
    )
    subset = Subset(full_training_dataset, selected_indices)
    selected_labels = full_training_dataset.targets[selected_indices].tolist()
    class_counts = dict(sorted(Counter(selected_labels).items()))

    data_config = config["data"]
    batch_size = min(int(data_config["batch_size"]), sample_count)
    num_workers = int(data_config["num_workers"])
    pin_memory = bool(data_config["pin_memory"] and torch.cuda.is_available())
    common_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": seed_worker,
        "persistent_workers": num_workers > 0,
    }

    train_loader = DataLoader(
        subset,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        **common_args,
    )
    evaluation_loader = DataLoader(subset, shuffle=False, **common_args)
    return train_loader, evaluation_loader, selected_indices, class_counts


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one optimization epoch and return mean loss and accuracy."""
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate loss, accuracy and Macro-F1 without updating parameters."""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        predictions = logits.argmax(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        all_labels.extend(labels.cpu().tolist())
        all_predictions.extend(predictions.cpu().tolist())

    accuracy = sum(
        prediction == label
        for prediction, label in zip(all_predictions, all_labels, strict=True)
    ) / total_samples
    macro_f1 = f1_score(all_labels, all_predictions, average="macro")
    return total_loss / total_samples, accuracy, float(macro_f1)


@torch.no_grad()
def evaluate_same_subset(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Measure memorization in eval mode on the same overfit subset."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return total_loss / total_samples, total_correct / total_samples


def save_overfit_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    accuracy: float,
    config: dict[str, Any],
    args: argparse.Namespace,
    selected_indices: list[int],
) -> None:
    """Save enough state to audit and reproduce the diagnostic run."""
    checkpoint = {
        "purpose": "small-data overfit diagnostic; not a formal experiment result",
        "epoch": epoch,
        "subset_accuracy": accuracy,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "project_config": config,
        "arguments": vars(args),
        "selected_train_indices": selected_indices,
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    torch.save(checkpoint, path)


def write_history(path: Path, history: list[dict[str, float | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["epoch", "optimization_loss", "eval_loss", "eval_accuracy"],
        )
        writer.writeheader()
        writer.writerows(history)


def run_check_only(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Validate one forward/loss pass without backward or optimizer updates."""
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(images)
        loss = criterion(logits, labels)
    print(f"check-only images: {tuple(images.shape)}")
    print(f"check-only labels: {tuple(labels.shape)}")
    print(f"check-only logits: {tuple(logits.shape)}")
    print(f"check-only loss: {loss.item():.4f}")
    print("check-only passed; no backward call and no optimizer step were executed")


def run_overfit(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    args.epochs = args.epochs or 100
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(config.get("device", "auto"))

    train_loader, evaluation_loader, selected_indices, class_counts = make_overfit_loaders(
        config, args.overfit_samples
    )
    model = build_tinyvit(config["model"], use_position_embedding=True).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"overfit samples: {len(selected_indices)}")
    print(f"class counts: {class_counts}")
    print(f"model parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")

    if args.check_only:
        run_check_only(model, evaluation_loader, criterion, device)
        return

    learning_rate = float(config["training"]["learning_rate"])
    weight_decay = float(config["training"]["weight_decay"])
    optimizer = AdamW(
        model.parameters(), learning_rate, weight_decay=weight_decay
    )

    run_name = f"overfit_tinyvit_seed{seed}"
    checkpoint_dir = resolve_project_path("checkpoints") / run_name
    output_dir = resolve_project_path("outputs") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    start_time = time.perf_counter()

    print(
        f"starting diagnostic: epochs={args.epochs}, lr={learning_rate}, "
        f"target_accuracy={args.target_accuracy:.1%}"
    )
    for epoch in range(1, args.epochs + 1):
        optimization_loss, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        eval_loss, eval_accuracy = evaluate_same_subset(
            model, evaluation_loader, criterion, device
        )
        record = {
            "epoch": epoch,
            "optimization_loss": optimization_loss,
            "eval_loss": eval_loss,
            "eval_accuracy": eval_accuracy,
        }
        history.append(record)
        write_history(output_dir / "history.csv", history)

        print(
            f"epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={optimization_loss:.4f} | "
            f"subset_eval_loss={eval_loss:.4f} | "
            f"subset_eval_accuracy={eval_accuracy:.2%}"
        )

        if eval_accuracy > best_accuracy:
            best_accuracy = eval_accuracy
            save_overfit_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                eval_accuracy,
                config,
                args,
                selected_indices,
            )

        if eval_accuracy >= args.target_accuracy:
            print(f"target reached at epoch {epoch}; stopping early")
            break

    elapsed_seconds = time.perf_counter() - start_time
    summary = {
        "purpose": "small-data overfit diagnostic; not a formal experiment result",
        "best_subset_accuracy": best_accuracy,
        "epochs_completed": len(history),
        "elapsed_seconds": elapsed_seconds,
        "target_accuracy": args.target_accuracy,
        "target_reached": best_accuracy >= args.target_accuracy,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if best_accuracy < args.target_accuracy:
        print(
            "diagnostic did not reach the target; inspect the curve before changing code "
            "or hyperparameters"
        )


def save_formal_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_accuracy: float,
    val_macro_f1: float,
    config: dict[str, Any],
    args: argparse.Namespace,
    run_name: str,
) -> None:
    """Save the validation-selected model and all information needed by evaluate.py."""
    checkpoint = {
        "purpose": "formal Fashion-MNIST classification experiment",
        "model_name": args.model,
        "run_name": run_name,
        "epoch": epoch,
        "selection_metric": "validation_accuracy_then_macro_f1",
        "validation_accuracy": val_accuracy,
        "validation_macro_f1": val_macro_f1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "model_config": asdict(model.config),
        "project_config": config,
        "arguments": vars(args),
        "class_names": list(FASHION_MNIST_CLASSES),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    torch.save(checkpoint, path)


def write_formal_history(
    path: Path, history: list[dict[str, float | int]]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "epoch_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(history)


def run_formal_check_only(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> None:
    """Check train/validation forward paths and prove parameters stay unchanged."""
    state_before = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.state_dict().items()
    }
    model.eval()
    with torch.no_grad():
        train_images, train_labels = next(iter(train_loader))
        val_images, val_labels = next(iter(val_loader))
        train_images = train_images.to(device, non_blocking=True)
        train_labels = train_labels.to(device, non_blocking=True)
        val_images = val_images.to(device, non_blocking=True)
        val_labels = val_labels.to(device, non_blocking=True)
        train_logits = model(train_images)
        val_logits = model(val_images)
        train_loss = criterion(train_logits, train_labels)
        val_loss = criterion(val_logits, val_labels)

    unchanged = all(
        torch.equal(state_before[name], tensor.detach().cpu())
        for name, tensor in model.state_dict().items()
    )
    print(f"formal check train batch: {tuple(train_images.shape)} -> {tuple(train_logits.shape)}")
    print(f"formal check val batch:   {tuple(val_images.shape)} -> {tuple(val_logits.shape)}")
    print(f"untrained train loss: {train_loss.item():.4f}")
    print(f"untrained val loss:   {val_loss.item():.4f}")
    print(f"parameters unchanged: {unchanged}")
    if not unchanged:
        raise RuntimeError("check-only unexpectedly changed model parameters")
    print("formal check-only passed; test_loader was not iterated")


def run_formal(args: argparse.Namespace) -> None:
    """Train TinyViT using train/validation only and save the validation best model."""
    config = load_config(args.config)
    args.epochs = args.epochs or int(config["training"]["epochs"])
    seed = int(config["seed"])
    set_seed(seed)
    device = select_device(config.get("device", "auto"))
    bundle = build_dataloaders(config, download=True)
    model = build_model(args.model, config["model"]).to(device)
    criterion = nn.CrossEntropyLoss()

    print(f"device: {device}")
    if device.type == "cuda":
        print(f"gpu: {torch.cuda.get_device_name(device)}")
    print(f"model: {args.model}")
    print(f"train/validation/test sizes: {len(bundle.train_loader.dataset)}/"
          f"{len(bundle.val_loader.dataset)}/{len(bundle.test_loader.dataset)}")
    print("selection rule: highest validation accuracy, then validation Macro-F1")
    print("test set policy: not iterated by train.py")

    if args.check_only:
        run_formal_check_only(
            model,
            bundle.train_loader,
            bundle.val_loader,
            criterion,
            device,
        )
        return

    optimizer = AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    run_name = f"{args.model}_seed{seed}"
    checkpoint_dir = resolve_project_path("checkpoints") / run_name
    output_dir = resolve_project_path("outputs") / run_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, float | int]] = []
    best_accuracy = -1.0
    best_macro_f1 = -1.0
    best_epoch = 0
    total_start = time.perf_counter()

    print(
        f"starting formal training: epochs={args.epochs}, "
        f"lr={config['training']['learning_rate']}, "
        f"weight_decay={config['training']['weight_decay']}"
    )
    for epoch in range(1, args.epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_start = time.perf_counter()

        train_loss, train_accuracy = train_one_epoch(
            model, bundle.train_loader, optimizer, criterion, device
        )
        val_loss, val_accuracy, val_macro_f1 = evaluate_validation(
            model, bundle.val_loader, criterion, device
        )

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_macro_f1": val_macro_f1,
                "epoch_seconds": epoch_seconds,
            }
        )
        write_formal_history(output_dir / "history.csv", history)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.2%} | "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.2%} "
            f"val_macro_f1={val_macro_f1:.4f} | {epoch_seconds:.2f}s"
        )

        is_better = val_accuracy > best_accuracy or (
            val_accuracy == best_accuracy and val_macro_f1 > best_macro_f1
        )
        if is_better:
            best_accuracy = val_accuracy
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            save_formal_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                epoch,
                val_accuracy,
                val_macro_f1,
                config,
                args,
                run_name,
            )
            print(f"  saved new validation best: {checkpoint_dir / 'best.pt'}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - total_start
    summary = {
        "purpose": "formal training/validation summary; test set not evaluated",
        "run_name": run_name,
        "best_epoch": best_epoch,
        "best_validation_accuracy": best_accuracy,
        "best_validation_macro_f1": best_macro_f1,
        "epochs_completed": args.epochs,
        "elapsed_seconds": elapsed_seconds,
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
        "test_set_evaluated": False,
    }
    with (output_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TinyViT overfit diagnostics or formal train/validation"
    )
    parser.add_argument("--mode", choices=("overfit", "formal"), default="overfit")
    parser.add_argument("--model", choices=MODEL_NAMES, default="tinyvit")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--overfit-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--target-accuracy", type=float, default=0.98)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run one forward/loss check without backward or parameter updates",
    )
    args = parser.parse_args()
    if args.overfit_samples <= 0:
        parser.error("--overfit-samples must be positive")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("--epochs must be positive")
    if not 0.0 < args.target_accuracy <= 1.0:
        parser.error("--target-accuracy must be in (0, 1]")
    if args.mode == "overfit" and args.model != "tinyvit":
        parser.error("overfit diagnostic is fixed to --model tinyvit")
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.mode == "formal":
        run_formal(parsed_args)
    else:
        run_overfit(parsed_args)
