from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Load a YAML config, resolving a relative path from the project root."""
    path = Path(config_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {path}")
    return config


def resolve_project_path(path: str | Path) -> Path:
    """Resolve project-relative paths consistently, independent of the shell cwd."""
    result = Path(path)
    return result if result.is_absolute() else PROJECT_ROOT / result


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for repeatable experiments."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # These settings improve repeatability without forcing every PyTorch op
    # into strict deterministic mode, which can reject otherwise valid ops.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic NumPy/Python seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def select_device(requested: str = "auto") -> torch.device:
    """Select CUDA when available, unless a specific device was requested."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)

