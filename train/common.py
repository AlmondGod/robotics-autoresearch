from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import torch


def device_from_arg(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def batches(count: int, batch_size: int, steps: int, seed: int = 0) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        yield rng.integers(0, count, size=batch_size)


def write_metrics(out_dir: Path, metrics: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"created_unix": time.time(), **metrics}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))


def save_checkpoint(out_dir: Path, name: str, model: torch.nn.Module, extra: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save({"state_dict": model.state_dict(), **(extra or {})}, path)
    return path
