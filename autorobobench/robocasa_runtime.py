from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def ensure_robocasa_runtime() -> None:
    """Make the local RoboCasa checkout importable in non-editable environments."""
    root = repo_root()
    for rel in ("third_party/robocasa", "third_party/robosuite", "."):
        path = str((root / rel).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)
    os.environ.setdefault("PYTHONPATH", os.pathsep.join(sys.path))
    _patch_lerobot_write_info()


def _patch_lerobot_write_info() -> None:
    try:
        import lerobot.datasets.utils as utils
    except ModuleNotFoundError:
        return
    if hasattr(utils, "write_info"):
        return

    def write_info(info: dict, root: str | Path) -> None:
        root_path = Path(root)
        path = root_path if root_path.name == "info.json" else root_path / "info.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2, sort_keys=True) + "\n")

    utils.write_info = write_info
