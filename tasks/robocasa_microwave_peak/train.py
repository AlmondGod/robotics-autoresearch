from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

from train.train_autorobobench_robocasa_bc5 import main  # noqa: E402


def _default(flag: str, value: str) -> None:
    if flag not in sys.argv:
        sys.argv.extend([flag, value])


if __name__ == "__main__":
    _default("--manifest", "data/autorobobench/robocasa_microwave_peak_manifest.json")
    _default("--split", "data/autorobobench/robocasa_microwave_peak_splits.json")
    _default("--video-pool", "data/autorobobench/robocasa_microwave_peak_video_pool.json")
    _default("--out-dir", "runs/autorobobench/robocasa_microwave_peak/bc5_base")
    _default("--train-episodes-per-task", "80")
    _default("--val-episodes-per-task", "10")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "1")
    _default("--steps", "10000")
    _default("--batch-size", "128")
    _default("--width", "256")
    _default("--dropout", "0.03")
    _default("--lr", "2e-4")
    _default("--image-noise", "0.005")
    _default("--proprio-noise", "0.005")
    _default("--action-smooth", "0.0005")
    _default("--chunk-decay", "0.85")
    _default("--video-pretrain-steps", "100")
    _default("--video-pretrain-episodes-per-task", "12")
    _default("--video-pretrain-batch-size", "128")
    _default("--video-pretrain-gap", "8")
    main()
