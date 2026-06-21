from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()


FROZEN_MANIFEST = "data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json"
FROZEN_SPLIT = "data/autorobobench/robocasa_choose_measuring_cup_language_splits.json"


def main() -> None:
    _default("--manifest", FROZEN_MANIFEST)
    _default("--split", FROZEN_SPLIT)
    _default("--out-dir", "runs/autorobobench/robocasa_choose_measuring_cup_language/smolvlm_flow")
    _default("--train-episodes-per-task", "16")
    _default("--val-episodes-per-task", "2")
    _default("--policy-kind", "frozen_smolvlm_flow")
    _default("--vlm-encoder-name", "HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    _default("--chunk-horizon", "16")
    _default("--frame-stride", "2")
    _default("--steps", "800")
    _default("--batch-size", "64")
    _default("--width", "256")
    _default("--dropout", "0.03")
    _default("--lr", "2e-4")
    _default("--weight-decay", "1e-4")
    _default("--image-noise", "0.004")
    _default("--proprio-noise", "0.004")
    _default("--chunk-decay", "0.85")
    _default("--bc-aux-weight", "0.1")
    _default("--flow-steps", "8")
    _default("--flow-source", "noise")
    _default("--flow-eval-start", "bc")
    _default("--action-depth", "2")
    _default("--heads", "4")
    _default("--history-stride", "16")
    _default("--eval-commit-steps", "8")
    _default("--balanced-sampling", None)

    from train.train_autorobobench_robocasa_bc5 import main as train_main

    train_main()


def _default(flag: str, value: str | None) -> None:
    if flag in sys.argv:
        return
    sys.argv.append(flag)
    if value is not None:
        sys.argv.append(value)


if __name__ == "__main__":
    main()
