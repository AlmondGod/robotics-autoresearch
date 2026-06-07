from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import prepare


# Autoresearch agents should edit this file, not prepare.py or eval/.
METHOD = "bc"
CHANGE = "BC baseline"
TRAIN_STEPS = 1_000_000
BATCH_SIZE = 64
LR = 3e-4
N_EMBD = 256
LOG_EVERY = 250
DEVICE = "auto"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="runs/libero/train_py")
    parser.add_argument("--max-train-seconds", type=float, default=prepare.TRAIN_TIME_SECONDS)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            "python",
            "train/train_policy.py",
            "--data",
            str(prepare.PAIRED_SHARD),
            "--out-dir",
            str(out_dir),
            "--method",
            METHOD,
            "--steps",
            str(TRAIN_STEPS),
            "--batch-size",
            str(BATCH_SIZE),
            "--lr",
            str(LR),
            "--n-embd",
            str(N_EMBD),
            "--log-every",
            str(LOG_EVERY),
            "--device",
            DEVICE,
            "--max-train-seconds",
            str(args.max_train_seconds),
        ]
    )

    metrics = json.loads((out_dir / "metrics.json").read_text())
    metrics["change"] = CHANGE
    metrics["commit"] = _git_commit()
    metrics["max_train_seconds"] = args.max_train_seconds
    if not args.skip_eval:
        success = prepare.eval_policy(out_dir / "policy.pt", out_dir / "libero_success.json")
        metrics["success_rate"] = success["success_rate"]
        metrics["success_episodes"] = success["episodes"]
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    subprocess.run(cmd, check=True)


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        return "unknown"


if __name__ == "__main__":
    main()
