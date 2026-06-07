from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


TRAIN_FILE = Path("train.py")


def main() -> None:
    judge = _load_judge()
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--baseline", default="")
    parser.add_argument("--runs-root", default="runs/libero/autoresearch")
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--keep-rejected", action="store_true")
    args = parser.parse_args()

    baseline = Path(args.baseline) if args.baseline else None
    runs_root = Path(args.runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    for idx in range(args.iterations):
        backup = TRAIN_FILE.read_text()
        run_dir = runs_root / (time.strftime("%Y%m%dT%H%M%SZ") + f"-iter{idx:02d}")
        _run(
            [
                sys.executable,
                "train.py",
                "--out-dir",
                str(run_dir),
                "--max-train-seconds",
                str(args.max_train_seconds),
            ]
        )
        if baseline is None:
            baseline = run_dir
            _write_decision(run_dir, {"accepted": True, "reason": "initialized_baseline"})
            print(f"initialized baseline: {baseline}")
            continue

        decision = judge(baseline, run_dir)
        _write_decision(run_dir, decision)
        print(json.dumps(decision, indent=2, sort_keys=True))
        if decision["accepted"]:
            baseline = run_dir
        else:
            TRAIN_FILE.write_text(backup)
            if not args.keep_rejected:
                shutil.rmtree(run_dir, ignore_errors=False)


def _write_decision(run_dir: Path, decision: dict) -> None:
    (run_dir / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _load_judge():
    spec = importlib.util.spec_from_file_location("libero_judge", Path("research/judge.py"))
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load research/judge.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.judge


if __name__ == "__main__":
    main()
