from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ARCHIVE = Path("research/archive.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--skip-success-eval", action="store_true")
    parser.add_argument("--max-train-seconds", type=float, default=300.0)
    parser.add_argument("--eval-episodes-per-task", type=int, default=1)
    parser.add_argument("--eval-max-steps", type=int, default=150)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    run_id = time.strftime("%Y%m%dT%H%M%SZ") + "-" + config["method"]
    out_dir = Path(args.out_dir or f"runs/libero/{run_id}")
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = run_config(
        config=config,
        out_dir=out_dir,
        skip_success_eval=args.skip_success_eval,
        max_train_seconds=args.max_train_seconds,
        eval_episodes_per_task=args.eval_episodes_per_task,
        eval_max_steps=args.eval_max_steps,
    )
    archive = {"run_id": run_id, "config": args.config, "out_dir": str(out_dir), "metrics": metrics}
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    with ARCHIVE.open("a") as handle:
        handle.write(json.dumps(archive, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def run_config(
    config: dict,
    out_dir: Path,
    skip_success_eval: bool,
    max_train_seconds: float = 300.0,
    eval_episodes_per_task: int = 1,
    eval_max_steps: int = 150,
) -> dict:
    steps = int(config.get("train_steps", 2000))
    data = config["data"]
    models = config.get("models", {})
    metrics: dict[str, float | None | str] = {
        "change": config.get("change", ""),
        "commit": _git_commit(),
        "method": config["method"],
    }

    tokenizer_path = None
    if models.get("tokenizer") is not None or config["method"] in {"tokenizer", "world", "world_inverse"}:
        tok_dir = out_dir / "tokenizer"
        tok = models.get("tokenizer") or {}
        _run(
            [
                "python",
                "train/train_tokenizer.py",
                "--data",
                data["video"],
                "--out-dir",
                str(tok_dir),
                "--steps",
                str(steps),
                "--batch-size",
                str(tok.get("batch_size", 64)),
                "--codebook-size",
                str(tok.get("codebook_size", 128)),
                "--embed-dim",
                str(tok.get("embed_dim", 64)),
                "--lr",
                str(tok.get("lr", 3e-4)),
                "--max-train-seconds",
                str(max_train_seconds),
            ]
        )
        metrics.update(_read_metrics(tok_dir))
        tokenizer_path = tok_dir / "tokenizer.pt"

    if models.get("world_model") is not None:
        wm_dir = out_dir / "world_model"
        wm = models["world_model"]
        _run(
            [
                "python",
                "train/train_world_model.py",
                "--data",
                data["video"],
                "--tokenizer",
                str(tokenizer_path),
                "--out-dir",
                str(wm_dir),
                "--steps",
                str(steps),
                "--batch-size",
                str(wm.get("batch_size", 64)),
                "--layers",
                str(wm.get("layers", 4)),
                "--lr",
                str(wm.get("lr", 3e-4)),
                "--max-train-seconds",
                str(max_train_seconds),
            ]
        )
        metrics.update(_read_metrics(wm_dir))

    if models.get("inverse_dynamics") is not None:
        inv_dir = out_dir / "inverse"
        inv = models["inverse_dynamics"]
        _run(
            [
                "python",
                "train/train_inverse.py",
                "--data",
                data["paired"],
                "--tokenizer",
                str(tokenizer_path),
                "--out-dir",
                str(inv_dir),
                "--steps",
                str(steps),
                "--batch-size",
                str(inv.get("batch_size", 64)),
                "--lr",
                str(inv.get("lr", 3e-4)),
            ]
        )
        metrics.update(_read_metrics(inv_dir))

    pol_dir = out_dir / "policy"
    pol = models.get("policy") or {}
    _run(
        [
            "python",
            "train/train_policy.py",
            "--data",
            data["paired"],
            "--out-dir",
            str(pol_dir),
            "--steps",
            str(steps),
            "--batch-size",
            str(pol.get("batch_size", 64)),
            "--lr",
            str(pol.get("lr", 3e-4)),
            "--method",
            config["method"],
            "--max-train-seconds",
            str(max_train_seconds),
        ]
    )
    metrics.update(_read_metrics(pol_dir))

    if skip_success_eval:
        metrics["success_rate"] = None
    else:
        success_path = pol_dir / "libero_success.json"
        _run(
            [
                "python",
                "eval/eval_libero_success.py",
                "--policy",
                str(pol_dir / "policy.pt"),
                "--episodes-per-task",
                str(eval_episodes_per_task),
                "--max-steps",
                str(eval_max_steps),
                "--device",
                "cpu",
                "--out",
                str(success_path),
            ]
        )
        success = json.loads(success_path.read_text())
        metrics["success_rate"] = success["success_rate"]
        metrics["success_episodes"] = success["episodes"]

    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return metrics


def _run(cmd: list[str]) -> None:
    if cmd and cmd[0] == "python":
        cmd = [sys.executable, *cmd[1:]]
    subprocess.run(cmd, check=True)


def _read_metrics(run_dir: Path) -> dict:
    path = run_dir / "metrics.json"
    return json.loads(path.read_text())


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except subprocess.SubprocessError:
        return "unknown"


if __name__ == "__main__":
    main()
