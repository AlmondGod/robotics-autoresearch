from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect and score a scaled RoboCasa-5 held-out world-evaluator archive.")
    parser.add_argument("--config", default="configs/world_model_evaluator_scale.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of candidates to process.")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    out_dir = Path(config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(config["candidates"])
    if int(args.limit) > 0:
        candidates = candidates[: int(args.limit)]

    archive_specs: list[str] = []
    for candidate in candidates:
        name = str(candidate["name"])
        split = str(candidate.get("split", "test"))
        eval_path = out_dir / name / "eval.json"
        trace_dir = out_dir / name / "traces"
        archive_specs.append(f"{split},{name},{eval_path}")
        if bool(args.skip_existing) and eval_path.exists() and _has_traces(eval_path):
            print(f"skip existing {name}: {eval_path}")
            continue
        cmd = [
            sys.executable,
            "tasks/robocasa_bc5/eval.py",
            "--checkpoint",
            str(candidate["checkpoint"]),
            "--manifest",
            str(config["manifest"]),
            "--split",
            str(config["split"]),
            "--out",
            str(eval_path),
            "--trace-dir",
            str(trace_dir),
            "--eval-episodes-per-task",
            str(int(config.get("eval_episodes_per_task", 10))),
            "--max-steps",
            str(int(config.get("max_steps", 260))),
            "--commit-steps",
            str(int(config.get("commit_steps", 16))),
            "--device",
            str(args.device),
        ]
        _run(cmd, dry_run=bool(args.dry_run))

    archive_path = out_dir / "archive.jsonl"
    build_cmd = [sys.executable, "tasks/world_model_evaluator/build_archive.py", "--out", str(archive_path)]
    for spec in archive_specs:
        build_cmd.extend(["--candidate", spec])
    _run(build_cmd, dry_run=bool(args.dry_run))

    if bool(args.score):
        for checkpoint in config.get("world_evaluators", []):
            stem = Path(checkpoint).parent.name
            score_cmd = [
                sys.executable,
                "tasks/world_model_evaluator/eval.py",
                "--checkpoint",
                str(checkpoint),
                "--archive",
                str(archive_path),
                "--out",
                str(out_dir / f"world_eval_{stem}.json"),
                "--scores-out",
                str(out_dir / f"world_scores_{stem}.jsonl"),
                "--plot",
                str(out_dir / f"world_corr_{stem}.svg"),
                "--imagined-rollouts",
                "1",
                "--imagined-steps",
                str(int(config.get("max_steps", 260))),
                "--action-noise",
                "0.0",
                "--invert-learned-score",
                "--top-k",
                "5",
                "--device",
                str(args.device),
            ]
            _run(score_cmd, dry_run=bool(args.dry_run))

    print(json.dumps({"out_dir": str(out_dir), "archive": str(archive_path), "candidates": len(candidates)}, indent=2))


def _has_traces(eval_path: Path) -> bool:
    try:
        payload = json.loads(eval_path.read_text())
    except Exception:
        return False
    return any(detail.get("trace_path") and Path(detail["trace_path"]).exists() for detail in payload.get("details", []))


def _run(cmd: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
