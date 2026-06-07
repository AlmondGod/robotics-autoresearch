from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.libero_dataset import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-root", default="third_party/LIBERO")
    parser.add_argument("--out-dir", default="data/libero_object5")
    parser.add_argument("--task-count", type=int, default=5)
    parser.add_argument("--video-task-count", type=int, default=10)
    parser.add_argument("--paired-demos-per-task", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = build_manifest(
        libero_root=Path(args.libero_root),
        out_dir=Path(args.out_dir),
        task_count=args.task_count,
        video_task_count=args.video_task_count,
        paired_demos_per_task=args.paired_demos_per_task,
        seed=args.seed,
    )
    print(json.dumps({k: manifest[k] for k in ["suite", "task_count", "video_task_count", "paired_demos_per_task"]}, indent=2))
    print(Path(args.out_dir) / "manifest.json")


if __name__ == "__main__":
    main()
