from __future__ import annotations

import argparse
import json
from pathlib import Path

from robotbench.config import load_task
from robotbench.demos import record_arx_l5_demos
from robotbench.logging import RUNS_DIR, utc_now_id, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="arx_l5", choices=["arx_l5"])
    parser.add_argument("--task", default="reach", choices=["reach"])
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="")
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    task = load_task(args.task)
    out_path = Path(args.out) if args.out else RUNS_DIR / f"{utc_now_id()}-{args.backend}-{args.task}-demos.npz"
    summary = record_arx_l5_demos(
        task=task,
        episodes=args.episodes,
        seed=args.seed,
        out_path=out_path,
        include_video=not args.no_video,
    )
    write_json(out_path.with_suffix(".json"), summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
