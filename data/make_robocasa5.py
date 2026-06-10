from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_TASKS = [
    {
        "task_id": 0,
        "alias": "OpenDrawer",
        "robocasa_task": "OpenDrawer",
        "description": "Open a kitchen drawer.",
    },
    {
        "task_id": 1,
        "alias": "CloseDrawer",
        "robocasa_task": "CloseDrawer",
        "description": "Close a kitchen drawer.",
    },
    {
        "task_id": 2,
        "alias": "PickPlaceCounterToStove",
        "robocasa_task": "PickPlaceCounterToStove",
        "description": "Pick an object from the counter and place it on the stove.",
    },
    {
        "task_id": 3,
        "alias": "TurnOffStove",
        "robocasa_task": "TurnOffStove",
        "description": "Turn a stove knob off.",
    },
    {
        "task_id": 4,
        "alias": "PickPlaceObjectBetweenRegions",
        "robocasa_task": "PickPlaceObjectBetweenRegions",
        "description": "Move an object between marked regions or containers.",
    },
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/robocasa5/manifest.json")
    parser.add_argument(
        "--task-map",
        default="",
        help="Optional JSON file mapping aliases to local RoboCasa task names.",
    )
    parser.add_argument("--demos-per-task", type=int, default=50)
    parser.add_argument("--views", nargs="+", default=["robot0_agentview_left", "robot0_agentview_right"])
    args = parser.parse_args()

    tasks = [dict(task) for task in DEFAULT_TASKS]
    if args.task_map:
        task_map = json.loads(Path(args.task_map).read_text())
        for task in tasks:
            task["robocasa_task"] = task_map.get(task["alias"], task["robocasa_task"])

    manifest = {
        "benchmark": "robocasa5",
        "suite": "robocasa",
        "demos_per_task": args.demos_per_task,
        "views": args.views,
        "action_dim": 7,
        "tasks": tasks,
        "notes": [
            "This manifest is intentionally declarative.",
            "Use task-map if local RoboCasa task ids differ from the aliases.",
            "Real dataset paths are filled by the RoboCasa/LeRobot data adapter.",
        ],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(out)


if __name__ == "__main__":
    main()

