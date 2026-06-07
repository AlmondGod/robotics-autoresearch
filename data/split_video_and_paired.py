from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.libero_dataset import materialize_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/libero_object5/manifest.json")
    parser.add_argument("--out-dir", default="data/libero_object5")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--max-transitions-per-demo", type=int, default=120)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--action-horizon", type=int, default=4)
    parser.add_argument("--video-repeat-factor", type=int, default=0)
    args = parser.parse_args()

    summary = materialize_shards(
        manifest_path=Path(args.manifest),
        out_dir=Path(args.out_dir),
        image_size=args.image_size,
        max_transitions_per_demo=args.max_transitions_per_demo,
        history=args.history,
        action_horizon=args.action_horizon,
        video_repeat_factor=args.video_repeat_factor or None,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
