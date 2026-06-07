from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="configs/libero_v0_bc.json")
    parser.add_argument("--out", default="configs/proposed.json")
    parser.add_argument("--change", default="increase policy training steps")
    args = parser.parse_args()

    config = json.loads(Path(args.base).read_text())
    config["change"] = args.change
    config["train_steps"] = int(config.get("train_steps", 2000) * 1.5)
    Path(args.out).write_text(json.dumps(config, indent=2, sort_keys=True))
    print(args.out)


if __name__ == "__main__":
    main()
