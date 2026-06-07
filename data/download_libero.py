from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default="third_party/LIBERO")
    parser.add_argument("--dataset", default="libero_object")
    parser.add_argument("--use-huggingface", action="store_true")
    args = parser.parse_args()

    dest = Path(args.dest)
    if not dest.exists():
        subprocess.run(["git", "clone", REPO_URL, str(dest)], check=True)
    config_path = Path(".libero_config").resolve()
    os.environ["LIBERO_CONFIG_PATH"] = str(config_path)
    cmd = [
        "python",
        "benchmark_scripts/download_libero_datasets.py",
        "--datasets",
        args.dataset,
    ]
    if args.use_huggingface:
        cmd.append("--use-huggingface")
    subprocess.run(cmd, cwd=dest, check=True)


if __name__ == "__main__":
    main()
