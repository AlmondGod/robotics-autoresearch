from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path


HF_DROID_100 = "lerobot/droid_100"
GCS_RLDS_FULL = "gs://gresearch/robotics/droid"
GCS_RLDS_100 = "gs://gresearch/robotics/droid_100"
GCS_RAW = "gs://gresearch/robotics/droid_raw"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DROID as an external robot-video corpus.")
    parser.add_argument("--root", default="data/external/droid")
    parser.add_argument(
        "--mode",
        choices=["manifest-only", "hf-debug", "gcs-debug", "gcs-full-rlds", "gcs-full-raw-no-stereo"],
        default="manifest-only",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    download_path: Path | None = None
    commands: list[list[str]] = []

    if args.mode == "hf-debug":
        download_path = root / "hf_droid_100"
        commands.append(_hf_download_command(HF_DROID_100, download_path))
    elif args.mode == "gcs-debug":
        download_path = root / "gcs_droid_100"
        commands.append(["gsutil", "-m", "cp", "-r", GCS_RLDS_100, str(download_path)])
    elif args.mode == "gcs-full-rlds":
        download_path = root / "gcs_droid_full_rlds"
        commands.append(["gsutil", "-m", "cp", "-r", GCS_RLDS_FULL, str(download_path)])
    elif args.mode == "gcs-full-raw-no-stereo":
        download_path = root / "gcs_droid_raw_no_stereo"
        commands.append(
            [
                "gsutil",
                "-m",
                "rsync",
                "-r",
                "-x",
                ".*SVO.*|.*stereo.*\\.mp4$",
                GCS_RAW,
                str(download_path),
            ]
        )

    for command in commands:
        _run(command, dry_run=bool(args.dry_run))

    manifest = _manifest(root=root, mode=str(args.mode), download_path=download_path, commands=commands)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"manifest": str(manifest_path), **manifest}, indent=2, sort_keys=True))


def _hf_download_command(repo_id: str, out_dir: Path) -> list[str]:
    return [
        "huggingface-cli",
        "download",
        repo_id,
        "--repo-type",
        "dataset",
        "--local-dir",
        str(out_dir),
    ]


def _run(command: list[str], *, dry_run: bool) -> None:
    print("+ " + " ".join(command), flush=True)
    if dry_run:
        return
    executable = shutil.which(command[0])
    if executable is None:
        raise FileNotFoundError(
            f"{command[0]!r} is not installed. Re-run with --dry-run to print the command, "
            "or install the required CLI on the target machine."
        )
    subprocess.run(command, check=True)


def _manifest(*, root: Path, mode: str, download_path: Path | None, commands: list[list[str]]) -> dict:
    video_files = []
    parquet_files = []
    if download_path is not None and download_path.exists():
        video_files = [str(path.relative_to(root)) for path in sorted(download_path.rglob("*.mp4"))[:1000]]
        parquet_files = [str(path.relative_to(root)) for path in sorted(download_path.rglob("*.parquet"))[:1000]]
    return {
        "id": "droid_external_v0",
        "mode": mode,
        "root": str(root),
        "download_path": str(download_path) if download_path is not None else None,
        "commands": [" ".join(command) for command in commands],
        "source": {
            "hf_debug": HF_DROID_100,
            "gcs_rlds_full": GCS_RLDS_FULL,
            "gcs_rlds_100": GCS_RLDS_100,
            "gcs_raw": GCS_RAW,
        },
        "license_note": "DROID is released for research use; check the upstream dataset page before redistribution.",
        "intended_use": {
            "video_only_pretraining": True,
            "paired_action_pretraining": True,
            "final_eval": False,
        },
        "indexed_files": {
            "video_count_capped": len(video_files),
            "parquet_count_capped": len(parquet_files),
            "videos": video_files,
            "parquets": parquet_files,
        },
    }


if __name__ == "__main__":
    main()
