from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


PACKAGING_COMMANDS = {
    "bdist",
    "bdist_wheel",
    "build",
    "build_ext",
    "develop",
    "dist_info",
    "editable_wheel",
    "egg_info",
    "install",
    "sdist",
}

ROOT = Path(__file__).resolve().parent

CONFIGS = (
    "configs/autorobobench_v0.json",
    "configs/autorobobench_visual_world_model_v0.json",
    "configs/autorobobench_world_model_posttraining_v0.json",
)

JSON_FILES = (
    "data/robocasa5/manifest.json",
    "data/autorobobench/robocasa_bc5_splits.json",
    "data/autorobobench/robocasa_long_horizon_manifest.json",
    "data/autorobobench/robocasa_long_horizon_splits.json",
    "data/autorobobench/video_policy_transfer_splits.json",
    "data/autorobobench/video_policy_transfer_video_pool.json",
    "data/autorobobench/robocasa_world_model_policy_set.json",
    "data/autorobobench/robocasa_world_model_video_pool.json",
    "data/autorobobench/robocasa_faucet_peak_manifest.json",
    "data/autorobobench/robocasa_faucet_peak_splits.json",
    "data/autorobobench/robocasa_faucet_peak_video_pool.json",
    "data/autorobobench/robocasa_stand_mixer_peak_manifest.json",
    "data/autorobobench/robocasa_stand_mixer_peak_splits.json",
    "data/autorobobench/robocasa_stand_mixer_peak_video_pool.json",
    "data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json",
    "data/autorobobench/robocasa_choose_measuring_cup_language_splits.json",
)

METADATA_SETUP_COMMANDS = (
    ("robocasa_bc5", ("tasks/robocasa_bc5/setup.py",)),
    ("robocasa_long_horizon", ("tasks/robocasa_long_horizon/setup.py",)),
    ("video_policy_transfer", ("tasks/video_policy_transfer/setup.py",)),
    ("robocasa_world_model", ("tasks/robocasa_world_model/setup.py",)),
    ("robocasa_faucet_peak", ("tasks/robocasa_faucet_peak/setup.py",)),
    ("robocasa_stand_mixer_peak", ("tasks/robocasa_stand_mixer_peak/setup.py",)),
    ("robocasa_offlinerl_posttraining", ("tasks/robocasa_offlinerl_posttraining/setup.py",)),
    ("robocasa_world_model_posttraining", ("tasks/robocasa_world_model_posttraining/setup.py",)),
)

DATA_SETUP_COMMANDS = (
    ("robocasa_choose_measuring_cup_language", ("tasks/robocasa_choose_measuring_cup_language/setup.py",)),
    ("robocasa_visual_world_model", ("tasks/robocasa_visual_world_model/setup.py",)),
)

MANIFESTS_FOR_DOWNLOAD = (
    "data/robocasa5/manifest.json",
    "data/autorobobench/robocasa_long_horizon_manifest.json",
    "data/autorobobench/robocasa_faucet_peak_manifest.json",
    "data/autorobobench/robocasa_stand_mixer_peak_manifest.json",
    "data/autorobobench/robocasa_choose_measuring_cup_language_manifest.json",
)


def main() -> None:
    if _called_for_packaging():
        from setuptools import setup

        setup()
        return

    parser = argparse.ArgumentParser(
        description="Universal setup and verifier for the AutoRoboBench RoboCasa benchmark."
    )
    parser.add_argument("--verify", action="store_true", help="Verify local RoboCasa dataset files, not just metadata.")
    parser.add_argument("--download-robocasa", action="store_true", help="Download the RoboCasa tasks referenced by checked-in manifests.")
    parser.add_argument("--yes", action="store_true", help="Answer yes to RoboCasa downloader confirmation prompts.")
    parser.add_argument("--rebuild-bc5-manifest", action="store_true", help="Rebuild data/robocasa5/manifest.json from the local RoboCasa registry.")
    parser.add_argument("--skip-task-setup", action="store_true", help="Only check dependencies, JSON, and configs.")
    args = parser.parse_args()

    os.chdir(ROOT)
    _add_runtime_paths()
    print(json.dumps({"repo": str(ROOT), "python": sys.version.split()[0]}, sort_keys=True), flush=True)
    _check_python()
    _check_imports()

    if args.rebuild_bc5_manifest:
        cmd = [sys.executable, "data/make_robocasa5.py", "--out", "data/robocasa5/manifest.json"]
        if args.verify:
            cmd.append("--verify-exists")
        _run("rebuild_bc5_manifest", cmd)

    if args.download_robocasa:
        _download_robocasa(yes=bool(args.yes))

    _check_json_files()
    _describe_configs()

    if not args.skip_task_setup:
        _run_task_setups(verify=bool(args.verify))

    print(
        json.dumps(
            {
                "ok": True,
                "verify_data": bool(args.verify),
                "downloaded_robocasa": bool(args.download_robocasa),
            },
            indent=2,
            sort_keys=True,
        )
    )


def _called_for_packaging() -> bool:
    return len(sys.argv) > 1 and sys.argv[1] in PACKAGING_COMMANDS


def _check_python() -> None:
    if sys.version_info < (3, 10):
        raise SystemExit("Python >=3.10 is required.")


def _check_imports() -> None:
    required = ("numpy",)
    recommended = ("torch", "pandas", "pyarrow", "imageio", "lpips", "robocasa", "robosuite")
    payload = {
        "required_imports": {name: importlib.util.find_spec(name) is not None for name in required},
        "recommended_imports": {name: importlib.util.find_spec(name) is not None for name in recommended},
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    missing = [name for name, ok in payload["required_imports"].items() if not ok]
    if missing:
        raise ModuleNotFoundError(f"missing required imports: {', '.join(missing)}")


def _add_runtime_paths() -> None:
    for rel in ("third_party/robocasa", "third_party/robosuite"):
        path = str(ROOT / rel)
        if path not in sys.path:
            sys.path.insert(0, path)


def _check_json_files() -> None:
    checked = []
    for rel in JSON_FILES:
        path = ROOT / rel
        if not path.exists():
            raise FileNotFoundError(f"missing required JSON file: {rel}")
        json.loads(path.read_text())
        checked.append(rel)
    print(json.dumps({"json_files_checked": checked}, indent=2, sort_keys=True), flush=True)


def _describe_configs() -> None:
    for rel in CONFIGS:
        if not (ROOT / rel).exists():
            continue
        _run(f"describe:{rel}", [sys.executable, "-m", "autorobobench.cli", "describe", "--config", rel])


def _run_task_setups(*, verify: bool) -> None:
    commands = list(METADATA_SETUP_COMMANDS)
    if verify:
        commands.extend(DATA_SETUP_COMMANDS)
    for name, base_cmd in commands:
        cmd = [sys.executable, *base_cmd]
        if verify and _supports_verify(ROOT / base_cmd[0]):
            cmd.append("--verify")
        if name == "robocasa_visual_world_model" and not verify:
            cmd.append("--skip-lpips-check")
        _run(f"setup:{name}", cmd)


def _supports_verify(path: Path) -> bool:
    return "--verify" in path.read_text()


def _download_robocasa(*, yes: bool) -> None:
    tasks = sorted(_manifest_tasks())
    if not tasks:
        raise ValueError("no RoboCasa tasks found in checked-in manifests")
    cmd = [
        sys.executable,
        "data/download_robocasa.py",
        "--tasks",
        *tasks,
        "--split",
        "pretrain",
        "--source",
        "human",
    ]
    if yes:
        cmd.append("--yes")
    _run("download_robocasa", cmd)


def _manifest_tasks() -> set[str]:
    tasks: set[str] = set()
    for rel in MANIFESTS_FOR_DOWNLOAD:
        path = ROOT / rel
        if not path.exists():
            continue
        payload = json.loads(path.read_text())
        for task in payload.get("tasks", []):
            name = task.get("robocasa_task") or task.get("alias")
            if name:
                tasks.add(str(name))
    return tasks


def _run(label: str, cmd: list[str]) -> None:
    env = os.environ.copy()
    pythonpath = [str(ROOT / "third_party/robocasa"), str(ROOT / "third_party/robosuite")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    print(json.dumps({"run": label, "cmd": cmd}, sort_keys=True), flush=True)
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
