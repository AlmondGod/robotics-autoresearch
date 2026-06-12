from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import pandas as pd
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="data/cosmos_robocasa_action")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--max-demos-per-task", type=int, default=0)
    parser.add_argument("--views", nargs="+", default=["robot0_agentview_left", "robot0_agentview_right"])
    parser.add_argument("--layout", choices=["copy_first", "side_by_side"], default="side_by_side")
    parser.add_argument("--resize", type=int, default=320)
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    out_dir = Path(args.out_dir)
    videos_dir = out_dir / "videos"
    ann_dir = out_dir / "annotations"
    meta_dir = out_dir / "metas"
    videos_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for task in manifest["tasks"]:
        dataset_root = Path(task["dataset_path"])
        episode_paths = sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet"))
        kept = 0
        for episode_path in episode_paths:
            episode_idx = int(episode_path.stem.split("_")[-1])
            frame = pd.read_parquet(episode_path)
            task_index = int(frame["task_index"].iloc[0])
            if args.robocasa_task_index and task_index not in set(args.robocasa_task_index):
                continue
            if args.max_demos_per_task and kept >= int(args.max_demos_per_task):
                break
            uid = f"{task['alias']}_taskidx{task_index}_ep{episode_idx:06d}"
            video_out = videos_dir / f"{uid}.mp4"
            ann_out = ann_dir / f"{uid}.json"
            meta_out = meta_dir / f"{uid}.txt"
            prompt = _prompt(task, task_index)
            _write_video(dataset_root, episode_idx, args.views, video_out, layout=str(args.layout), resize=int(args.resize), fps=int(args.fps))
            annotation = _annotation(frame, prompt=prompt, task=task, task_index=task_index, source_episode=episode_idx)
            ann_out.write_text(json.dumps(annotation, indent=2, sort_keys=True))
            meta_out.write_text(prompt + "\n")
            rows.append(
                {
                    "id": uid,
                    "task": task["alias"],
                    "task_index": task_index,
                    "episode_id": episode_idx,
                    "video": str(video_out.relative_to(out_dir)),
                    "annotation": str(ann_out.relative_to(out_dir)),
                    "meta": str(meta_out.relative_to(out_dir)),
                    "frames": len(frame),
                    "prompt": prompt,
                }
            )
            kept += 1
            print(json.dumps(rows[-1]), flush=True)
    summary = {
        "format": "cosmos_action_conditioned_bridge_like",
        "notes": [
            "videos/*.mp4 and annotations/*.json follow the Cosmos action-conditioned Bridge-style directory convention.",
            "RoboCasa actions are preserved at native 12D. Bridge uses 7D, so the Cosmos dataloader/config must set action_dim=12 or adapt this field.",
            "state_eef6 is observation.state[:6]; state_full is the native RoboCasa observation.state vector.",
        ],
        "views": args.views,
        "layout": args.layout,
        "resize": int(args.resize),
        "fps": int(args.fps),
        "clips": len(rows),
        "rows": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"out_dir": str(out_dir), "clips": len(rows)}, indent=2))


def _filtered_manifest(path: Path, aliases: list[str]) -> dict:
    manifest = json.loads(path.read_text())
    if aliases:
        keep = set(aliases)
        manifest["tasks"] = [task for task in manifest["tasks"] if task["alias"] in keep]
    if not manifest["tasks"]:
        raise ValueError("no tasks selected")
    return manifest


def _prompt(task: dict, task_index: int) -> str:
    description = str(task.get("description") or task.get("alias") or task.get("robocasa_task"))
    return f"Robot manipulation in a RoboCasa kitchen. Task: {description} RoboCasa task_index={task_index}."


def _annotation(frame: pd.DataFrame, *, prompt: str, task: dict, task_index: int, source_episode: int) -> dict:
    state_full = np.stack(frame["observation.state"].to_numpy()).astype(float)
    action = np.stack(frame["action"].to_numpy()).astype(float)
    gripper = _gripper_state(state_full, action)
    return {
        "prompt": prompt,
        "task": str(task["alias"]),
        "task_index": int(task_index),
        "source_episode": int(source_episode),
        "state": state_full[:, :6].tolist(),
        "state_full": state_full.tolist(),
        "continuous_gripper_state": gripper.tolist(),
        "action": action.tolist(),
        "reward": frame["next.reward"].astype(float).tolist() if "next.reward" in frame else [],
        "done": frame["next.done"].astype(bool).tolist() if "next.done" in frame else [],
        "timestamp": frame["timestamp"].astype(float).tolist() if "timestamp" in frame else list(range(len(frame))),
        "action_dim": int(action.shape[-1]),
        "state_dim": int(state_full.shape[-1]),
    }


def _gripper_state(state: np.ndarray, action: np.ndarray) -> np.ndarray:
    if state.shape[-1] >= 7:
        return state[:, 6].astype(float)
    if action.shape[-1] >= 1:
        return action[:, -1].astype(float)
    return np.zeros((state.shape[0],), dtype=float)


def _write_video(
    dataset_root: Path,
    episode_idx: int,
    views: list[str],
    out: Path,
    *,
    layout: str,
    resize: int,
    fps: int,
) -> None:
    if layout == "copy_first":
        src = dataset_root / "videos" / "chunk-000" / f"observation.images.{views[0]}" / f"episode_{episode_idx:06d}.mp4"
        shutil.copyfile(src, out)
        return
    streams = [iio.imiter(dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_idx:06d}.mp4") for view in views]
    frames = []
    for frame_tuple in zip(*streams, strict=False):
        ims = [_resize(np.asarray(frame)[..., :3], resize) for frame in frame_tuple]
        frames.append(np.concatenate(ims, axis=1))
    iio.imwrite(out, frames, fps=fps, codec="libx264")


def _resize(image: np.ndarray, size: int) -> np.ndarray:
    return np.asarray(Image.fromarray(image.astype(np.uint8)).resize((size, size), Image.Resampling.BILINEAR), dtype=np.uint8)


if __name__ == "__main__":
    main()
