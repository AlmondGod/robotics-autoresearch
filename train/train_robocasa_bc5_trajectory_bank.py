from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from autorobobench.robocasa_runtime import ensure_robocasa_runtime


ensure_robocasa_runtime()

import robocasa.utils.lerobot_utils as LU  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a RoboCasa BC-5 nearest-trajectory-bank baseline.")
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--split", default="data/autorobobench/robocasa_bc5_splits.json")
    parser.add_argument("--out-dir", default="runs/autorobobench/robocasa_bc5/trajectory_bank")
    parser.add_argument("--train-episodes-per-task", type=int, default=80)
    parser.add_argument("--include-val", action="store_true")
    parser.add_argument("--include-eval", action="store_true", help="Dev oracle only: include frozen eval episodes in the bank.")
    parser.add_argument("--include-extra-non-eval", action="store_true")
    parser.add_argument("--horizon", type=int, default=320)
    parser.add_argument("--embedding", choices=["rgb8", "rgb16"], default="rgb16")
    parser.add_argument("--eval-chunk", type=int, default=16)
    parser.add_argument("--select-by-episode-id", action="store_true", help="Dev oracle only: select bank row by eval episode id when available.")
    args = parser.parse_args()

    start = time.monotonic()
    manifest = json.loads(Path(args.manifest).read_text())
    split = json.loads(Path(args.split).read_text())
    manifest_tasks = {task["alias"]: task for task in manifest["tasks"]}
    embeddings: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    task_ids: list[int] = []
    episode_ids: list[int] = []
    aliases: list[str] = []

    for split_task in split["tasks"]:
        alias = split_task["alias"]
        dataset_root = Path(manifest_tasks[alias]["dataset_path"])
        episode_ids_for_task = list(split_task["train_episode_ids"])
        if args.train_episodes_per_task > 0:
            episode_ids_for_task = episode_ids_for_task[: int(args.train_episodes_per_task)]
        if args.include_val:
            episode_ids_for_task.extend(split_task["val_episode_ids"])
        if args.include_eval:
            episode_ids_for_task.extend(split_task["eval_episode_ids"])
        if args.include_extra_non_eval:
            dataset_root = Path(manifest_tasks[alias]["dataset_path"])
            blocked = set() if args.include_eval else set(int(ep) for ep in split_task["eval_episode_ids"])
            known = set(int(ep) for ep in episode_ids_for_task)
            for episode_path in sorted((dataset_root / "data" / "chunk-000").glob("episode_*.parquet")):
                episode_id = int(episode_path.stem.split("_")[-1])
                if episode_id not in blocked and episode_id not in known:
                    episode_ids_for_task.append(episode_id)
        for episode_id in episode_ids_for_task:
            embeddings.append(_episode_embedding(dataset_root, int(episode_id), str(args.embedding)))
            actions.append(_episode_actions(dataset_root, int(episode_id), int(args.horizon)))
            task_ids.append(int(split_task["task_id"]))
            episode_ids.append(int(episode_id))
            aliases.append(alias)

    checkpoint = {
        "policy_type": "robocasa_bc5_trajectory_bank",
        "task_count": len(split["tasks"]),
        "action_dim": int(actions[0].shape[-1]),
        "horizon": int(args.horizon),
        "eval_chunk": int(args.eval_chunk),
        "embedding": str(args.embedding),
        "embeddings": np.stack(embeddings).astype(np.float32),
        "actions": np.stack(actions).astype(np.float32),
        "task_ids": np.asarray(task_ids, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "aliases": aliases,
        "select_by_episode_id": bool(args.select_by_episode_id),
        "manifest": str(Path(args.manifest)),
        "split": str(Path(args.split)),
        "train_seconds": float(time.monotonic() - start),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, out_dir / "policy.pt")
    metrics = {
        "checkpoint": str(out_dir / "policy.pt"),
        "train_seconds": checkpoint["train_seconds"],
        "train_trajectories": len(actions),
        "train_episodes_per_task": int(args.train_episodes_per_task),
        "include_val": bool(args.include_val),
        "include_eval": bool(args.include_eval),
        "include_extra_non_eval": bool(args.include_extra_non_eval),
        "horizon": int(args.horizon),
        "embedding": str(args.embedding),
        "eval_chunk": int(args.eval_chunk),
        "select_by_episode_id": bool(args.select_by_episode_id),
        "tasks": sorted(set(aliases)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _episode_embedding(dataset_root: Path, episode_id: int, embedding: str) -> np.ndarray:
    size = 8 if embedding == "rgb8" else 16
    parts = []
    for view in ("robot0_agentview_left", "robot0_agentview_right"):
        path = dataset_root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{episode_id:06d}.mp4"
        frame = iio.imread(path, index=0)[..., :3]
        small = Image.fromarray(np.asarray(frame, dtype=np.uint8)).resize((size, size), Image.Resampling.BILINEAR)
        parts.append(np.asarray(small, dtype=np.float32).reshape(-1) / 255.0)
    return np.concatenate(parts).astype(np.float32)


def _episode_actions(dataset_root: Path, episode_id: int, horizon: int) -> np.ndarray:
    actions = LU.get_episode_actions(dataset_root, episode_id).astype(np.float32)
    padded = np.zeros((horizon, actions.shape[-1]), dtype=np.float32)
    n = min(horizon, len(actions))
    padded[:n] = actions[:n]
    if n < horizon and n > 0:
        padded[n:] = actions[n - 1]
    return padded


if __name__ == "__main__":
    main()
