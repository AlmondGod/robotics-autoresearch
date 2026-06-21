from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) in sys.path:
    sys.path.remove(str(ROOT))
sys.path.insert(0, str(ROOT))

from tasks.robocasa_visual_world_model.inference import load_world_model
from tasks.robocasa_world_model.data import (
    DEFAULT_MANIFEST,
    DEFAULT_SPLIT,
    TransitionData,
    load_transition_data,
    load_video_frames,
    normalize_data,
    save_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate visually grounded RoboCasa world model.")
    parser.add_argument("--checkpoint", "--world-model", dest="checkpoint", required=True)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--split", default=str(DEFAULT_SPLIT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--train-episodes-per-task", type=int, default=20)
    parser.add_argument("--val-episodes-per-task", type=int, default=5)
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lpips-net", choices=["alex", "vgg", "squeeze"], default="alex")
    parser.add_argument("--lpips-size", type=int, default=64)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    start = time.monotonic()
    world = load_world_model(str(args.checkpoint), device=str(args.device))
    ckpt = world["checkpoint"]
    cfg = world["config"]
    _, val_raw, summary = load_transition_data(
        manifest_path=args.manifest,
        split_path=args.split,
        train_episodes_per_task=int(args.train_episodes_per_task),
        val_episodes_per_task=int(args.val_episodes_per_task),
        task_aliases=set(args.task_alias),
        frame_stride=int(args.frame_stride),
    )
    val = normalize_data(val_raw, ckpt["stats"])
    rgb, next_rgb = _precompute_rgb_targets(
        val,
        summary,
        view=str(cfg.get("view", "robot0_agentview_right")),
        image_size=int(cfg["image_size"]),
    )
    lpips_model = _load_lpips_model(str(args.lpips_net), world["device"])
    metrics = _visual_transition_eval(world, val, rgb, next_rgb, int(args.batch_size), lpips_model, int(args.lpips_size))
    benchmark = _benchmark_score(metrics)
    payload = {
        "task": "robocasa_visual_world_model",
        "checkpoint": str(args.checkpoint),
        "metric": "visual_world_model_score",
        **benchmark,
        "reproducibility_integrity": 1.0,
        "visual_transition_metrics": metrics,
        "summary": summary,
        "eval_seconds": time.monotonic() - start,
    }
    save_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


@torch.no_grad()
def _visual_transition_eval(
    world: dict,
    data: TransitionData,
    rgb: np.ndarray,
    next_rgb: np.ndarray,
    batch_size: int,
    lpips_model: torch.nn.Module,
    lpips_size: int,
) -> dict[str, float]:
    model = world["model"]
    device = world["device"]
    model.eval()
    sums = {
        "next_state_mse_norm": 0.0,
        "next_progress_mse": 0.0,
        "next_reward_mse": 0.0,
        "success_bce": 0.0,
        "next_rgb_mse": 0.0,
        "next_rgb_mae": 0.0,
        "next_rgb_lpips": 0.0,
        "next_rgb_delta_mse": 0.0,
    }
    count = 0
    for start in range(0, len(data), int(batch_size)):
        end = min(len(data), start + int(batch_size))
        batch = {
            "state": torch.as_tensor(data.state[start:end], dtype=torch.float32, device=device),
            "action": torch.as_tensor(data.action[start:end], dtype=torch.float32, device=device),
            "next_state": torch.as_tensor(data.next_state[start:end], dtype=torch.float32, device=device),
            "progress": torch.as_tensor(data.progress[start:end], dtype=torch.float32, device=device),
            "next_progress": torch.as_tensor(data.next_progress[start:end], dtype=torch.float32, device=device),
            "reward": torch.as_tensor(data.reward[start:end], dtype=torch.float32, device=device),
            "success": torch.as_tensor(data.success[start:end], dtype=torch.float32, device=device),
            "task_id": torch.as_tensor(data.task_id[start:end], dtype=torch.long, device=device),
            "rgb": torch.as_tensor(rgb[start:end], dtype=torch.float32, device=device),
            "next_rgb": torch.as_tensor(next_rgb[start:end], dtype=torch.float32, device=device),
        }
        out = model(batch["state"], batch["action"], batch["task_id"], batch["progress"])
        pred_delta = out["next_rgb"] - batch["rgb"]
        true_delta = batch["next_rgb"] - batch["rgb"]
        n = end - start
        sums["next_state_mse_norm"] += float((out["next_state"] - batch["next_state"]).square().mean(dim=-1).sum().detach().cpu())
        sums["next_progress_mse"] += float((out["next_progress"] - batch["next_progress"]).square().sum().detach().cpu())
        sums["next_reward_mse"] += float((out["reward"] - batch["reward"]).square().sum().detach().cpu())
        sums["success_bce"] += float(torch.nn.functional.binary_cross_entropy_with_logits(out["success_logit"], batch["success"], reduction="sum").detach().cpu())
        sums["next_rgb_mse"] += float((out["next_rgb"] - batch["next_rgb"]).square().mean(dim=(1, 2, 3)).sum().detach().cpu())
        sums["next_rgb_mae"] += float((out["next_rgb"] - batch["next_rgb"]).abs().mean(dim=(1, 2, 3)).sum().detach().cpu())
        sums["next_rgb_lpips"] += float(
            lpips_model(_lpips_input(out["next_rgb"], lpips_size), _lpips_input(batch["next_rgb"], lpips_size))
            .reshape(-1)
            .sum()
            .detach()
            .cpu()
        )
        sums["next_rgb_delta_mse"] += float((pred_delta - true_delta).square().mean(dim=(1, 2, 3)).sum().detach().cpu())
        count += n
    metrics = {key: value / max(1, count) for key, value in sums.items()}
    metrics["samples"] = int(count)
    metrics["next_rgb_psnr"] = float(-10.0 * math.log10(max(metrics["next_rgb_mse"], 1e-12)))
    return metrics


def _precompute_rgb_targets(
    data: TransitionData,
    summary: list[dict],
    *,
    view: str,
    image_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rgb = np.empty((len(data), 3, int(image_size), int(image_size)), dtype=np.float32)
    next_rgb = np.empty_like(rgb)
    dataset_by_task = {int(row["task_id"]): Path(row["dataset_path"]) for row in summary}
    groups: dict[tuple[int, int], list[int]] = {}
    for index, (task_id, episode_id) in enumerate(zip(data.task_id, data.episode_id)):
        groups.setdefault((int(task_id), int(episode_id)), []).append(index)
    for (task_id, episode_id), indices in sorted(groups.items()):
        root = dataset_by_task[int(task_id)]
        video = root / "videos" / "chunk-000" / f"observation.images.{view}" / f"episode_{int(episode_id):06d}.mp4"
        frames = load_video_frames(video)
        for index in indices:
            frame_idx = int(np.clip(data.frame_idx[index], 0, max(0, len(frames) - 1)))
            next_idx = min(frame_idx + 1, max(0, len(frames) - 1))
            rgb[index] = _preprocess_frame(frames[frame_idx], image_size)
            next_rgb[index] = _preprocess_frame(frames[next_idx], image_size)
    return rgb, next_rgb


def _preprocess_frame(frame: np.ndarray, image_size: int) -> np.ndarray:
    try:
        import cv2  # type: ignore

        resized = cv2.resize(frame, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    except ModuleNotFoundError:
        from PIL import Image

        resized = np.asarray(Image.fromarray(frame).resize((int(image_size), int(image_size))))
    return np.transpose(resized.astype(np.float32) / 255.0, (2, 0, 1))


def _load_lpips_model(net: str, device: torch.device) -> torch.nn.Module:
    try:
        import lpips  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "robocasa_visual_world_model eval requires lpips. Install with `pip install lpips` "
            "or install this repo with the robocasa extra after updating dependencies."
        ) from exc
    model = lpips.LPIPS(net=str(net), verbose=False).to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def _lpips_input(image: torch.Tensor, lpips_size: int) -> torch.Tensor:
    image = image.clamp(0.0, 1.0)
    if image.shape[-1] < int(lpips_size) or image.shape[-2] < int(lpips_size):
        image = F.interpolate(image, size=(int(lpips_size), int(lpips_size)), mode="bilinear", align_corners=False)
    return image * 2.0 - 1.0


def _benchmark_score(metrics: dict[str, float]) -> dict[str, float | dict[str, float]]:
    visual_perceptual = _mse_like_score(metrics.get("next_rgb_lpips"), scale=0.5)
    visual_reconstruction = _mse_like_score(metrics.get("next_rgb_mse"), scale=0.08)
    visual_delta = _mse_like_score(metrics.get("next_rgb_delta_mse"), scale=0.025)
    next_state = _mse_like_score(metrics.get("next_state_mse_norm"), scale=0.05)
    reward_progress = 0.5 * (
        _mse_like_score(metrics.get("next_reward_mse"), scale=0.25)
        + _mse_like_score(metrics.get("next_progress_mse"), scale=0.05)
    )
    success = _mse_like_score(metrics.get("success_bce"), scale=0.1)
    weights = {
        "visual_perceptual_score": 0.50,
        "visual_reconstruction_score": 0.10,
        "visual_delta_score": 0.15,
        "next_state_score": 0.15,
        "reward_progress_score": 0.05,
        "success_score": 0.05,
    }
    score = (
        weights["visual_perceptual_score"] * visual_perceptual
        + weights["visual_reconstruction_score"] * visual_reconstruction
        + weights["visual_delta_score"] * visual_delta
        + weights["next_state_score"] * next_state
        + weights["reward_progress_score"] * reward_progress
        + weights["success_score"] * success
    )
    return {
        "visual_world_model_score": float(max(0.0, min(1.0, score))),
        "visual_perceptual_score": float(visual_perceptual),
        "visual_reconstruction_score": float(visual_reconstruction),
        "visual_delta_score": float(visual_delta),
        "next_state_score": float(next_state),
        "reward_progress_score": float(reward_progress),
        "success_score": float(success),
        "benchmark_score_weights": weights,
    }


def _mse_like_score(value: float | None, *, scale: float) -> float:
    if value is None:
        return 0.0
    return float(max(0.0, min(1.0, 1.0 - float(value) / max(float(scale), 1e-12))))


if __name__ == "__main__":
    main()
