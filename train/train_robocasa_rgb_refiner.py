from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from models.robocasa_rgb_refiner import RoboCasaRGBRefiner
from models.robocasa_tiny_evaluator import RoboCasaVAEWorldModel
from train.common import device_from_arg
from train.train_robocasa_flow_next_rgb import _apply_checkpoint_norm
from train.train_robocasa_tiny_evaluator import _batch, _filtered_manifest, _load_data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-checkpoint", required=True)
    parser.add_argument("--manifest", default="data/robocasa5/manifest.json")
    parser.add_argument("--out-dir", default="runs/robocasa/world_evaluator/rgb_refiner")
    parser.add_argument("--task-alias", action="append", default=[])
    parser.add_argument("--robocasa-task-index", action="append", type=int, default=[])
    parser.add_argument("--condition-on-robocasa-task-index", action="store_true")
    parser.add_argument("--train-demos-per-task", type=int, default=80)
    parser.add_argument("--val-episode-id", action="append", type=int, default=[])
    parser.add_argument("--frame-stride", type=int, default=8)
    parser.add_argument("--success-window", type=float, default=0.9)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--cond-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    vae_ckpt = torch.load(args.vae_checkpoint, map_location=device, weights_only=False)
    if vae_ckpt.get("model_type") != "robocasa_vae_world_model":
        raise ValueError("expected robocasa_vae_world_model checkpoint")
    vae = RoboCasaVAEWorldModel(
        proprio_dim=int(vae_ckpt["proprio_dim"]),
        action_dim=int(vae_ckpt["action_dim"]),
        task_count=int(vae_ckpt["task_count"]),
        latent_dim=int(vae_ckpt["latent_dim"]),
        width=int(vae_ckpt.get("width", 512)),
        dropout=float(vae_ckpt.get("dropout", 0.0)),
    ).to(device)
    vae.load_state_dict(vae_ckpt["state_dict"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad_(False)

    manifest = _filtered_manifest(Path(args.manifest), args.task_alias)
    train, val = _load_data(
        manifest,
        train_demos_per_task=int(args.train_demos_per_task),
        val_episode_ids=set(args.val_episode_id),
        robocasa_task_indices=set(args.robocasa_task_index),
        condition_on_robocasa_task_index=bool(args.condition_on_robocasa_task_index),
        frame_stride=int(args.frame_stride),
        success_window=float(args.success_window),
    )
    if len(train) == 0 or len(val) == 0:
        raise ValueError("need non-empty train and val transition data")
    _apply_checkpoint_norm(train, vae_ckpt)
    _apply_checkpoint_norm(val, vae_ckpt)

    model = RoboCasaRGBRefiner(
        latent_dim=int(vae_ckpt["latent_dim"]),
        action_dim=int(vae_ckpt["action_dim"]),
        task_count=int(vae_ckpt["task_count"]),
        base_channels=int(args.base_channels),
        cond_dim=int(args.cond_dim),
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    rng = np.random.default_rng(int(args.seed))
    history: list[dict] = []
    best_psnr = -math.inf
    best_state = None
    best_step = 0
    started = time.time()

    for step in range(1, int(args.steps) + 1):
        idx = rng.integers(0, len(train), size=int(args.batch_size))
        batch = _batch(train, idx, device)
        loss, parts = _loss(vae, model, batch)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        record = {"step": step, **parts}
        history.append(record)
        if step == 1 or step % int(args.log_interval) == 0 or step == int(args.steps):
            val_metrics = _eval(vae, model, val, device, int(args.batch_size))
            record.update({f"val_{key}": value for key, value in val_metrics.items()})
            if val_metrics["psnr"] > best_psnr:
                best_psnr = float(val_metrics["psnr"])
                best_step = step
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            print(
                f"step={step} loss={parts['loss']:.6f} val_psnr={val_metrics['psnr']:.2f} "
                f"val_prior_psnr={val_metrics['prior_psnr']:.2f}",
                flush=True,
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "state_dict": model.state_dict(),
        "model_type": "robocasa_rgb_refiner",
        "vae_checkpoint": str(Path(args.vae_checkpoint)),
        "latent_dim": int(vae_ckpt["latent_dim"]),
        "action_dim": int(vae_ckpt["action_dim"]),
        "task_count": int(vae_ckpt["task_count"]),
        "base_channels": int(args.base_channels),
        "cond_dim": int(args.cond_dim),
        "manifest": str(Path(args.manifest)),
        "views": ["robot0_agentview_left", "robot0_agentview_right"],
        "condition_on_robocasa_task_index": bool(args.condition_on_robocasa_task_index),
    }
    torch.save(checkpoint, out_dir / "rgb_refiner.pt")
    best_checkpoint = dict(checkpoint)
    if best_state is not None:
        best_checkpoint["state_dict"] = best_state
        best_checkpoint["best_step"] = int(best_step)
        best_checkpoint["best_psnr"] = float(best_psnr)
    torch.save(best_checkpoint, out_dir / "rgb_refiner_best.pt")
    metrics = {
        "checkpoint": str(out_dir / "rgb_refiner.pt"),
        "best_checkpoint": str(out_dir / "rgb_refiner_best.pt"),
        "best_step": int(best_step),
        "best_psnr": float(best_psnr),
        "val": _eval(vae, model, val, device, int(args.batch_size)),
        "train_samples": len(train),
        "val_samples": len(val),
        "train_demos_per_task": int(args.train_demos_per_task),
        "val_episode_ids": [int(ep) for ep in args.val_episode_id],
        "robocasa_task_indices": [int(idx) for idx in args.robocasa_task_index],
        "frame_stride": int(args.frame_stride),
        "base_channels": int(args.base_channels),
        "cond_dim": int(args.cond_dim),
        "train_seconds": time.time() - started,
    }
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _loss(
    vae: RoboCasaVAEWorldModel,
    model: RoboCasaRGBRefiner,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    target = _next_target(batch)
    with torch.no_grad():
        latent = vae.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
        next_latent, _ = vae.step(latent, batch["action"], batch["task_id"])
        prior = vae.decode(next_latent, batch["task_id"])
    pred = model(prior, next_latent, batch["action"], batch["task_id"])
    mse = F.mse_loss(pred, target)
    l1 = F.l1_loss(pred, target)
    loss = mse + 0.1 * l1
    return loss, {"loss": float(loss.detach().cpu()), "mse": float(mse.detach().cpu()), "l1": float(l1.detach().cpu())}


def _eval(
    vae: RoboCasaVAEWorldModel,
    model: RoboCasaRGBRefiner,
    data,
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_prior_mse = 0.0
    count = 0
    with torch.no_grad():
        for start in range(0, len(data), batch_size):
            idx = np.arange(start, min(len(data), start + batch_size))
            batch = _batch(data, idx, device)
            target = _next_target(batch)
            latent = vae.encode(batch["agent"], batch["wrist"], batch["proprio"], batch["task_id"])
            next_latent, _ = vae.step(latent, batch["action"], batch["task_id"])
            prior = vae.decode(next_latent, batch["task_id"])
            pred = model(prior, next_latent, batch["action"], batch["task_id"])
            n = len(idx)
            total_mse += float(F.mse_loss(pred, target).detach().cpu()) * n
            total_prior_mse += float(F.mse_loss(prior, target).detach().cpu()) * n
            count += n
    model.train()
    mse = total_mse / max(1, count)
    prior_mse = total_prior_mse / max(1, count)
    return {
        "mse": mse,
        "psnr": -10.0 * math.log10(max(mse, 1e-12)),
        "prior_mse": prior_mse,
        "prior_psnr": -10.0 * math.log10(max(prior_mse, 1e-12)),
    }


def _next_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    agent = batch["next_agent"] / 255.0 if batch["next_agent"].max() > 1.5 else batch["next_agent"]
    wrist = batch["next_wrist"] / 255.0 if batch["next_wrist"].max() > 1.5 else batch["next_wrist"]
    return torch.cat([agent, wrist], dim=1).clamp(0.0, 1.0)


if __name__ == "__main__":
    main()
