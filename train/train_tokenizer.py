from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from data.libero_dataset import load_video_npz
from models.tokenizer import TinyVQTokenizer, images_to_tensor
from train.common import batches, device_from_arg, save_checkpoint, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/libero_object5/libero_object5_video.npz")
    parser.add_argument("--out-dir", default="runs/libero/tokenizer")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--codebook-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max-train-seconds", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    train = load_video_npz(Path(args.data), split="train")
    val = load_video_npz(Path(args.data), split="val")
    model = TinyVQTokenizer(args.codebook_size, args.embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    frames = train["frames"]
    started = time.time()
    last_loss = None
    for step, idx in enumerate(batches(len(frames), args.batch_size, args.steps), start=1):
        x = images_to_tensor(frames[idx]).to(device)
        out = model(x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        last_loss = float(out["loss"].detach().cpu())
        if args.log_every > 0 and (step == 1 or step % args.log_every == 0):
            print(f"step={step} tokenizer_loss={last_loss:.6f} elapsed_s={time.time() - started:.1f}", flush=True)
        if args.max_train_seconds > 0 and time.time() - started >= args.max_train_seconds:
            print(f"stopping_at_step={step} elapsed_s={time.time() - started:.1f}", flush=True)
            break

    with torch.no_grad():
        val_x = images_to_tensor(val["frames"][: min(len(val["frames"]), 256)]).to(device)
        val_out = model(val_x)
        val_recon_mse = float(F.mse_loss(val_out["recon"], val_x).cpu())
        val_loss = float(val_out["loss"].cpu())

    out_dir = Path(args.out_dir)
    ckpt = save_checkpoint(
        out_dir,
        "tokenizer.pt",
        model,
        {"codebook_size": args.codebook_size, "embed_dim": args.embed_dim},
    )
    write_metrics(
        out_dir,
        {
            "video_loss": val_recon_mse,
            "tokenizer_loss": val_loss,
            "last_train_loss": last_loss,
            "checkpoint": str(ckpt),
            "train_seconds": time.time() - started,
        },
    )
    print(out_dir / "metrics.json")


if __name__ == "__main__":
    main()
