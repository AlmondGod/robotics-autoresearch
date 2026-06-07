from __future__ import annotations

import argparse
from pathlib import Path

import torch

from data.libero_dataset import load_video_npz
from models.tokenizer import TinyVQTokenizer, images_to_tensor
from models.world_model import NanoVideoGPT
from train.common import batches, device_from_arg, save_checkpoint, write_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/libero_object5/libero_object5_video.npz")
    parser.add_argument("--tokenizer", default="runs/libero/tokenizer/tokenizer.pt")
    parser.add_argument("--out-dir", default="runs/libero/world_model")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--embd", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    tok_ckpt = torch.load(args.tokenizer, map_location=device)
    tokenizer = TinyVQTokenizer(tok_ckpt["codebook_size"], tok_ckpt["embed_dim"]).to(device)
    tokenizer.load_state_dict(tok_ckpt["state_dict"])
    tokenizer.eval()
    train = load_video_npz(Path(args.data), split="train")
    val = load_video_npz(Path(args.data), split="val")
    model = NanoVideoGPT(
        vocab_size=tok_ckpt["codebook_size"],
        n_layer=args.layers,
        n_head=args.heads,
        n_embd=args.embd,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for idx in batches(len(train["frames"]), args.batch_size, args.steps):
        with torch.no_grad():
            z = tokenizer.encode_indices(images_to_tensor(train["frames"][idx]).to(device)).reshape(len(idx), -1)
            z_next = tokenizer.encode_indices(images_to_tensor(train["next_frames"][idx]).to(device)).reshape(len(idx), -1)
        task_id = torch.as_tensor(train["task_id"][idx], dtype=torch.long, device=device)
        _, loss = model(z, task_id, targets=z_next)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        n = min(len(val["frames"]), 256)
        z = tokenizer.encode_indices(images_to_tensor(val["frames"][:n]).to(device)).reshape(n, -1)
        z_next = tokenizer.encode_indices(images_to_tensor(val["next_frames"][:n]).to(device)).reshape(n, -1)
        task_id = torch.as_tensor(val["task_id"][:n], dtype=torch.long, device=device)
        _, val_loss = model(z, task_id, targets=z_next)

    out_dir = Path(args.out_dir)
    ckpt = save_checkpoint(out_dir, "world_model.pt", model, {"vocab_size": tok_ckpt["codebook_size"]})
    write_metrics(out_dir, {"val_video_nll": float(val_loss.cpu()), "checkpoint": str(ckpt)})
    print(out_dir / "metrics.json")


if __name__ == "__main__":
    main()
