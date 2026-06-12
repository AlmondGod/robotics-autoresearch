from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from eval.eval_world_model_ranking import compute_metrics
from eval.eval_world_model_ranking import _write_svg as write_ranking_svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", required=True, help="Folder containing candidate_id/episode_id.mp4 or flat *.mp4.")
    parser.add_argument("--sim-archive", default="runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl")
    parser.add_argument("--reference-root", default="", help="Optional folder containing matching reference videos.")
    parser.add_argument("--out", default="runs/robocasa/cosmos_eval/candidate_scores.jsonl")
    parser.add_argument("--metrics-out", default="runs/robocasa/cosmos_eval/correlation_metrics.json")
    parser.add_argument("--plot", default="runs/robocasa/cosmos_eval/correlation.svg")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    sim_rows = _load_sim_rows(Path(args.sim_archive))
    gen_root = Path(args.generations)
    ref_root = Path(args.reference_root) if args.reference_root else None
    out_rows = []
    for row in sim_rows:
        cid = int(row["experiment"])
        videos = _candidate_videos(gen_root, cid)
        if not videos:
            continue
        per_video = []
        for video in videos:
            ref = _matching_reference(ref_root, video, cid) if ref_root else None
            per_video.append(_video_metrics(video, ref))
        agg = _aggregate(per_video)
        learned_score = _quality_score(agg)
        out_row = {
            "candidate_id": cid,
            "change": row.get("change"),
            "learned_score": learned_score,
            "sim_success": float(row.get("success_rate", row.get("score", 0.0))),
            "sim_successes": row.get("successes"),
            "videos": [str(path) for path in videos],
            **agg,
        }
        out_rows.append(out_row)
        print(json.dumps(out_row), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(row, sort_keys=True) for row in out_rows) + "\n")
    metrics = compute_metrics(out_rows, top_k=int(args.top_k))
    Path(args.metrics_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_out).write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
    write_ranking_svg(out_rows, metrics, Path(args.plot))
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _load_sim_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _candidate_videos(root: Path, candidate_id: int) -> list[Path]:
    nested = sorted((root / f"candidate_{candidate_id:03d}").glob("*.mp4"))
    if nested:
        return nested
    return sorted(root.glob(f"*{candidate_id:03d}*.mp4"))


def _matching_reference(ref_root: Path | None, generated: Path, candidate_id: int) -> Path | None:
    if ref_root is None:
        return None
    direct = ref_root / generated.name
    if direct.exists():
        return direct
    nested = ref_root / f"candidate_{candidate_id:03d}" / generated.name
    if nested.exists():
        return nested
    return None


def _video_metrics(video: Path, reference: Path | None) -> dict[str, float]:
    frames = _read_video(video)
    if len(frames) == 0:
        return {"sharpness": 0.0, "motion": 0.0, "nonblank": 0.0, "psnr": 0.0}
    sharpness = float(np.mean([_laplacian_var(frame) for frame in frames]))
    motion = float(np.mean(np.abs(np.diff(frames.astype(np.float32) / 255.0, axis=0)))) if len(frames) > 1 else 0.0
    nonblank = float(np.mean(np.std(frames.astype(np.float32) / 255.0, axis=(1, 2, 3)) > 0.02))
    out = {"sharpness": sharpness, "motion": motion, "nonblank": nonblank}
    if reference is not None and reference.exists():
        ref = _read_video(reference, max_frames=len(frames))
        n = min(len(frames), len(ref))
        if n > 0:
            mse = float(np.mean((frames[:n].astype(np.float32) / 255.0 - ref[:n].astype(np.float32) / 255.0) ** 2))
            out["psnr"] = -10.0 * math.log10(max(mse, 1e-12))
    return out


def _read_video(path: Path, max_frames: int | None = None) -> np.ndarray:
    frames = []
    for idx, frame in enumerate(iio.imiter(path)):
        if max_frames is not None and idx >= max_frames:
            break
        image = np.asarray(frame)[..., :3]
        frames.append(image.astype(np.uint8))
    return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), dtype=np.uint8)


def _laplacian_var(frame: np.ndarray) -> float:
    gray = frame.astype(np.float32).mean(axis=-1) / 255.0
    lap = -4 * gray
    lap[1:, :] += gray[:-1, :]
    lap[:-1, :] += gray[1:, :]
    lap[:, 1:] += gray[:, :-1]
    lap[:, :-1] += gray[:, 1:]
    return float(lap.var())


def _aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted({key for row in rows for key in row})
    return {key: float(np.mean([row[key] for row in rows if key in row])) for key in keys}


def _quality_score(metrics: dict[str, float]) -> float:
    if "psnr" in metrics:
        return float(metrics["psnr"])
    return float(metrics.get("nonblank", 0.0) + 0.5 * metrics.get("motion", 0.0) + 0.1 * metrics.get("sharpness", 0.0))


if __name__ == "__main__":
    main()
