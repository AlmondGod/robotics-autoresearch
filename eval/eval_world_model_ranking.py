from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True, help="JSONL candidate archive.")
    parser.add_argument("--out", default="runs/robocasa5/evaluator_ranking_metrics.json")
    parser.add_argument("--plot", default="runs/robocasa5/evaluator_correlation.svg")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    rows = _load_rows(Path(args.candidates))
    metrics = compute_metrics(rows, top_k=args.top_k)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")

    plot = Path(args.plot)
    plot.parent.mkdir(parents=True, exist_ok=True)
    _write_svg(rows, metrics, plot)

    print(json.dumps(metrics, indent=2, sort_keys=True))


def compute_metrics(rows: list[dict], top_k: int = 5) -> dict:
    usable = [
        row
        for row in rows
        if row.get("learned_score") is not None and row.get("sim_success") is not None
    ]
    if len(usable) < 2:
        raise ValueError("need at least two rows with learned_score and sim_success")

    learned = [float(row["learned_score"]) for row in usable]
    sim = [float(row["sim_success"]) for row in usable]
    pearson = _pearson(learned, sim)
    spearman = _pearson(_ranks(learned), _ranks(sim))
    top_hit = _top_k_hit_rate(usable, top_k=top_k)

    learned_rollouts = sum(float(row.get("learned_eval_rollouts") or 0.0) for row in usable)
    learned_seconds = sum(float(row.get("learned_eval_seconds") or 0.0) for row in usable)
    sim_rollouts = sum(float(row.get("sim_eval_rollouts") or 0.0) for row in usable)
    sim_seconds = sum(float(row.get("sim_eval_seconds") or 0.0) for row in usable)
    learned_rps = learned_rollouts / learned_seconds if learned_seconds > 0 else None
    sim_rps = sim_rollouts / sim_seconds if sim_seconds > 0 else None
    speedup = learned_rps / sim_rps if learned_rps is not None and sim_rps not in {None, 0.0} else None

    best_sim = max(usable, key=lambda row: float(row["sim_success"]))
    best_learned = max(usable, key=lambda row: float(row["learned_score"]))

    return {
        "n": len(usable),
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "top_k": top_k,
        "top_k_hit_rate": top_hit,
        "learned_rollouts_per_second": learned_rps,
        "sim_rollouts_per_second": sim_rps,
        "speedup_ratio": speedup,
        "best_sim_candidate_id": best_sim.get("candidate_id"),
        "best_sim_success": float(best_sim["sim_success"]),
        "best_learned_candidate_id": best_learned.get("candidate_id"),
        "best_learned_score": float(best_learned["learned_score"]),
    }


def _load_rows(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError("length mismatch")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    dx = [x - mx for x in xs]
    dy = [y - my for y in ys]
    denom = math.sqrt(sum(x * x for x in dx) * sum(y * y for y in dy))
    if denom == 0:
        return 0.0
    return sum(x * y for x, y in zip(dx, dy, strict=True)) / denom


def _ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        avg = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            ranks[order[k]] = avg
        i = j
    return ranks


def _top_k_hit_rate(rows: list[dict], top_k: int) -> float:
    k = min(top_k, len(rows))
    by_learned = sorted(rows, key=lambda row: float(row["learned_score"]), reverse=True)[:k]
    best_sim = max(rows, key=lambda row: float(row["sim_success"]))
    best_id = best_sim.get("candidate_id")
    if best_id is not None:
        return 1.0 if any(row.get("candidate_id") == best_id for row in by_learned) else 0.0
    return 1.0 if best_sim in by_learned else 0.0


def _write_svg(rows: list[dict], metrics: dict, out: Path) -> None:
    usable = [
        row
        for row in rows
        if row.get("learned_score") is not None and row.get("sim_success") is not None
    ]
    xs = [float(row["learned_score"]) for row in usable]
    ys = [float(row["sim_success"]) for row in usable]
    min_x, max_x = _bounds(xs)
    min_y, max_y = _bounds(ys)

    width = 900
    height = 560
    left = 82
    right = 28
    top = 72
    bottom = 72
    plot_w = width - left - right
    plot_h = height - top - bottom

    def point(xv: float, yv: float) -> tuple[float, float]:
        x = left + plot_w * (xv - min_x) / (max_x - min_x)
        y = top + plot_h * (1.0 - (yv - min_y) / (max_y - min_y))
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">Learned Evaluator Ranking</text>',
        (
            f'<text x="24" y="58" font-family="Arial" font-size="13" fill="#444">'
            f'n={metrics["n"]}, Spearman={metrics["spearman_rho"]:.3f}, '
            f'Pearson={metrics["pearson_r"]:.3f}, top-{metrics["top_k"]} hit={metrics["top_k_hit_rate"]:.1f}'
            '</text>'
        ),
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#222"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222"/>',
    ]
    for i in range(5):
        gx = left + plot_w * i / 4
        gy = top + plot_h * i / 4
        parts.append(f'<line x1="{gx:.1f}" y1="{top}" x2="{gx:.1f}" y2="{top + plot_h}" stroke="#e5e5e5"/>')
        parts.append(f'<line x1="{left}" y1="{gy:.1f}" x2="{width - right}" y2="{gy:.1f}" stroke="#e5e5e5"/>')

    for row in usable:
        x, y = point(float(row["learned_score"]), float(row["sim_success"]))
        label = _xml_escape(str(row.get("candidate_id") or row.get("change") or "candidate"))
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="#2f80ed" opacity="0.75"/>',
                f'<title>{label}</title>',
            ]
        )
    parts.extend(
        [
            f'<text x="{left + plot_w / 2:.1f}" y="{height - 24}" text-anchor="middle" font-family="Arial" font-size="13">learned evaluator predicted success</text>',
            f'<text x="24" y="{top + plot_h / 2:.1f}" transform="rotate(-90 24 {top + plot_h / 2:.1f})" text-anchor="middle" font-family="Arial" font-size="13">actual RoboCasa sim success</text>',
            "</svg>",
        ]
    )
    out.write_text("\n".join(parts) + "\n")


def _bounds(values: list[float]) -> tuple[float, float]:
    lo = min(values)
    hi = max(values)
    if lo == hi:
        lo -= 0.5
        hi += 0.5
    pad = (hi - lo) * 0.08
    return lo - pad, hi + pad


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


if __name__ == "__main__":
    main()

