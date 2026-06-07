from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", default="runs/research_log.jsonl")
    parser.add_argument("--out", default="runs/progress.svg")
    parser.add_argument("--backend", default="")
    parser.add_argument("--task", default="")
    args = parser.parse_args()
    plot_progress(Path(args.log), Path(args.out), backend=args.backend, task=args.task)


def plot_progress(log_path: Path, out_path: Path, backend: str = "", task: str = "") -> None:
    if not log_path.exists():
        raise FileNotFoundError(f"missing research log: {log_path}")
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    if backend:
        rows = [row for row in rows if row.get("backend") == backend]
    if task:
        rows = [row for row in rows if row.get("task") == task]
    if not rows:
        raise ValueError(f"empty research log: {log_path}")

    x = list(range(len(rows)))
    costs = [-float(row["eval_score"]) for row in rows]
    accepted = [row.get("accepted") for row in rows]
    labels = [
        row.get("change_note", "")[:36]
        for row in rows
    ]

    if out_path.suffix.lower() == ".png":
        try:
            _plot_png(x=x, costs=costs, labels=labels, accepted=accepted, out_path=out_path)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PNG output requires matplotlib. Use --out runs/progress.svg "
                "or install the optional plot dependency."
            ) from exc
    else:
        _plot_svg(costs=costs, labels=labels, accepted=accepted, out_path=out_path)
    print(out_path)


def _plot_png(
    x: list[int],
    costs: list[float],
    labels: list[str],
    accepted: list[bool | None],
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    out_path.parent.mkdir(parents=True, exist_ok=True)
    colors = [_status_color(status) for status in accepted]
    best_x, best_y = _running_best_points(costs, accepted)
    plt.figure(figsize=(max(8, len(costs) * 1.2), 4.8))
    plt.scatter(x, costs, c=colors, zorder=3)
    plt.step(best_x, best_y, where="post", linewidth=2, color="#2ca02c")
    plt.xticks(x, labels, rotation=35, ha="right")
    plt.ylabel("Eval cost (-score, lower is better)")
    plt.xlabel("Experiment")
    plt.title(_title(accepted))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)


def _plot_svg(
    costs: list[float],
    labels: list[str],
    accepted: list[bool | None],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    width = max(900, 85 * len(costs))
    height = 520
    margin_left = 82
    margin_right = 30
    margin_top = 50
    margin_bottom = 150
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    min_cost = min(costs)
    max_cost = max(costs)
    if min_cost == max_cost:
        min_cost -= 0.5
        max_cost += 0.5
    padding = (max_cost - min_cost) * 0.08
    min_axis = min_cost - padding
    max_axis = max_cost + padding

    def point(idx: int, cost: float) -> tuple[float, float]:
        x = margin_left + (plot_w * idx / max(1, len(costs) - 1))
        y = margin_top + plot_h * (1.0 - (cost - min_axis) / (max_axis - min_axis))
        return x, y

    points = [point(idx, cost) for idx, cost in enumerate(costs)]
    best_x, best_y = _running_best_points(costs, accepted)
    best_points = [point(idx, cost) for idx, cost in zip(best_x, best_y, strict=True)]
    best_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in best_points)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="28" font-family="Arial" font-size="20" font-weight="700">{_xml_escape(_title(accepted))}</text>',
        '<text x="24" y="48" font-family="Arial" font-size="12"><tspan fill="#b8b8b8">discarded</tspan> / <tspan fill="#2ecc71">kept</tspan> / <tspan fill="#2ecc71">running best</tspan></text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{width - margin_right}" y2="{margin_top + plot_h}" stroke="#222"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#222"/>',
    ]
    tick_count = 5
    for tick_idx in range(tick_count):
        value = min_axis + (max_axis - min_axis) * tick_idx / (tick_count - 1)
        y = margin_top + plot_h * (1.0 - (value - min_axis) / (max_axis - min_axis))
        parts.extend(
            [
                f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e5e5e5"/>',
                f'<text x="{margin_left - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#444">{value:.3f}</text>',
            ]
        )
    parts.append(f'<polyline points="{best_polyline}" fill="none" stroke="#2ecc71" stroke-width="3"/>')
    for idx, ((x, y), cost, label, status) in enumerate(
        zip(points, costs, labels, accepted, strict=True)
    ):
        safe_label = _xml_escape(label.replace("\n", " "))
        color = _status_color(status)
        radius = 5.5 if status is True else 3.5
        opacity = "1.0" if status is True else "0.35"
        parts.extend(
            [
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" opacity="{opacity}" stroke="#174b2b" stroke-width="{1 if status is True else 0}"/>',
            ]
        )
        if status is True:
            parts.extend(
                [
                    f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial" font-size="11" fill="#174b2b">{cost:.3f}</text>',
                    f'<text x="{x + 5:.1f}" y="{y - 18:.1f}" transform="rotate(-32 {x + 5:.1f} {y - 18:.1f})" font-family="Arial" font-size="11" fill="#278c49">{safe_label}</text>',
                ]
            )
    parts.extend(
        [
            f'<text x="22" y="{margin_top + plot_h / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_h / 2:.1f})" font-family="Arial" font-size="12">Eval cost (-score, lower is better)</text>',
            f'<text x="{margin_left + plot_w / 2:.1f}" y="{height - 16}" text-anchor="middle" font-family="Arial" font-size="12">Experiment #</text>',
            "</svg>",
        ]
    )
    out_path.write_text("\n".join(parts) + "\n")


def _running_best_points(costs: list[float], accepted: list[bool | None]) -> tuple[list[int], list[float]]:
    best = costs[0]
    xs = [0]
    ys = [best]
    for idx, (cost, status) in enumerate(zip(costs[1:], accepted[1:], strict=True), start=1):
        if status is True and cost < best:
            xs.extend([idx, idx])
            ys.extend([best, cost])
            best = cost
        else:
            xs.append(idx)
            ys.append(best)
    return xs, ys


def _title(accepted: list[bool | None]) -> str:
    kept = sum(1 for status in accepted if status is True)
    return f"Autoresearch Progress: {len(accepted)} Experiments, {kept} Kept Improvements"


def _xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _status_color(status: bool | None) -> str:
    if status is True:
        return "#2ecc71"
    if status is False:
        return "#b8b8b8"
    return "#d8d8d8"


if __name__ == "__main__":
    main()
