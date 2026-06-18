from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot AutoroboBench RoboCasa BC-5 agent artifacts.")
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    ledger = _read_jsonl(Path(args.ledger))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "experiment_list.svg").write_text(_experiment_svg(ledger))
    (out_dir / "loss_curves.svg").write_text(_loss_svg(ledger))
    (out_dir / "summary.json").write_text(json.dumps(_summary(ledger), indent=2, sort_keys=True) + "\n")
    print(json.dumps({"experiments": len(ledger), "out_dir": str(out_dir)}, indent=2))


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _summary(rows: list[dict]) -> dict:
    best = None
    for row in rows:
        score = row.get("success_rate", row.get("val_loss"))
        if score is None:
            continue
        if best is None or float(row.get("success_rate", 0.0)) > float(best.get("success_rate", 0.0)):
            best = row
    return {"experiments": len(rows), "best": best}


def _experiment_svg(rows: list[dict]) -> str:
    width = 1100
    row_h = 30
    height = max(180, 70 + row_h * max(1, len(rows)))
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" fill="#111">AutoroboBench RoboCasa BC-5 Experiments</text>',
        '<text x="24" y="58" font-family="Arial" font-size="12" fill="#555">success rate bars use the committed frozen eval split subset used by the run</text>',
    ]
    x0 = 24
    y0 = 84
    bar_x = 650
    bar_w = 360
    for idx, row in enumerate(rows):
        y = y0 + idx * row_h
        success = float(row.get("success_rate", 0.0) or 0.0)
        val_loss = row.get("val_loss")
        accepted = bool(row.get("accepted", False))
        color = "#20b26b" if accepted else "#9ca3af"
        label = str(row.get("change", row.get("name", f"experiment {idx}")))[:86]
        lines.extend(
            [
                f'<text x="{x0}" y="{y}" font-family="Arial" font-size="13" fill="#111">{_esc(idx)}. {_esc(label)}</text>',
                f'<rect x="{bar_x}" y="{y - 13}" width="{bar_w}" height="14" rx="2" fill="#eef2f7"/>',
                f'<rect x="{bar_x}" y="{y - 13}" width="{bar_w * max(0.0, min(1.0, success)):.1f}" height="14" rx="2" fill="{color}"/>',
                f'<text x="{bar_x + bar_w + 12}" y="{y}" font-family="Arial" font-size="12" fill="#111">sr={success:.3f}</text>',
            ]
        )
        if val_loss is not None:
            lines.append(f'<text x="{bar_x + bar_w + 86}" y="{y}" font-family="Arial" font-size="12" fill="#555">val={float(val_loss):.4f}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _loss_svg(rows: list[dict]) -> str:
    series = []
    for row in rows:
        history_path = row.get("history")
        if not history_path:
            continue
        path = Path(history_path)
        if not path.exists():
            continue
        history = json.loads(path.read_text())
        points = [(float(h["step"]), float(h["train_loss"])) for h in history if "train_loss" in h]
        val_points = [(float(h["step"]), float(h["val_loss"])) for h in history if "val_loss" in h]
        if points:
            series.append((row, points, val_points))
    width = 1100
    height = 520
    left, top, plot_w, plot_h = 70, 64, 980, 380
    all_points = [pt for _, train, val in series for pt in train + val]
    if not all_points:
        return _empty_svg(width, height, "No loss history found")
    max_x = max(x for x, _ in all_points)
    min_y = min(y for _, y in all_points)
    max_y = max(y for _, y in all_points)
    if max_y <= min_y:
        max_y = min_y + 1.0

    def sx(x: float) -> float:
        return left + (x / max(1.0, max_x)) * plot_w

    def sy(y: float) -> float:
        return top + (1.0 - (y - min_y) / (max_y - min_y)) * plot_h

    palette = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2"]
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="34" font-family="Arial" font-size="22" fill="#111">Training Loss Over Time</text>',
        f'<rect x="{left}" y="{top}" width="{plot_w}" height="{plot_h}" fill="#fbfdff" stroke="#d1d5db"/>',
        f'<text x="{left}" y="{top + plot_h + 34}" font-family="Arial" font-size="12" fill="#555">step</text>',
        f'<text x="20" y="{top + 12}" font-family="Arial" font-size="12" fill="#555">loss</text>',
    ]
    for tick in range(6):
        frac = tick / 5
        y = top + frac * plot_h
        value = max_y - frac * (max_y - min_y)
        lines.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        lines.append(f'<text x="8" y="{y + 4:.1f}" font-family="Arial" font-size="11" fill="#555">{value:.3f}</text>')
    for idx, (row, train, val) in enumerate(series):
        color = palette[idx % len(palette)]
        name = str(row.get("name", row.get("change", f"exp {idx}")))[:36]
        lines.append(_polyline([(sx(x), sy(y)) for x, y in train], color, 2.0))
        if val:
            lines.append(_polyline([(sx(x), sy(y)) for x, y in val], "#111827", 2.4, dash="5 4"))
        legend_y = top + plot_h + 58 + 18 * idx
        lines.append(f'<line x1="{left + 80}" y1="{legend_y}" x2="{left + 120}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>')
        lines.append(f'<text x="{left + 128}" y="{legend_y + 4}" font-family="Arial" font-size="12" fill="#111">{_esc(name)}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _polyline(points: list[tuple[float, float]], color: str, width: float, dash: str | None = None) -> str:
    attrs = f'fill="none" stroke="{color}" stroke-width="{width}"'
    if dash:
        attrs += f' stroke-dasharray="{dash}"'
    encoded = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
    return f'<polyline points="{encoded}" {attrs}/>'


def _empty_svg(width: int, height: int, message: str) -> str:
    return "\n".join(
        [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
            '<rect width="100%" height="100%" fill="white"/>',
            f'<text x="24" y="40" font-family="Arial" font-size="20" fill="#111">{_esc(message)}</text>',
            "</svg>",
        ]
    )


def _esc(value: object) -> str:
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
