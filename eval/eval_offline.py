from __future__ import annotations

import argparse
import json
from pathlib import Path


METHOD_FIELDS = {
    "BC baseline": ("bc", ["bc_loss", "success_rate"]),
    "BC + tokenizer": ("tokenizer", ["video_loss", "bc_loss", "success_rate"]),
    "BC + world loss": ("world", ["val_video_nll", "bc_loss", "success_rate"]),
    "BC + inverse loss": ("world_inverse", ["val_video_nll", "action_mse", "bc_loss", "success_rate"]),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", default="runs/libero")
    parser.add_argument("--out", default="runs/libero/table.json")
    args = parser.parse_args()

    rows = []
    for name, (method_key, fields) in METHOD_FIELDS.items():
        row = {"method": name}
        metrics = _find_method_metrics(Path(args.runs_root), method_key)
        for field in fields:
            row[field] = None if metrics is None else metrics.get(field)
        rows.append(row)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, sort_keys=True))
    print(_markdown_table(rows))


def _find_method_metrics(root: Path, method: str) -> dict | None:
    matches = []
    for path in root.glob("**/metrics.json"):
        data = json.loads(path.read_text())
        if data.get("method") == method:
            matches.append((path.stat().st_mtime, data))
    if not matches:
        return None
    return sorted(matches, key=lambda item: item[0])[-1][1]


def _markdown_table(rows: list[dict]) -> str:
    columns = ["method", "video_loss", "action_mse", "bc_loss", "success_rate"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join("-" if row.get(col) is None else str(row.get(col)) for col in columns) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
