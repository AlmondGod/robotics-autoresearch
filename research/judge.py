from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    args = parser.parse_args()
    decision = judge(Path(args.baseline), Path(args.candidate))
    print(json.dumps(decision, indent=2, sort_keys=True))


def judge(baseline_dir: Path, candidate_dir: Path) -> dict:
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    candidate = json.loads((candidate_dir / "metrics.json").read_text())
    b_success = _metric(baseline, "success_rate")
    c_success = _metric(candidate, "success_rate")
    if c_success is not None and b_success is not None and c_success != b_success:
        accepted = c_success > b_success
        reason = "success_rate"
    else:
        b_bc = _metric(baseline, "bc_loss", default=float("inf"))
        c_bc = _metric(candidate, "bc_loss", default=float("inf"))
        accepted = c_bc < b_bc
        reason = "bc_loss_tiebreak"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "baseline": {"success_rate": b_success, "bc_loss": baseline.get("bc_loss")},
        "candidate": {"success_rate": c_success, "bc_loss": candidate.get("bc_loss")},
    }


def _metric(metrics: dict, key: str, default=None):
    value = metrics.get(key, default)
    return default if value is None else value


if __name__ == "__main__":
    main()
