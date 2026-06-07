from __future__ import annotations

from statistics import mean
from typing import Any


def aggregate_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not episodes:
        raise ValueError("cannot aggregate zero episodes")

    success_rate = mean(1.0 if ep["success"] else 0.0 for ep in episodes)
    catastrophe_rate = mean(1.0 if ep["catastrophe"] else 0.0 for ep in episodes)
    avg_return = mean(float(ep["return"]) for ep in episodes)
    avg_distance = mean(float(ep["distance"]) for ep in episodes)
    avg_energy = mean(float(ep["energy"]) for ep in episodes)
    avg_jerk = mean(float(ep["jerk"]) for ep in episodes)
    joint_violations = sum(int(ep["joint_limit_violations"]) for ep in episodes)
    torque_violations = sum(int(ep["torque_limit_violations"]) for ep in episodes)

    score = (
        success_rate
        + 0.02 * avg_return
        - 2.0 * catastrophe_rate
        - 0.001 * avg_energy
        - 0.001 * avg_jerk
        - 0.02 * joint_violations
        - 0.01 * torque_violations
        - 0.1 * avg_distance
    )

    result = {
        "episodes": len(episodes),
        "score": score,
        "success_rate": success_rate,
        "catastrophe_rate": catastrophe_rate,
        "avg_return": avg_return,
        "avg_distance": avg_distance,
        "avg_energy": avg_energy,
        "avg_jerk": avg_jerk,
        "joint_limit_violations": joint_violations,
        "torque_limit_violations": torque_violations,
    }
    for key in ["reach_object", "grasp", "lift", "place"]:
        if key in episodes[0]:
            result[f"{key}_rate"] = mean(1.0 if ep.get(key, False) else 0.0 for ep in episodes)
    for source, target in [
        ("object_distance_min", "avg_object_distance_min"),
        ("place_distance_min", "avg_place_distance_min"),
        ("lift_height_max", "avg_lift_height_max"),
    ]:
        if source in episodes[0]:
            result[target] = mean(float(ep.get(source, 0.0)) for ep in episodes)
    return result
