from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import make_env
from robotbench.metrics import aggregate_episodes


PolicyFn = Callable[[np.ndarray], np.ndarray]


def rollout(
    task: TaskConfig,
    world: str,
    seed: int,
    policy_fn: PolicyFn,
    backend: str = "toy",
) -> dict[str, Any]:
    env = make_env(task=task, world=world, seed=seed, backend=backend)
    obs = env.reset()
    total_reward = 0.0
    energy = 0.0
    jerk = 0.0
    final_info: dict[str, Any] = {}
    stage_flags = {
        "reach_object": False,
        "grasp": False,
        "lift": False,
        "place": False,
    }
    min_object_distance = float("inf")
    min_place_distance = float("inf")
    max_lift_height = 0.0
    for _ in range(task.horizon):
        result = env.step(policy_fn(obs))
        obs = result.obs
        total_reward += result.reward
        energy += float(result.info.get("energy", 0.0))
        jerk += float(result.info.get("jerk", 0.0))
        for key in stage_flags:
            stage_flags[key] = stage_flags[key] or bool(result.info.get(key, False))
        if "object_distance" in result.info:
            min_object_distance = min(min_object_distance, float(result.info["object_distance"]))
        if "place_distance" in result.info:
            min_place_distance = min(min_place_distance, float(result.info["place_distance"]))
        if "lift_height" in result.info:
            max_lift_height = max(max_lift_height, float(result.info["lift_height"]))
        final_info = result.info
        if result.terminated or result.truncated:
            break

    episode = {
        "return": total_reward,
        "success": bool(final_info.get("success", False)),
        "distance": float(final_info.get("distance", 999.0)),
        "energy": energy,
        "jerk": jerk,
        "joint_limit_violations": int(final_info.get("joint_limit_violations", 0)),
        "torque_limit_violations": int(final_info.get("torque_limit_violations", 0)),
        "catastrophe": bool(final_info.get("catastrophe", False)),
    }
    if "curriculum_stage" in final_info:
        episode.update(stage_flags)
        episode["object_distance_min"] = min_object_distance
        episode["place_distance_min"] = min_place_distance
        episode["lift_height_max"] = max_lift_height
        episode["curriculum_stage"] = str(final_info["curriculum_stage"])
    return episode


def evaluate_policy(
    task: TaskConfig,
    world: str,
    seeds: list[int],
    episodes_per_seed: int,
    policy_fn: PolicyFn,
    backend: str = "toy",
) -> dict[str, Any]:
    episodes = []
    for seed in seeds:
        for episode_idx in range(episodes_per_seed):
            episodes.append(
                rollout(
                    task=task,
                    world=world,
                    seed=seed * 10_000 + episode_idx,
                    policy_fn=policy_fn,
                    backend=backend,
                )
            )
    return aggregate_episodes(episodes)
