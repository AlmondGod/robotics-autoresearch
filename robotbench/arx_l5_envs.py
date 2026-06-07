from __future__ import annotations

from pathlib import Path

import numpy as np

from robotbench.config import TaskConfig
from robotbench.envs import StepResult
from robotbench.safety import SafetyTracker


MENAGERIE_DIR = Path("third_party/mujoco_menagerie")
ARX_L5_DIR = MENAGERIE_DIR / "arx_l5"
GENERATED_SCENE = ARX_L5_DIR / "robot_autoresearch_scene.xml"


class ArxL5CameraEnv:
    """Menagerie ARX L5 single-arm reaching with one camera observation."""

    camera_height = 24
    camera_width = 24
    image_obs_dim = camera_height * camera_width
    proprio_dim = 21
    obs_dim = image_obs_dim + proprio_dim
    act_dim = 7

    def __init__(self, task: TaskConfig, world: str, seed: int):
        try:
            import mujoco
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ARX L5 backend requires `pip install -e .[mujoco,ppo]` or `pip install mujoco torch`."
            ) from exc

        if not (ARX_L5_DIR / "arx_l5.xml").exists():
            raise FileNotFoundError(
                "Missing Menagerie ARX L5 assets. Run: "
                "python scripts/fetch_menagerie.py --model arx_l5"
            )

        _ensure_generated_scene()
        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(GENERATED_SCENE))
        self.data = mujoco.MjData(self.model)
        self.task = task
        self.world = world
        self.params = getattr(task, world)
        self.rng = np.random.default_rng(seed)
        self.safety = SafetyTracker(action_limit=task.action_limit)
        self.t = 0
        self.prev_action = np.zeros(self.act_dim)
        self.obs_renderer = mujoco.Renderer(self.model, height=self.camera_height, width=self.camera_width)

        if self.model.nu != self.act_dim:
            raise ValueError(f"expected 7 ARX L5 actuators, got {self.model.nu}")
        self.ctrlrange = self.model.actuator_ctrlrange.copy()
        self.current_ctrl = np.zeros(self.act_dim)
        self.home_key = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_KEY, "home")
        self.target_body = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            "autoresearch_target",
        )
        self.object_body = self.mujoco.mj_name2id(
            self.model,
            self.mujoco.mjtObj.mjOBJ_BODY,
            "autoresearch_object",
        )
        self.left_finger = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "link7")
        self.right_finger = self.mujoco.mj_name2id(self.model, self.mujoco.mjtObj.mjOBJ_BODY, "link8")
        self.grasped = False
        self.grasp_offset = np.zeros(3)
        self.object_start_z = 0.0

    def reset(self) -> np.ndarray:
        self.mujoco.mj_resetData(self.model, self.data)
        if self.home_key >= 0:
            self.mujoco.mj_resetDataKeyframe(self.model, self.data, self.home_key)
        self.current_ctrl = self.data.ctrl.copy()
        self.t = 0
        self.safety.reset()
        self.prev_action = np.zeros(self.act_dim)
        self.grasped = False
        self.grasp_offset = np.zeros(3)
        self.object_start_z = 0.0
        self.mujoco.mj_forward(self.model, self.data)

        gripper = self._gripper_pos()
        if self.task.name == "pick_place":
            self._reset_pick_place(gripper)
        else:
            if self.world == "eval":
                offset = self.rng.uniform([0.08, -0.08, -0.05], [0.22, 0.08, 0.08])
            else:
                offset = self.rng.uniform([0.08, -0.06, -0.04], [0.18, 0.06, 0.06])
            self.model.body_pos[self.target_body] = gripper + offset
            self.model.body_pos[self.object_body] = np.array([0.0, 0.0, -0.2])
        self.mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> StepResult:
        self.t += 1
        raw_action = np.asarray(action, dtype=np.float64).reshape(-1)
        if raw_action.size < self.act_dim:
            raw_action = np.pad(raw_action, (0, self.act_dim - raw_action.size))
        raw_action = raw_action[: self.act_dim]
        action = np.clip(raw_action, -self.task.action_limit, self.task.action_limit)
        self.safety.observe(raw_action=raw_action, clipped_action=action)

        train_scale = float(self.params.get("control_scale", 0.05))
        eval_scale = 0.8 * train_scale
        delta_scale = train_scale if self.world == "train" else eval_scale
        ctrl_delta = delta_scale * action
        lo = self.ctrlrange[:, 0]
        hi = self.ctrlrange[:, 1]
        self.current_ctrl = np.clip(self.current_ctrl + ctrl_delta, lo, hi)
        self.data.ctrl[:] = self.current_ctrl

        frame_skip = int(self.params.get("frame_skip", 8))
        for _ in range(frame_skip):
            self.mujoco.mj_step(self.model, self.data)
            if self.task.name == "pick_place":
                self._update_pick_place_object()

        dist = self._distance_to_goal()
        success = self._success(dist)
        energy = float(np.sum(np.square(action)))
        jerk = float(np.sum(np.square(action - self.prev_action)))
        reward = self._reward(dist, energy, jerk, success)
        self.prev_action = action
        truncated = self.t >= self.task.horizon
        info = {
            "success": success,
            "distance": dist,
            "energy": energy,
            "jerk": jerk,
            **self.safety.snapshot(),
            **self._stage_info(),
        }
        return StepResult(self._obs(), float(reward), bool(success), truncated, info)

    def render_rgb(self, width: int = 720, height: int = 720) -> np.ndarray:
        renderer = self.mujoco.Renderer(self.model, height=height, width=width)
        renderer.update_scene(self.data, camera="overview_cam")
        image = renderer.render()
        renderer.close()
        return image

    def _obs(self) -> np.ndarray:
        self.obs_renderer.update_scene(self.data, camera="wrist_cam")
        image = self.obs_renderer.render()
        gray = (
            0.299 * image[:, :, 0]
            + 0.587 * image[:, :, 1]
            + 0.114 * image[:, :, 2]
        ) / 255.0
        noise = float(self.params.get("observation_noise", 0.0))
        if noise:
            gray = gray + self.rng.normal(0.0, noise, size=gray.shape)
        image_obs = np.clip(gray, 0.0, 1.0).astype(np.float32).reshape(-1)
        return np.concatenate([image_obs, self._proprio_obs()]).astype(np.float32)

    def _proprio_obs(self) -> np.ndarray:
        qpos = self.data.qpos[:8].copy()
        qpos_scale = np.array([3.14, 3.14, 3.14, 1.7, 1.7, 3.14, 0.044, 0.044])
        qpos = np.clip(qpos / qpos_scale, -1.0, 1.0)
        gripper = self._gripper_pos()
        object_pos = self._object_pos()
        target = self.model.body_pos[self.target_body]
        if self.task.name == "pick_place":
            task_delta_a = object_pos - gripper
            task_delta_b = target - object_pos
        else:
            task_delta_a = target - gripper
            task_delta_b = np.zeros(3)
        return np.concatenate([qpos, self.current_ctrl, task_delta_a, task_delta_b]).astype(np.float32)

    def _distance_to_goal(self) -> float:
        if self.task.name != "pick_place":
            return float(np.linalg.norm(self._gripper_pos() - self.model.body_pos[self.target_body]))
        stage = self._pick_place_stage()
        object_distance = self._object_distance()
        if stage == "reach":
            return object_distance
        if stage == "grasp":
            return 0.0 if self.grasped else object_distance
        if stage == "lift":
            if not self.grasped:
                return object_distance + self._lift_threshold()
            return max(0.0, self._lift_threshold() - self._lift_height())
        return self._place_distance()

    def _success(self, dist: float) -> bool:
        if self.task.name != "pick_place":
            return dist <= self.task.success_tolerance
        stage = self._pick_place_stage()
        if stage == "reach":
            return self._object_distance() <= self.task.success_tolerance
        if stage == "grasp":
            return self.grasped
        if stage == "lift":
            return self.grasped and self._lift_height() >= self._lift_threshold()
        return self._place_distance() <= self.task.success_tolerance

    def _reward(self, dist: float, energy: float, jerk: float, success: bool) -> float:
        if self.task.name != "pick_place":
            return -dist - 0.004 * energy - 0.001 * jerk + (1.0 if success else 0.0)
        stage = self._pick_place_stage()
        gripper_to_object = self._object_distance()
        if stage == "reach":
            return -gripper_to_object - 0.003 * energy - 0.001 * jerk + (1.0 if success else 0.0)
        if stage == "grasp":
            return -gripper_to_object + (1.0 if self.grasped else 0.0) - 0.003 * energy - 0.001 * jerk
        if stage == "lift":
            lift_progress = min(self._lift_height() / self._lift_threshold(), 1.0)
            return (
                -0.5 * gripper_to_object
                + (0.5 if self.grasped else 0.0)
                + lift_progress
                - 0.003 * energy
                - 0.001 * jerk
            )
        lift_bonus = 0.15 if self._lift_height() >= self._lift_threshold() else 0.0
        grasp_bonus = 0.25 if self.grasped else 0.0
        place_bonus = 2.0 if success else 0.0
        return (
            -0.6 * gripper_to_object
            -1.0 * dist
            + lift_bonus
            + grasp_bonus
            + place_bonus
            - 0.003 * energy
            - 0.001 * jerk
        )

    def _reset_pick_place(self, gripper: np.ndarray) -> None:
        if self.world == "eval":
            object_offset = self.rng.uniform([0.09, -0.08, -0.04], [0.18, 0.08, 0.04])
            target_offset = self.rng.uniform([0.02, 0.14, -0.02], [0.18, 0.26, 0.06])
        else:
            object_offset = self.rng.uniform([0.09, -0.06, -0.03], [0.16, 0.06, 0.03])
            target_offset = self.rng.uniform([0.04, 0.12, -0.02], [0.16, 0.22, 0.05])
        object_pos = gripper + object_offset
        object_pos[2] = max(object_pos[2], 0.045)
        target_pos = object_pos + target_offset
        target_pos[2] = object_pos[2]
        self.model.body_pos[self.object_body] = object_pos
        self.model.body_pos[self.target_body] = target_pos
        self.object_start_z = float(object_pos[2])

    def _update_pick_place_object(self) -> None:
        gripper = self._gripper_pos()
        object_pos = self._object_pos()
        closed = self.current_ctrl[6] < 0.018
        if self.grasped:
            if not closed:
                self.grasped = False
            else:
                self.model.body_pos[self.object_body] = gripper + self.grasp_offset
        elif closed and np.linalg.norm(gripper - object_pos) < 0.055:
            self.grasped = True
            self.grasp_offset = object_pos - gripper
        if not self.grasped:
            self.model.body_pos[self.object_body, 2] = max(self.model.body_pos[self.object_body, 2], 0.035)
        self.mujoco.mj_forward(self.model, self.data)

    def _object_pos(self) -> np.ndarray:
        return self.model.body_pos[self.object_body].copy()

    def _object_distance(self) -> float:
        return float(np.linalg.norm(self._gripper_pos() - self._object_pos()))

    def _place_distance(self) -> float:
        return float(np.linalg.norm(self._object_pos() - self.model.body_pos[self.target_body]))

    def _lift_height(self) -> float:
        return float(max(0.0, self._object_pos()[2] - self.object_start_z))

    def _lift_threshold(self) -> float:
        return float(self.params.get("lift_threshold", 0.04))

    def _pick_place_stage(self) -> str:
        stage = str(self.params.get("curriculum_stage", "full"))
        return stage if stage in {"reach", "grasp", "lift", "place", "full"} else "full"

    def _stage_info(self) -> dict[str, object]:
        if self.task.name != "pick_place":
            return {}
        object_distance = self._object_distance()
        place_distance = self._place_distance()
        lift_height = self._lift_height()
        lift = self.grasped and lift_height >= self._lift_threshold()
        place = place_distance <= self.task.success_tolerance
        return {
            "curriculum_stage": self._pick_place_stage(),
            "reach_object": object_distance <= self.task.success_tolerance,
            "grasp": bool(self.grasped),
            "lift": bool(lift),
            "place": bool(place),
            "object_distance": object_distance,
            "place_distance": place_distance,
            "lift_height": lift_height,
        }

    def _gripper_pos(self) -> np.ndarray:
        return 0.5 * (self.data.xpos[self.left_finger] + self.data.xpos[self.right_finger])

    def _ee_xy(self) -> np.ndarray:
        return self._gripper_pos()[0:2].copy()

    def _object_xy(self) -> np.ndarray:
        return self._object_pos()[0:2].copy()

    def _target_xy(self) -> np.ndarray:
        return self.model.body_pos[self.target_body, 0:2].copy()


def make_arx_l5_env(task: TaskConfig, world: str, seed: int) -> ArxL5CameraEnv:
    return ArxL5CameraEnv(task=task, world=world, seed=seed)


def _ensure_generated_scene() -> None:
    scene = GENERATED_SCENE.read_text() if GENERATED_SCENE.exists() else (ARX_L5_DIR / "scene.xml").read_text()
    scene = scene.replace(
        '<global azimuth="140" elevation="-20"/>',
        '<global azimuth="140" elevation="-20" offwidth="1280" offheight="720"/>',
    )
    additions = ""
    if "overview_cam" not in scene:
        additions += """
    <camera name="overview_cam" pos="0.62 -0.72 0.58" xyaxes="0.76 0.65 0 -0.34 0.40 0.85"/>
"""
    if "autoresearch_target" not in scene:
        additions += """
    <body name="autoresearch_target" pos="0.2 0 0.16">
      <geom name="autoresearch_target_geom" type="sphere" size="0.025" rgba="0.1 0.85 0.25 1" contype="0" conaffinity="0"/>
    </body>
"""
    if "autoresearch_object" not in scene:
        additions += """
    <body name="autoresearch_object" pos="0.12 0 0.06">
      <geom name="autoresearch_object_geom" type="box" size="0.022 0.022 0.022" rgba="0.95 0.32 0.12 1" contype="0" conaffinity="0"/>
    </body>
"""
    if additions:
        scene = scene.replace("  </worldbody>", additions + "  </worldbody>")
    GENERATED_SCENE.write_text(scene)
