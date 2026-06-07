# nano-robot-worlds

Offline-first robotics autoresearch on LIBERO-Object-5.

The active v0 question is:

```text
Can an AI researcher improve a tiny robot learning stack using cheap offline
losses, while final selection is grounded by closed-loop robot task success?
```

RL is not part of the active v0 stack. The current stack is:

- video-only LIBERO demonstrations for image/token/world-model training
- 5-10 paired action demonstrations per task for BC/inverse dynamics
- tiny image tokenizer
- nanoGPT-style video-token world model
- inverse dynamics model
- tiny BC policy
- offline metrics plus fixed LIBERO closed-loop success evaluation

## LIBERO Setup

Install local dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[libero]"
```

Fetch LIBERO and download LIBERO-Object:

```bash
python data/download_libero.py --dataset libero_object --use-huggingface
```

Create the seed benchmark:

```bash
python data/make_libero5.py --libero-root third_party/LIBERO --paired-demos-per-task 10
python data/split_video_and_paired.py --manifest data/libero_object5/manifest.json
```

This produces:

```text
data/libero_object5/manifest.json
data/libero_object5/libero_object5_video.npz
data/libero_object5/libero_object5_paired.npz
```

## V0 Training

BC baseline:

```bash
python research/run_experiment.py --config configs/libero_v0_bc.json --skip-success-eval
```

Tokenizer + world model + inverse dynamics + BC:

```bash
python research/run_experiment.py --config configs/libero_v0_world_inverse.json --skip-success-eval
```

Offline metric table:

```bash
python eval/eval_offline.py --runs-root runs/libero
```

Closed-loop LIBERO success evaluation lives in `eval/eval_libero_success.py`.
That file is the fixed eval surface for v0 and should not be modified by
autoresearch experiments once the LIBERO observation/action adapter is wired.

## V0 Milestone Table

```text
method              video_loss   action_mse   bc_loss   success_rate
BC baseline         -            -            x         x%
BC + tokenizer      x            -            x         x%
BC + world loss     x            -            x         x%
BC + inverse loss   x            x            x         x%
```

Autoresearch experiments are JSON configs under `configs/`. The agent may vary
model size, tokenizer type, context length, data mix, loss weights, learning
rate, augmentation, demo count, and which model is evaluated. It may not touch
fixed eval code.

## Legacy Sim Harness

This repository is a small, constrained robotics research loop inspired by
Karpathy's `autoresearch`, adapted for simulated sim-to-real gaps. The agent's
editable surface is intentionally narrow: future autoresearch agents should edit
`research.py` while the benchmark, evaluator, judge, and task definitions remain
fixed.

## Phase 1 Scope

- Lightweight continuous-control robot benchmark.
- Separate training and evaluation worlds.
- Evaluation world has shifted dynamics, latency, noise, and object parameters.
- Fixed scoring with safety penalties.
- JSON run artifacts plus a repository-level `research_log.jsonl`.
- Plotting utility for commit/change/score progress over time.

Phase 1 intentionally excludes real robots, internet data, behavior cloning
demos, and external datasets. The ARX L5 backend adds a first vision-policy
training surface inside simulation.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python run_experiment.py --task reach --budget-seconds 5 --seeds 0 1
python run_experiment.py --task push --budget-seconds 5 --seeds 0 1 --change-note "baseline push smoke"
python plot_progress.py
```

Run artifacts are written to `runs/<run-id>/`. The long-lived progress ledger is
`runs/research_log.jsonl`.

## MuJoCo Backend

The default backend is the lightweight toy simulator. A MuJoCo backend is also
available for real rigid-body physics, MJCF models, contact, and camera
rendering.

Install the optional dependency:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[mujoco]"
```

Run MuJoCo evaluation:

```bash
python run_experiment.py --backend mujoco --task reach --budget-seconds 60 --seeds 0 1 2
python run_experiment.py --backend mujoco --task push --budget-seconds 60 --seeds 0 1 2
```

Render a MuJoCo camera video:

```bash
python render_eval_video.py --backend mujoco --task reach --out runs/mujoco_eval_reach.mp4
```

The MJCF model lives at
`robotbench/assets/mujoco/planar_arm.xml`.

## ALOHA 14-Actuator Backend

The `aloha` backend uses the real ALOHA bimanual robot model from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie). The
model has 16 physical joints and a 14-actuator control interface: 6 arm joints
plus one gripper actuator per side.

Fetch the Apache-2.0 Menagerie assets:

```bash
python scripts/fetch_menagerie.py --model aloha
```

Run and render ALOHA:

```bash
python run_experiment.py --backend aloha --task reach --budget-seconds 60 --seeds 0 1 2
python render_eval_video.py --backend aloha --task reach --out runs/aloha_eval_reach.mp4
```

The fetched assets are stored under `third_party/mujoco_menagerie/` and are
ignored by git. Keep using the fetch script so the source and license are clear.

## ARX L5 Camera PPO Backend

The `arx_l5` backend uses the single-arm ARX L5 model from MuJoCo Menagerie.
Observations are one 24x24 grayscale frame from the robot wrist camera plus
compact proprio/task deltas. PPO uses a small CNN encoder. Actions are 7
normalized position-control deltas: 6 arm joints plus the gripper actuator.

Install the optional dependencies and fetch the model:

```bash
pip install -e ".[mujoco,ppo]"
python scripts/fetch_menagerie.py --model arx_l5
```

Run a short PPO experiment and render the saved policy:

```bash
python run_experiment.py --backend arx_l5 --task reach --budget-seconds 60 --seeds 0
python render_eval_video.py --backend arx_l5 --task reach --policy-path runs/<run-id>/policy_seed0.npz --out runs/arx_l5_eval_reach.mp4
```

`pick_place` adds a kinematic object and placement target. Metrics include
stage rates for reaching the object, grasping, lifting, and placing. Training
can use a staged curriculum while eval remains full pick-and-place:

```bash
python run_experiment.py --backend arx_l5 --task pick_place --curriculum-stage reach --budget-seconds 60 --seeds 0
python run_experiment.py --backend arx_l5 --task pick_place --curriculum-stage full --budget-seconds 300 --seeds 0
```

Short CPU runs are mostly smoke tests because MuJoCo camera rendering is the
bottleneck. Longer runs or a CNN policy are needed before treating this as a
strong pixel-control baseline.

## Mobile ALOHA Status

Actual Mobile ALOHA is not implemented yet. The public
[Mobile ALOHA repository](https://github.com/MarkFzp/mobile-aloha) is a
ROS/hardware/data-collection codebase, not a clean MuJoCo asset package that can
be dropped into this benchmark.

`mobile_aloha_mock` is only a placeholder mobile-manipulation setting built from
the Menagerie ALOHA model. It mounts the ALOHA arms on a kinematic mobile base
and adds base translation commands before the 14 ALOHA controls:

```text
action[0:3]   mobile base command
action[3:17]  ALOHA 14-actuator command
```

This is not Mobile ALOHA and is not an official Menagerie robot asset. It is a
temporary benchmark environment for testing mobile manipulation research loops
until we import or author a real Mobile ALOHA MJCF.

```bash
python scripts/fetch_menagerie.py --model aloha
python run_experiment.py --backend mobile_aloha_mock --task reach --budget-seconds 60 --seeds 0 1 2
python render_eval_video.py --backend mobile_aloha_mock --task reach --out runs/mobile_aloha_mock_eval_reach.mp4
```

## Main Commands

Run one candidate:

```bash
python run_experiment.py --backend toy --task reach --budget-seconds 60 --seeds 0 1 2
```

Compare a candidate to a baseline:

```bash
python judge.py --baseline runs/<baseline-id> --candidate runs/<candidate-id>
```

Create a progress plot. SVG output works with only the Python standard library;
PNG output uses Matplotlib when it is installed:

```bash
python plot_progress.py --log runs/research_log.jsonl --out runs/progress.svg
```

## Autoresearch Contract

The benchmark assumes future agents may edit only `research.py`. See
`program.md` for the full operating contract.
