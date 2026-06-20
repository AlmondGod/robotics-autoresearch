# AutoroboBench v0

AutoroboBench scores robotics research loops under fixed data, compute, and
evaluation contracts. This repository currently keeps only the executable
RoboCasa profile so it is easier to migrate the task/data layer later.

## Current Tracks

`configs/autorobobench_v0.json` defines four tracks totaling 120 points.

| ID | Track | Points | Primary metric |
| --- | --- | ---: | --- |
| `robocasa_bc5` | RoboCasa BC-5 | 30 | hidden held-out success |
| `robocasa_long_horizon` | Long-Horizon RoboCasa | 40 | full success plus subgoal progress |
| `video_policy_transfer` | Video Data to Policy Transfer | 30 | scarce-action success with video-only data |
| `robocasa_microwave_peak` | Microwave Peak Reliability | 20 | single-task near-perfect success |

The retained public assets are:

- `data/robocasa5/manifest.json`
- `data/autorobobench/robocasa_bc5_splits.json`
- `data/autorobobench/robocasa_long_horizon_splits.json`
- `data/autorobobench/video_policy_transfer_splits.json`
- `data/autorobobench/video_policy_transfer_video_pool.json`
- `data/autorobobench/robocasa_microwave_peak_manifest.json`
- `data/autorobobench/robocasa_microwave_peak_splits.json`
- `data/autorobobench/robocasa_microwave_peak_video_pool.json`

## Scoring

Each track result is a JSON object with metrics in `[0, 1]`. The score uses
the configured weighted component average, with normalized primary progress
computed against starter and reference metrics.

Run a local score smoke test:

```bash
python -m autorobobench.cli score \
  --config configs/autorobobench_v0.json \
  --results examples/autorobobench_v0_results.json
```

Create a hash manifest for immutable files:

```bash
python -m autorobobench.cli hash-manifest \
  --config configs/autorobobench_v0.json \
  --out runs/autorobobench/v0_immutable_hashes.json
```

## Task Packages

Each task package follows the same contract:

- `setup.py`: verifies public metadata and local datasets
- `train.py`: editable training entrypoint
- `inference.py`: policy loading/action interface used by eval
- `eval.py`: immutable evaluator wrapper
- `task.json`: task metadata

## RoboCasa BC-5

The BC-5 track uses five RoboCasa tasks with frozen public train/validation/eval
episode IDs. A small baseline can be trained with:

```bash
python tasks/robocasa_bc5/train.py \
  --out-dir runs/autorobobench/robocasa_bc5/baseline \
  --train-episodes-per-task 4 \
  --val-episodes-per-task 2 \
  --steps 200
```

Evaluate with:

```bash
python tasks/robocasa_bc5/eval.py \
  --policy runs/autorobobench/robocasa_bc5/baseline/policy_best.pt \
  --out runs/autorobobench/robocasa_bc5/baseline/eval_success.json \
  --eval-episodes-per-task 1
```

## Long-Horizon RoboCasa

The long-horizon track reuses the BC-5 policy interface but evaluates the three
multi-stage RoboCasa seed tasks with longer rollouts and shorter action commits.

```bash
python tasks/robocasa_long_horizon/setup.py --verify
python tasks/robocasa_long_horizon/train.py --steps 200
```

## Video Policy Transfer

The video-transfer track limits paired-action training to two demos per task and
exposes a larger RGB-only video pool without actions or proprio.

```bash
python tasks/video_policy_transfer/setup.py --verify
python tasks/video_policy_transfer/train.py --max-train-seconds 300
```

## Microwave Peak Reliability

The microwave peak track isolates one visually clear repetitive task:
`PickPlaceCounterToMicrowave`. It uses the current BC-5 policy/training base,
80 target-task cloning demos, and an optional generic RGB-only video pool from
other RoboCasa tasks. The generic pool excludes the target task to avoid exposing
held-out microwave eval videos.

```bash
python tasks/robocasa_microwave_peak/setup.py --verify
python tasks/robocasa_microwave_peak/train.py
python tasks/robocasa_microwave_peak/eval.py \
  --policy runs/autorobobench/robocasa_microwave_peak/bc5_base/policy_best.pt \
  --out runs/autorobobench/robocasa_microwave_peak/bc5_base/eval_success.json
```
