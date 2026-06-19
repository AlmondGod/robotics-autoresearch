# External DROID Data Plan

Goal:
- Give the world-model/autoresearch stack access to a much larger robot-video corpus without leaking into final RoboCasa simulator evaluation.

Dataset:
- Primary external corpus: DROID.
- Scale: 76k robot manipulation trajectories, roughly 350 hours, across many real scenes/tasks.
- Local debug source: `lerobot/droid_100`.
- Full source: official DROID Google Cloud bucket.

Why DROID first:
- It is real robot manipulation video, not only human egocentric video.
- It includes paired robot actions, so it can train action-conditioned prediction and inverse dynamics.
- It can also be treated as video-only data by dropping actions.

Benchmark rules:
- DROID is training-only external data.
- RoboCasa-5 simulator success remains final evaluation.
- Any run using DROID should report:
  - DROID mode: `hf-debug`, `gcs-debug`, `gcs-full-rlds`, or `gcs-full-raw-no-stereo`.
  - video-only frames/episodes used.
  - paired action episodes used.
  - whether RoboCasa-only performance regressed.

Implemented entrypoints:
- `configs/external_droid_v0.json`
- `data/prepare_droid_external.py`

Recommended local smoke:
```bash
python data/prepare_droid_external.py --mode hf-debug --root data/external/droid
```

Recommended full-data machine:
```bash
python data/prepare_droid_external.py --mode gcs-full-rlds --root /data/external/droid
```

Scaled held-out world-evaluator benchmark:
- Spec: `configs/world_model_evaluator_scale.json`
- Runner: `tasks/world_model_evaluator/run_scaled_bc5.py`
- Target scale: 10 held-out test policies, 5 tasks, 10 eval episodes/task = 500 RoboCasa rollouts.

Dry run:
```bash
python tasks/world_model_evaluator/run_scaled_bc5.py --dry-run
```

