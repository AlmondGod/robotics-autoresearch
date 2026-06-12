# Cosmos RoboCasa Finetune Plan

Goal:
- Finetune a Cosmos video world model on our RoboCasa data.
- Measure both visual prediction quality and correlation with real RoboCasa eval success.

Recommended model:
- First pass: `Cosmos-Predict2.5-2B-Video2World`.
- Avoid 14B until the 2B pipeline is working.
- For correlation/value prediction, prefer the Cosmos Policy recipe over plain Video2World because it predicts future images, proprio, actions, and value.

Relevant official docs:
- Cosmos Predict2.5 action-conditioned Video2World post-training:
  `https://docs.nvidia.com/cosmos/latest/predict2.5/post-training/video2world_action-conditioned.html`
- Cosmos Policy RoboCasa recipe:
  `https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/predict2/cosmos_policy/post_training.html`

Local prep implemented:
- `data/export_robocasa_cosmos_action.py`
  - Exports our LeRobot RoboCasa demos into a Bridge-like Cosmos folder:
    - `videos/*.mp4`
    - `annotations/*.json`
    - `metas/*.txt`
    - `manifest.json`
  - Preserves native RoboCasa `action` as 12D.
  - Writes `state_full`, `state[:6]`, gripper state, reward/done, and prompt.
- `eval/eval_cosmos_video_quality.py`
  - Scores generated videos for visual quality.
  - If reference videos exist, computes PSNR.
  - Correlates generated-video scores with real RoboCasa sim success archive.

Export command:

```bash
PYTHONPATH=.:third_party/robocasa:third_party/robosuite \
/opt/anaconda3/envs/robocasa/bin/python data/export_robocasa_cosmos_action.py \
  --manifest data/robocasa5/manifest.json \
  --task-alias OpenDrawer \
  --robocasa-task-index 0 \
  --max-demos-per-task 80 \
  --out-dir data/cosmos_robocasa_action/opendrawer_task0 \
  --layout side_by_side \
  --resize 320 \
  --fps 20
```

Cosmos setup on GPU host:

```bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git
cd cosmos-predict2.5
huggingface-cli login
python -m scripts.download_checkpoints --model_types video2world --model_sizes 2B
```

Official action-conditioned post-training command shape:

```bash
torchrun --nproc_per_node=1 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py -- \
  experiment=ac_reason_embeddings_rectified_flow_2b_256_320 \
  ~dataloader_train.dataloaders
```

Needed Cosmos-side edit:
- The official action-conditioned example uses Bridge-style action annotations.
- Bridge action is 7D.
- RoboCasa action here is 12D.
- Cosmos config / conditioner must either:
  - set `action_dim=12`, or
  - map RoboCasa actions to a 7D end-effector action representation.

Evaluation after Cosmos generations:

```bash
PYTHONPATH=. /opt/anaconda3/envs/robocasa/bin/python eval/eval_cosmos_video_quality.py \
  --generations runs/robocasa/cosmos_generations/opendrawer_task0 \
  --sim-archive runs/robocasa/world_evaluator/trace_eval_frontier/archive_trace_frontier.jsonl \
  --out runs/robocasa/cosmos_eval/opendrawer_task0/candidate_scores.jsonl \
  --metrics-out runs/robocasa/cosmos_eval/opendrawer_task0/correlation_metrics.json \
  --plot runs/robocasa/cosmos_eval/opendrawer_task0/correlation.svg
```

Current blocker:
- The previous GPU host is not reachable:
  - `ssh root@216.81.245.138 -p 15521 -i ~/.ssh/id_ed25519`
  - error: connection refused
- Local machine is not appropriate for Cosmos 2B post-training.

Expected resource shape:
- Cosmos 2B is roughly 2B parameters.
- Inference/post-training needs a large CUDA GPU; NVIDIA docs list tens of GB of VRAM for 2B Video2World.
- Use H100/H200/A100 80GB class hardware for the clean path.

First acceptance criteria:
- Visual: generated videos are nonblank, temporally coherent, and qualitatively resemble RoboCasa camera views.
- Offline visual metrics: PSNR if paired references are available, plus motion/sharpness/nonblank scores.
- Correlation: generated-video/value score should rank the known trace candidates similarly to real RoboCasa success.
- Beat the tiny evaluator baseline only if correlation is positive on held-out candidate policies.
