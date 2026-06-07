# Autoresearch Program

You are a robotics research agent improving a tiny offline robot-learning stack
on LIBERO-Object-5.

## Objective

Improve closed-loop task success in LIBERO while using only the fixed prepared
data and the fixed evaluation harness.

Primary metric:

- `success_rate`

Tie-breakers, in order:

- lower `bc_loss`
- lower action/video losses when those models are enabled

## Editable Surface

You may edit:

- `train.py`

You may change architecture, optimizer, learning rate, batch size, temporal
context, augmentation, loss weights, data mix, and training schedule inside
`train.py`.

You may not edit:

- `prepare.py`
- `data/`
- `eval/`
- `research/judge.py`
- `loop_libero.py`
- generated datasets or run artifacts

## Fixed Data

`prepare.py` owns one-time setup and runtime constants:

- 5 LIBERO object tasks
- 250 video-only human teleop demos
- 50 paired action/proprio demos
- 4-frame history
- 4-step action chunks
- agentview and wrist RGB frames

The model should treat paired actions as scarce. Video-only data is abundant
relative to action labels and can be used for tokenizer/world-model style
auxiliary learning.

## Runtime Budget

Each candidate run has at most 5 minutes of training:

```bash
python train.py --out-dir runs/libero/candidate --max-train-seconds 300
```

Evaluation runs after training and is not part of the training budget:

```bash
python prepare.py --eval-policy runs/libero/candidate/policy.pt
```

The loop command is:

```bash
python loop_libero.py --iterations 1 --max-train-seconds 300
```

## Research Rules

Each change should have a short hypothesis. Prefer small, testable changes over
large rewrites.

Do not exploit evaluation implementation details, edit success criteria, change
task definitions, tune to a single reset state, or modify fixed data splits.

The clean thesis for v0 is:

```text
Can an AI researcher improve a tiny robot learning stack using cheap offline
losses, while final selection is grounded by closed-loop robot task success?
```
