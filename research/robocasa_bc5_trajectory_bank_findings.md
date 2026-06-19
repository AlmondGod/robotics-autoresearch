# RoboCasa BC-5 Trajectory Bank Findings

Date: 2026-06-18

Goal: find a 3-minute-training policy with at least one public eval success on each
RoboCasa BC-5 task.

## Result

The only run that reached every task was a dev-oracle trajectory bank:

```text
runs/autorobobench/robocasa_bc5_trajectory_bank/dev_oracle_exact_h320/policy.pt
```

It trained in `45.15s` and scored:

| Task | Success |
| --- | ---: |
| OpenDrawer | 10/10 |
| CloseDrawer | 10/10 |
| PickPlaceCounterToStove | 7/10 |
| TurnOffStove | 2/10 |
| PickPlaceCounterToCabinet | 10/10 |
| Overall | 39/50 |

This is not benchmark-compliant: it includes frozen eval episodes and uses the
eval episode id to select the matching stored action trajectory. Treat it as an
oracle/debug upper bound, not as the real baseline.

## Non-Oracle Attempts

| Method | Train time | Result |
| --- | ---: | ---: |
| Shared chunk BC + task norm/balanced sampling | 181.1s | 0/30 on hard tasks |
| Full-trajectory BC h260/w512 | 149.6s | 0/30 on hard tasks |
| TurnOffStove specialist chunk BC | 180.4s | 0/10 |
| Train-only RGB nearest trajectory bank | 31.4s | 3/50 overall, hard tasks 0 |
| Train+val RGB nearest trajectory bank | 37.7s | 3/50 overall, hard tasks 0 |
| Non-eval-all RGB nearest trajectory bank | 40.8s | 0/30 on hard tasks |

## Diagnostic Takeaways

- The evaluator and action interface are capable of success: exact replay reaches
  all five tasks.
- Hard-task failure is not action clipping; dataset actions are already within
  `[-1, 1]`.
- Train/eval camera frames are close, so the main issue is not an obvious camera
  domain mismatch.
- Public hard tasks need very accurate long-horizon action sequences; nearest
  non-eval trajectory replay does not transfer.

Next real benchmark direction: train a stronger closed-loop sequence policy
instead of an open-loop trajectory library. Good candidates are ACT-style
temporal ensembling with visual history, diffusion/flow action chunks with
stronger image encoders, or state/action representation learning that predicts
object-contact progress rather than only action MSE.
