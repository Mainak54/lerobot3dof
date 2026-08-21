# SO-101 stage 1 — 3-DOF variant

```
TRAINED : shoulder_pan   shoulder_lift   wrist_flex
FIXED   : elbow_flex     wrist_roll      gripper
```

This is the 6-DOF stage-1 project with **only** the changes the 3-servo and
OpenCV switch forces. Everything else — dwell episodes, the reward weights,
the success tolerances, the randomisation, the snapshot layout — is carried
over unchanged.

## The complete diff against `configs/stage1.yaml`

**Forced by the 3-servo change**

| key | before | after |
|---|---|---|
| `env.home_qpos` | 6 entries | 3 — indices 0, 1, 3 of the original |
| `env.max_delta` | 6 entries | 3 — same values as pan/lift/wrist_flex |
| `env.fixed_angles` | — | new; written by `tools/setup_fixed_angles.py` |
| `reward.weights.w_grip` | 1.0 | removed — constant on a frozen joint |
| `success.grip_min/max` | 0.40 / 0.65 | removed — always true or always false |
| action space | 6 | 3 |
| observation | 23-dim, detection at `[18:22]` | 14-dim, detection at `[9:13]` |

**Forced by the OpenCV-only change**

| key | before | after |
|---|---|---|
| `detector.backend` | `opencv` (of three) | removed; opencv is the only path |
| `detector.weights/conf/imgsz/device/half` | YOLO settings | removed |
| `detector.pixel_noise/dropout_prob/latency_steps` | analytic-only | removed |

**Everything else is identical.** `episode_seconds: 30.0`, `episode_mode:
dwell`, `action_filter: 0.5`, `home_noise: 0.08`, the full spawn block
including `min_clearance: 0.12` and `hidden_fraction: 0.4`, `v_min: 140`,
`s_max: 90`, `min_area_px: 25`, `randomize_thresholds: true`, 640×480, the
whole `randomization:` block, every reward weight, every success tolerance,
and the whole `train:` block down to `n_steps: 256` and `clip_obs: 10.0`.

`env.continuing` is the one key I did not carry over. It was `false`, so it
changed nothing, and implementing an unused respawn path would be adding code
you did not ask for. Say the word and it goes back in.

## File structure

```
so101_3dof/
├── tools/
│   ├── setup_fixed_angles.py   choose + validate the frozen angles  ← RUN FIRST
│   └── tune_hsv.py             tune v_min / s_max against sim and real frames
├── vision/
│   ├── __init__.py
│   └── cv_detector.py          the opencv backend, extracted
├── mujoco/                     ← copy from the 6-DOF project, unchanged
├── configs/
│   ├── stage1_3dof.yaml
│   └── fixed_angles.yaml       written by setup_fixed_angles.py, not by hand
├── reward_3dof.py
├── env_3dof.py
├── sanity_check.py             scripted IK feasibility probe      ← RUN SECOND
├── train_3dof.py
├── evaluate.py
└── visualize_policy.py
```

Snapshots land in `policy/snapshots/<run_name>/` and logs in
`train_logs/<run_name>/`, exactly as before: `best.pt` + `vecnorm_best.pkl`,
`final.pt` + `vecnorm_final.pkl`, `ckpt_<steps>.pt` + matching pkl, plus
`config.yaml` and `fixed_angles.yaml`.

## Pipeline

```bash
# 1. Choose the frozen angles — the elbow decides approach pitch at every radius
python tools/setup_fixed_angles.py sweep --scene mujoco/train_scene.xml
python tools/setup_fixed_angles.py check --elbow <best> --roll 0.0
python tools/setup_fixed_angles.py tune  --elbow <best> --roll 0.0   # 's' saves

# 2. Copy the reachable radii from `check` into spawn.radius_min / radius_max

# 3. Confirm the task is feasible before spending GPU hours
python sanity_check.py --episodes 30 --all-visible --no-dr

# 4. Train
MUJOCO_GL=egl python train_3dof.py --config configs/stage1_3dof.yaml

# 5. Evaluate, including the vision ablation
python evaluate.py --snapshot policy/snapshots/stage1_approach_3dof/best.pt --episodes 100
python evaluate.py --snapshot ... --episodes 100 --blind

# 6. Watch
python visualize_policy.py --snapshot ... --speed 0.5
python visualize_policy.py --scripted
```

`setup_fixed_angles.py` is the one genuinely new step. `shoulder_lift`,
`elbow_flex` and `wrist_flex` are parallel pitch axes, so freezing the elbow
leaves two joints in the vertical plane — exactly enough to hit a target
(r, z), which means approach pitch is no longer a free variable:

```
pitch = shoulder_lift + elbow_fixed + wrist_flex
```

The elbow angle you freeze decides how close to `align_min: 0.94` you can get
at each radius. No reward coefficient fixes a bad choice. If `approach_angle`
dominates the failure breakdown, that is the knob — not `align_min`.

The scripted IK in `sanity_check.py` is position-only for the same reason:
three joints against three position constraints is square, so orientation is
not commanded and the script reports the resulting angle instead of solving
for it.

## Changing the control rate

One knob:

```yaml
env:
  control_hz: 30      # 5, 30, 60 ... must divide physics_hz evenly
```

`rate.py` derives everything else from it — `frame_skip`, `max_delta`,
`action_filter`, `action_delay_min/max`, `hold_steps`, `w_hold`, `step_cost`,
`gamma`, `total_timesteps`, `checkpoint_every`. Every other number in the
config is authored at `reference_hz: 30` and rescaled from there. The env and
`train_3dof.py` print the resolved table on startup.

The rule: per-step quantities scale, per-event and physical ones don't.
`success_bonus`, `failure_penalty`, `episode_seconds` and every tolerance are
untouched. `gamma` is **exponentiated**, not scaled — it's a per-step decay, so
0.99 at 30 Hz means a 3.3 s horizon but 20 s at 5 Hz; `0.99**6 = 0.941` keeps
the horizon fixed, which is the trap this file exists to close.

Verified invariant across 5 / 30 / 60 Hz:

| | 30 Hz | 5 Hz | 60 Hz |
|---|---|---|---|
| commanded joint speed | 1.5 rad/s | 1.5 | 1.5 |
| gamma horizon | 3.33 s | 3.42 | 3.32 |
| hold pay per second | 15.0 | 15.0 | 15.0 |
| full-hold return | 450 | 450 | 450 |
| training episodes | 22,222 | 22,222 | 22,222 |

`hold_steps` uses ceiling division so it never rounds to zero — at 5 Hz that
makes the hold 0.6 s rather than 0.5 s, the only quantity that can't land
exactly.

A rate that doesn't divide `physics_hz` is rejected with the list of valid
rates rather than silently rounding `frame_skip`. `physics_hz` is also checked
against the scene's actual timestep at construction.

## Reading a run

| symptom | means |
|---|---|
| `episode_success` at 1.0, returns still climbing | Expected in dwell mode. Read `z/dwell_frac`. |
| Success plateaus at a stable fraction | Spawn annulus exceeds the frozen elbow's reach. Re-run `setup_fixed_angles.py check`. |
| `approach_angle` dominates failures | Frozen elbow is wrong. Do not loosen `align_min`. |
| `z/collision_rate` above ~5% late in training | Raise `w_collision`. Too high from the start and the arm holds high and never approaches. |
| Success survives `--blind` | Information leak. `hidden_fraction` resets sample pan independently of cube azimuth for exactly this reason — verify it. |
| Policy loads but behaves randomly | Missing `vecnorm_*.pkl`. `normalize_obs` is true; the stats are part of the policy. |