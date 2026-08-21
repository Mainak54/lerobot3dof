#!/usr/bin/env python3
"""
Evaluate a trained 3-DOF approach policy.

    python evaluate.py --snapshot policy/snapshots/stage1_approach_3dof/best.pt
    python evaluate.py --snapshot ... --episodes 200 --deterministic
    python evaluate.py --snapshot ... --blind

WHAT IT REPORTS
    success rate with a Wilson confidence interval (a 20-episode run tells you
    much less than the point estimate suggests)
    time to success, so you can see whether the policy is fast or merely lucky
    which success criterion blocked each failed episode
    detection rate, and success conditioned on the cube ever being seen
    action smoothness and peak joint speed — your hardware-transfer predictors

--blind IS THE IMPORTANT ONE
    It forces the detector to report "not visible" on every step. If success
    collapses, the policy is genuinely vision-driven. If it stays high,
    something is leaking the cube position into the observation and you have a
    policy that will do nothing useful on hardware. Run it every time.

    This matters more in the 3-DOF version than it did before, because the
    observation now carries a last-seen memory (indices 8:11). That channel is
    exactly the sort of thing that could smuggle information if the cube spawn
    were correlated with the arm's start pose. It is not — azimuth is sampled
    independently of home — but the ablation is how you know rather than how
    you argue.

NORMALISER STATS ARE MANDATORY
    normalize_obs is true, so a policy loaded without its VecNormalize
    statistics scores like a random policy. This script refuses to run rather
    than report a misleading number.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# PyOpenGL selects its backend from PYOPENGL_PLATFORM, NOT from MUJOCO_GL.
# On a headless box with no display it otherwise resolves to None and fails
# with "'NoneType' object has no attribute 'eglQueryString'" deep inside the
# mujoco import. Setting both keeps them consistent.
os.environ.setdefault("PYOPENGL_PLATFORM",
                      os.environ.get("MUJOCO_GL", "egl"))

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent


def load_policy(PPO, path, device=None):
    """Load a checkpoint, defaulting to CPU and falling back to it.

    device="auto" grabs CUDA, which fails outright with
    cudaErrorDevicesUnavailable when a training run already holds the GPU —
    and you will usually be watching a policy WHILE the next run trains. CPU
    is also simply the right choice here: a 256x256 MLP doing single-step
    inference gains nothing from a GPU and would only compete with the
    trainer's EGL rendering for it.
    """
    dev = device or "cpu"
    try:
        return PPO.load(str(path), device=dev)
    except Exception as exc:                                  # noqa: BLE001
        if dev == "cpu":
            raise
        print(f"could not load on {dev} ({type(exc).__name__}: {exc}); "
              f"falling back to cpu")
        return PPO.load(str(path), device="cpu")


def wilson(successes, n, z=1.96):
    """Wilson score interval. Honest about small samples in a way +/-sqrt is not."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def failed_criterion(env, ever_seen):
    """
    Which success condition blocked this episode, checked in task order.

    Reads env.last_detection rather than the observation vector: the obs may
    have passed through a wrapper, and its detection entries are not
    guaranteed to be directly interpretable.

    The height check is on the STANDOFF above the cube (pinch.z - cube.z),
    which is what z_min / z_max mean and what the reward tests.
    """
    c = env.reward_fn.c
    det = env.last_detection
    if not ever_seen:
        return "never_detected"
    if det[3] < 0.5:
        return "lost_track"
    if abs(det[0]) > c.center_tol or abs(det[1]) > c.center_tol:
        return "not_centred"
    if np.linalg.norm(env.pinch_pos[:2] - env.cube_pos[:2]) > c.xy_tol:
        return "xy_distance"
    dz = env.pinch_pos[2] - env.cube_pos[2]
    if not (c.z_min <= dz <= c.z_max):
        return "standoff_height"
    if np.dot(env.approach_axis, np.array([0.0, 0.0, -1.0])) < c.align_min:
        return "approach_angle"
    return "not_settled"


def resolve_config(snap: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    cand = snap.parent / "config.yaml"
    if cand.exists():
        return cand
    raise SystemExit(
        f"config not found at {cand}. train_3dof.py copies the resolved "
        "config into the snapshot directory; pass --config explicitly if this "
        "run predates that.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--config", default=None,
                    help="defaults to config.yaml in the run directory")
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--deterministic", action="store_true",
                    help="use the mean action instead of sampling")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--all-visible", action="store_true",
                    help="spawn every cube in view; isolates servoing from search")
    ap.add_argument("--blind", action="store_true",
                    help="force the detection to (0,0,0,0) on EVERY step")
    ap.add_argument("--device", default="cpu",
                    help="torch device for inference; cpu avoids competing "
                         "with a running trainer for the GPU")
    ap.add_argument("--seed", type=int, default=10_000,
                    help="fixed by default so two checkpoints are scored on "
                         "the SAME cube layouts and the comparison is not "
                         "confounded by luck. Pass -1 to randomise.")
    ap.add_argument("--csv", default=None, help="write per-episode rows here")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from env_3dof import SO101Approach3DOF

    snap = Path(args.snapshot)
    if not snap.exists():
        raise SystemExit(f"snapshot not found: {snap}")

    cfg_path = resolve_config(snap, args.config)
    from rate import resolve as resolve_rate
    cfg = resolve_rate(yaml.safe_load(cfg_path.read_text()),
                       verbose=True)

    # Pin the run's own fixed angles. Evaluating a policy against different
    # frozen joints than it was trained with produces a number that means
    # nothing, and the failure is completely silent.
    fa = snap.parent / "fixed_angles.yaml"
    if fa.exists():
        cfg["env"]["fixed_angles"] = str(fa.resolve())

    if args.no_dr:
        cfg["randomization"]["enabled"] = False
    if args.all_visible:
        cfg["spawn"]["hidden_fraction"] = 0.0
    if args.blind:
        cfg["env"]["blind"] = True

    env = SO101Approach3DOF(cfg, seed=None if args.seed < 0 else args.seed)
    venv = DummyVecEnv([lambda: env])

    vecnorm = snap.parent / f"vecnorm_{snap.stem}.pkl"
    if not vecnorm.exists():
        raise SystemExit(
            f"normaliser stats missing: {vecnorm}\n"
            "normalize_obs is true, so evaluating without them produces "
            "numbers indistinguishable from a random policy. Find the matching "
            "vecnorm_*.pkl for this snapshot.")
    venv = VecNormalize.load(str(vecnorm), venv)
    venv.training = False
    venv.norm_reward = False

    model = load_policy(PPO, snap, args.device)

    print(f"snapshot   {snap}")
    print(f"config     {cfg_path}")
    print(f"fixed      " + "  ".join(f"{k}={v:+.3f}"
                                     for k, v in env.fixed_vals.items()))
    print(f"randomise  {cfg['randomization']['enabled']}")
    print(f"policy     {'deterministic' if args.deterministic else 'stochastic'}")
    if args.blind:
        print("BLIND MODE: detector permanently reports 'not visible'")
    print(f"episodes   {args.episodes}\n")

    successes = 0
    reasons = Counter()
    rows = []
    total_steps = 0
    t0 = time.time()

    for ep in range(args.episodes):
        obs = venv.reset()
        done = False
        ep_ret, steps = 0.0, 0
        ever_seen, det_frames, collisions, in_pose_frames = False, 0, 0, 0
        actions, joint_speed = [], []

        while not done:
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, dones, infos = venv.step(action)
            info = infos[0]
            done = bool(dones[0])
            ep_ret += float(reward[0])
            steps += 1
            actions.append(action[0].copy())
            joint_speed.append(float(np.abs(env.data.qvel[env.t_dadr]).max()))
            collisions += int(info.get("collision", False))
            if info.get("detected"):
                ever_seen = True
                det_frames += 1
            in_pose_frames += int(info.get("in_pose", False))

        success = bool(info.get("is_success", False))
        dwell_frac = float(info.get("dwell_frac",
                                    in_pose_frames / max(steps, 1)))
        ttf = float(info.get("time_to_first", -1.0))
        successes += int(success)
        if not success:
            reasons[failed_criterion(env, ever_seen)] += 1

        actions = np.array(actions)
        smooth = (float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))
                  if len(actions) > 1 else 0.0)
        total_steps += steps
        rows.append(dict(
            episode=ep, success=int(success), steps=steps,
            seconds=round(steps / env.control_hz, 2), ret=round(ep_ret, 2),
            ever_detected=int(ever_seen),
            detect_frac=round(det_frames / max(steps, 1), 3),
            collision_frac=round(collisions / max(steps, 1), 3),
            peak_joint_dps=round(float(np.degrees(max(joint_speed))), 1),
            dwell_frac=round(dwell_frac, 3), time_to_first=round(ttf, 2),
            action_delta=round(smooth, 4)))

    dt = time.time() - t0
    n = args.episodes
    lo, hi = wilson(successes, n)
    seen_rows = [r for r in rows if r["ever_detected"]]
    seen_success = sum(r["success"] for r in seen_rows)

    print(f"reached the pose   {successes}/{n} = {100*successes/n:.1f}%  "
          f"(95% CI {100*lo:.1f} - {100*hi:.1f}%)")
    if env.episode_mode == "dwell":
        dw = [r["dwell_frac"] for r in rows]
        print(f"DWELL FRACTION     mean {100*np.mean(dw):.1f}%  "
              f"median {100*np.median(dw):.1f}%   <- the headline number")
        print("                   (fraction of the episode spent holding the "
              "pose; the binary\n"
              "                    rate above saturates and stops being "
              "informative)")
    ttfs = [r["time_to_first"] for r in rows if r["time_to_first"] >= 0]
    if ttfs:
        print(f"time to first hold median {np.median(ttfs):.2f}s  "
              f"p90 {np.percentile(ttfs, 90):.2f}s")
    print(f"cube ever detected {len(seen_rows)}/{n} = {100*len(seen_rows)/n:.1f}%")
    if seen_rows:
        print(f"  success | seen   {seen_success}/{len(seen_rows)} = "
              f"{100*seen_success/len(seen_rows):.1f}%   "
              "(gap vs overall = cost of failed search)")
    print(f"detected frames    {100*np.mean([r['detect_frac'] for r in rows]):.1f}%")
    print(f"mean return        {np.mean([r['ret'] for r in rows]):+.2f}")
    print(f"action smoothness  {np.mean([r['action_delta'] for r in rows]):.4f} "
          "mean |a_t - a_t-1|  (high = servo chatter on hardware)")
    print(f"collision frames   "
          f"{100*np.mean([r['collision_frac'] for r in rows]):.1f}%  "
          "(arm touching the floor)")
    peaks = [r["peak_joint_dps"] for r in rows]
    ceiling = float(np.degrees(env.max_delta.max() * env.control_hz))
    print(f"peak joint speed   median {np.median(peaks):.0f} deg/s  "
          f"max {np.max(peaks):.0f} deg/s  (commanded ceiling {ceiling:.0f})")
    print(f"throughput         {total_steps/dt:.0f} env steps/s")

    if reasons:
        print("\nfailure breakdown:")
        for reason, count in reasons.most_common():
            print(f"   {reason:<20} {count:4d}  ({100*count/n:.1f}%)")

    if args.blind:
        print()
        if successes / n > 0.25:
            print(">>> Success survived blinding. The policy is NOT using "
                  "vision.")
            print(">>> Something leaks the cube position into the observation. "
                  "Check")
            print(">>> that spawn azimuth is independent of home_qpos, and that "
                  "the")
            print(">>> last-seen channels (obs[8:11]) are zero throughout a "
                  "blind run.")
        else:
            print(">>> Success collapsed under blinding, as it should. The "
                  "policy is genuinely vision-driven.")

    if args.csv:
        path = Path(args.csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nper-episode rows -> {path}")

    venv.close()


if __name__ == "__main__":
    main()