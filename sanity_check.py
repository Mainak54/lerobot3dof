#!/usr/bin/env python3
"""
Validate the 3-DOF environment BEFORE spending GPU time on PPO.

    python sanity_check.py --config configs/stage1_3dof.yaml --episodes 30

Drives the env with a scripted Jacobian-IK controller instead of a policy. If
a hand-written controller cannot reach the pre-grasp pose, the task is either
unreachable or the success predicate is unsatisfiable, and no amount of PPO
tuning fixes that. Run this every time you change the reward, the scene, or
the fixed angles.

THE SOLVER ADAPTS TO HOW MANY JOINTS YOU TRAIN
-----------------------------------------------
It reads env.trained_joints, so it follows the config rather than assuming a
particular arm.

3 TRAINED JOINTS (elbow frozen): shoulder_pan only rotates the vertical
plane, so position alone is three constraints against three DOF — a SQUARE
system. So:

  * The IK is now position-only and generally solves to machine precision.
    A large residual no longer means "the solver struggled"; it means the
    point is genuinely outside the reachable set.
  * Approach tilt is NOT commanded any more. It is whatever the geometry
    hands you at that (r, z). The script therefore REPORTS tilt instead of
    solving for it, and if tilt is the dominant failure the fix is a
    different frozen elbow angle (setup_fixed_angles.py), not a gain tweak.

4 TRAINED JOINTS (elbow freed): three parallel pitch axes make the planar arm
redundant, so position AND a vertical approach can be satisfied together. The
solver adds the orientation rows, tilt becomes a request rather than an
outcome, and `approach_angle` failures should mostly disappear.

Either way the point of running this before training is the same: it separates
"my controller is bad" from "this pose does not exist".
"""

from __future__ import annotations

import argparse
import os

# Headless-safe GL defaults. Without these MuJoCo tries GLFW, finds no
# DISPLAY, and dies with "gladLoadGL error" — even though this script only
# ever renders offscreen. PyOpenGL reads PYOPENGL_PLATFORM, not MUJOCO_GL,
# so both must be set.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import math
import time
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np
import yaml

from env_3dof import SO101Approach3DOF

ROOT = Path(__file__).resolve().parent
DOWN = np.array([0.0, 0.0, -1.0])


# ----------------------------------------------------------------------
def solve_ik(env, target_pos, iters=300, lam=1e-4):
    """
    Solve the three trained joints for a pad position at target_pos.

    Runs on a scratch MjData so the answer is "is this pose reachable",
    independent of whether a per-step controller can track it.

    Returns (q_target[n], position_residual_m, resulting_tilt_deg).
    """
    model = env.model
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = env.data.qpos

    # Fixed joints must be pinned in the scratch data too, or the solver
    # silently exploits an elbow the real policy cannot move.
    for n, v in env.fixed_vals.items():
        scratch.qpos[env.qadr[n]] = v

    q = env.data.qpos[env.t_qadr].copy()

    # NOTE THE SIGN. The shoulder body carries a 180 deg flip, so the
    # shoulder_pan joint axis (0,0,1) in its own frame points along world -Z.
    # Positive pan therefore swings the arm to NEGATIVE world azimuth. Seeding
    # with the wrong sign sends the solver 180 deg the wrong way and it never
    # recovers, because the Jacobian is locally happy there.
    q[0] = float(np.clip(-math.atan2(target_pos[1], target_pos[0]),
                         env.t_lo[0], env.t_hi[0]))

    dof = env.t_dadr
    n = len(dof)
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    pos_err = np.zeros(3)

    # With 3 trained joints the planar arm is exactly determined by position
    # alone, so orientation cannot be requested — it is whatever the geometry
    # gives. With 4 (elbow_flex freed) the arm is redundant in the plane, so
    # we ALSO drive the approach axis toward vertical; otherwise the damped
    # solve picks an arbitrary member of the null space and the resulting tilt
    # is luck rather than a property of the pose.
    solve_orientation = n >= 4

    for _ in range(iters):
        scratch.qpos[env.t_qadr] = q
        mujoco.mj_forward(model, scratch)

        R = scratch.xmat[env.gripper_body].reshape(3, 3)
        pad = scratch.xpos[env.gripper_body] + R @ np.array([-0.0069, 0.0, -0.0880])
        axis = R @ np.array([0.0, 0.0, -1.0])

        pos_err = target_pos - pad
        # Rotation error as the axis-angle that rotates `axis` onto straight
        # down; its z component is always ~0, so only x and y are usable rows.
        rot_err = np.cross(axis, DOWN)
        done = np.linalg.norm(pos_err) < 1e-5 and (
            not solve_orientation or np.linalg.norm(rot_err) < 1e-3)
        if done:
            break

        mujoco.mj_jac(model, scratch, jacp, jacr, pad, env.gripper_body)
        if solve_orientation:
            J = np.vstack([jacp[:, dof], jacr[:2, dof]])    # 5 x n
            err = np.concatenate([pos_err, rot_err[:2]])
        else:
            J = jacp[:, dof]                                # 3 x n
            err = pos_err
        # Damped least squares in the J^T J form: correct whether the system
        # is square, over- or under-determined, unlike the J J^T form which
        # blows up when rows are rank-deficient.
        dq = np.linalg.solve(J.T @ J + lam * np.eye(n), J.T @ err)
        q = np.clip(q + np.clip(dq, -0.1, 0.1), env.t_lo, env.t_hi)

    scratch.qpos[env.t_qadr] = q
    mujoco.mj_forward(model, scratch)
    approach = scratch.xmat[env.gripper_body].reshape(3, 3) @ np.array([0.0, 0.0, -1.0])
    tilt = math.degrees(math.acos(float(np.clip(-approach[2], -1.0, 1.0))))
    return q, float(np.linalg.norm(pos_err)), tilt


def track_step(env, q_target, gain=0.5):
    """
    Joint-space P controller onto a solved IK configuration.

    THE GAIN MATTERS. At gain 1.0 this is deadbeat — it commands exactly the
    remaining error every step — and that is a guaranteed limit cycle against
    any actuator lag: the arm arrives, overshoots, and never satisfies the
    speed criterion. At 0.5 it settles. This is a property of the controller,
    not the physics; a trained policy learns its own damping.
    """
    q = env.data.qpos[env.t_qadr]
    return np.clip(gain * (q_target - q) / env.max_delta, -1.0, 1.0)


def why_failed(env, det, pad, pad_vel):
    """Which success condition is violated right now, in task order."""
    c = env.reward_fn.c
    cube = env.cube_pos
    if det[3] < 0.5:
        return "not_detected"
    if abs(det[0]) > c.center_tol or abs(det[1]) > c.center_tol:
        return "not_centred"
    if np.linalg.norm(pad[:2] - cube[:2]) > c.xy_tol:
        return "xy_distance"
    dz = pad[2] - cube[2]
    if not (c.z_min <= dz <= c.z_max):
        return "standoff_height"
    if np.dot(env.approach_axis, DOWN) < c.align_min:
        return "approach_angle"
    if np.linalg.norm(pad_vel) > c.speed_max:
        return "too_fast"
    return "ok"


# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; episode N uses seed+N, so the same cube "
                         "sequence replays every run. Pass -1 for a different "
                         "random sequence each time.")
    ap.add_argument("--gain", type=float, default=0.5)
    ap.add_argument("--no-dr", action="store_true",
                    help="disable domain randomisation for a cleaner signal")
    ap.add_argument("--all-visible", action="store_true",
                    help="spawn every cube in view. The scripted controller "
                         "has no search behaviour, so off-camera starts fail "
                         "on perception, not on reachability.")
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    from rate import resolve as resolve_rate
    cfg = resolve_rate(yaml.safe_load(cfg_path.read_text()),
                       verbose=True)
    if args.no_dr:
        cfg["randomization"]["enabled"] = False
    if args.all_visible:
        cfg["spawn"]["hidden_fraction"] = 0.0

    env = SO101Approach3DOF(cfg, seed=args.seed)
    print(f"control rate       {env.control_hz:.2f} Hz")
    print(f"episode length     {env.max_steps} steps "
          f"({cfg['env']['episode_seconds']} s)")
    print(f"episode mode       {env.episode_mode}")
    print(f"observation dim    {env.observation_space.shape[0]}")
    print(f"action dim         {env.action_space.shape[0]}")
    print("fixed joints       " + "  ".join(
        f"{k}={v:+.3f}" for k, v in env.fixed_vals.items()))
    print()

    successes = 0
    returns, lengths, residuals, tilts = [], [], [], []
    dwells, ttfs = [], []
    unreachable = 0
    visible_at_start = 0
    fail = Counter()
    det_frames = tot_frames = 0
    obs_min = obs_max = None
    t0 = time.time()

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=None if args.seed < 0 else args.seed + ep)
        visible_at_start += int(env.last_detection[3] > 0.5)

        # Solve once per episode: the cube does not move during an episode.
        target = env.cube_pos.copy()
        target[2] += 0.5 * (env.reward_fn.c.z_min + env.reward_fn.c.z_max)
        q_target, res, tilt = solve_ik(env, target)
        residuals.append(res)
        tilts.append(tilt)
        if res > 0.005:
            unreachable += 1

        prev_pad = env.pinch_pos.copy()
        done, ep_ret, step = False, 0.0, 0
        reason = "timeout"

        while not done:
            obs, r, term, trunc, info = env.step(track_step(env, q_target,
                                                            args.gain))
            ep_ret += r
            step += 1
            obs_min = obs if obs_min is None else np.minimum(obs_min, obs)
            obs_max = obs if obs_max is None else np.maximum(obs_max, obs)
            tot_frames += 1
            det_frames += int(info["detected"])

            pad = env.pinch_pos
            reason = why_failed(env, env.last_detection, pad,
                                (pad - prev_pad) * env.control_hz)
            prev_pad = pad.copy()
            done = term or trunc

        if info.get("is_success"):
            successes += 1
        else:
            fail[reason] += 1
        dwells.append(float(info.get("dwell_frac", 0.0)))
        ttf = float(info.get("time_to_first", -1.0))
        if ttf >= 0:
            ttfs.append(ttf)
        returns.append(ep_ret)
        lengths.append(step)

    dt = time.time() - t0
    n = args.episodes
    print(f"scripted IK reached   {successes}/{n} ({100*successes/n:.0f}%)")
    if dwells:
        print(f"dwell fraction        {100*np.mean(dwells):.0f}% of the episode "
              f"held in the pre-grasp pose")
    if ttfs:
        print(f"time to first hold    median {np.median(ttfs):.2f}s")
    print(f"mean return           {np.mean(returns):+.2f}")
    print(f"mean episode length   {np.mean(lengths):.0f} steps")
    print(f"detection rate        {100*det_frames/max(tot_frames,1):.0f}%")
    print(f"visible at step 0     {100*visible_at_start/n:.0f}% "
          f"(the rest require an actual search)")
    print(f"throughput            {tot_frames/dt:.0f} env steps/s (single env)")
    print(f"IK residual           median {np.median(residuals)*1000:.3f} mm  "
          f"max {np.max(residuals)*1000:.1f} mm")
    print(f"  unreachable targets {unreachable}/{n}")
    print(f"resulting tilt        median {np.median(tilts):.1f} deg  "
          f"max {np.max(tilts):.1f} deg  "
          f"(tolerance {math.degrees(math.acos(env.reward_fn.c.align_min)):.0f})")

    if fail:
        print("\nfailure breakdown (condition violated at episode end):")
        for reason, count in fail.most_common():
            print(f"   {reason:<20} {count}")

    print(f"\nobservation range     min {obs_min.min():+.2f}  "
          f"max {obs_max.max():+.2f}")
    if obs_max.max() > 25 or obs_min.min() < -25:
        print("   WARNING: large observation magnitudes; check scaling")

    # --- the diagnosis, which is the point of this script ----------------
    rate = successes / n
    print()
    if unreachable > 0.1 * n:
        print(">>> The IK cannot even reach the requested pose in "
              f"{unreachable}/{n} episodes.")
        print(">>> Your spawn annulus extends past what the frozen elbow can "
              "reach.")
        print(">>> Re-run: python tools/setup_fixed_angles.py check --elbow "
              "<yours>")
        print(">>> and copy the printed radii into spawn.radius_min/max.")
    elif fail.get("approach_angle", 0) > 0.15 * n:
        print(">>> Tilt is the dominant failure, and with a frozen elbow tilt "
              "is NOT")
        print(">>> controllable — it is fixed by the geometry at each radius. "
              "Do not")
        print(">>> loosen tilt_tol_deg to hide this. Pick a better elbow angle "
              "with")
        print(">>> tools/setup_fixed_angles.py sweep, or train elbow_flex too.")
    elif rate < 0.5:
        print(">>> Scripted control cannot reliably reach the pre-grasp pose.")
        print(">>> Loosen the dominant failure condition above before starting "
              "PPO.")
        print(">>> Do not train against an unsatisfiable goal.")
    else:
        print(f">>> Task is feasible ({100*rate:.0f}% scripted). Safe to train.")


if __name__ == "__main__":
    main()