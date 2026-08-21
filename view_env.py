#!/usr/bin/env python3
"""
view_env.py — look at the training environment itself. No policy involved.

    python view_env.py                    # zero actions, arm holds home
    python view_env.py --random           # random actions, see the motion range
    python view_env.py --no-dr            # randomisation off, cleaner picture
    python view_env.py --all-visible      # never hide the cube at reset

    r + Enter   respawn the cube (new episode)
    q + Enter   quit

WHAT THIS SHOWS THAT `python -m mujoco.viewer --mjcf=...` DOES NOT
-------------------------------------------------------------------
The raw XML is not the environment. Everything that decides whether training
can succeed is applied by the env at reset, not written in the scene file:

    the fixed angles from configs/fixed_angles.yaml
    the cube spawned in the annulus, measured from the shoulder_pan axis
    hidden_fraction rotating the arm away so the cube starts off-camera
    domain randomisation of gains, damping, friction, mass, camera pose
    the detector thresholds, jittered per episode

So this is the thing to look at when the question is "why is the cube over
there" or "why can the arm never reach it".

The readout gives you, per reset: cube radius and azimuth measured from the
BASE (not the world origin), whether the detector currently sees it, and
whether that radius is inside the reachable band. That last column is the one
that explains most confusing behaviour — a cube outside the band is a cube the
frozen elbow cannot pre-grasp, and no amount of training fixes it.

MUJOCO_GL: the env always builds an offscreen renderer for the wrist camera,
so this defaults to egl. The viewer opens its own window regardless. Setting
glfw here would put two GLFW contexts in one process and segfault.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
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

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--random", action="store_true",
                    help="drive random actions instead of holding still")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--all-visible", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate; 0 = as fast as possible")
    ap.add_argument("--seed", type=int, default=-1,
                    help="-1 for a different sequence each run")
    a = ap.parse_args()

    import mujoco
    from rate import resolve as resolve_rate
    from env_3dof import SO101Approach3DOF

    cfg_path = HERE / a.config if not Path(a.config).is_absolute() else Path(a.config)
    cfg = resolve_rate(yaml.safe_load(cfg_path.read_text()), verbose=True)
    if a.no_dr:
        cfg["randomization"]["enabled"] = False
    if a.all_visible:
        cfg["spawn"]["hidden_fraction"] = 0.0
    cfg["verbose_rate"] = True

    env = SO101Approach3DOF(cfg, seed=None if a.seed < 0 else a.seed)
    sp = cfg["spawn"]

    try:
        import mujoco.viewer as mjv
    except ImportError as exc:
        sys.exit(f"mujoco.viewer unavailable: {exc}")

    state = {"reset": True, "quit": False}

    def on_key(keycode):
        ch = chr(keycode) if 0 < keycode < 0x110000 else ""
        if ch in ("r", "R"):
            state["reset"] = True
        elif ch in ("q", "Q"):
            state["quit"] = True

    def watch_stdin():
        for line in sys.stdin:
            c = line.strip().lower()
            if c in ("r", "reset"):
                state["reset"] = True
            elif c in ("q", "quit"):
                state["quit"] = True

    threading.Thread(target=watch_stdin, daemon=True).start()

    print("=" * 74)
    print(f"fixed joints  " + "  ".join(f"{k}={v:+.3f}"
                                        for k, v in env.fixed_vals.items()))
    print(f"spawn band    {sp['radius_min']:.3f} - {sp['radius_max']:.3f} m, "
          f"{sp['angle_min_deg']:+.0f} to {sp['angle_max_deg']:+.0f} deg, "
          f"hidden {sp['hidden_fraction']*100:.0f}%")
    print("r + Enter respawn    q + Enter quit")
    print("=" * 74)

    period = (1.0 / env.control_hz) / a.speed if a.speed > 0 else 0.0
    ep = 0

    try:
        with mjv.launch_passive(env.model, env.data, key_callback=on_key) as v:
            while v.is_running() and not state["quit"]:
                if state["reset"]:
                    state["reset"] = False
                    env.reset()
                    ep += 1
                    c = env.cube_pos
                    d = c[:2] - env.base_xy
                    r = float(np.linalg.norm(d))
                    th = math.degrees(math.atan2(d[1], d[0]))
                    inband = sp["radius_min"] <= r <= sp["radius_max"]
                    seen = env.last_detection[3] > 0.5
                    print(f"\nepisode {ep}: cube r={r:.3f} m  az={th:+6.1f} deg"
                          f"  {'in band' if inband else 'OUTSIDE BAND'}"
                          f"  {'visible' if seen else 'not visible at reset'}")
                    v.sync()

                # Size from the action space, never a literal: the joint
                # count follows env.trained_joints and changes with the config.
                act = (env.action_space.sample() * 0.3 if a.random
                       else np.zeros(env.action_space.shape[0], np.float32))
                _, _, term, trunc, info = env.step(act)
                if term or trunc:
                    state["reset"] = True

                if env.step_count % 10 == 0:
                    pad = env.pinch_pos
                    print(f"\r  step {env.step_count:4d}/{env.max_steps}  "
                          f"pad r={np.linalg.norm(pad[:2]-env.base_xy):.3f} "
                          f"z={pad[2]:.3f}  "
                          f"{'SEEN' if info['detected'] else '    '}"
                          f"{'  GROUND' if info.get('ground') else ''}"
                          f"{'  COLLIDE' if info.get('collision') else ''}",
                          end="", flush=True)

                v.sync()
                if period:
                    time.sleep(period)
    finally:
        env.close()
        print("\nclosed")


if __name__ == "__main__":
    main()