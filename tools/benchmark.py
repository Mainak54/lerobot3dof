#!/usr/bin/env python3
"""
benchmark.py — measure what actually limits training throughput.

    python tools/benchmark.py                    # the full sweep
    python tools/benchmark.py --quick            # just n_envs
    python tools/benchmark.py --envs 4 8 16 24

WHY THIS EXISTS
---------------
"Use the GPU to get more steps" is the wrong model for this environment. The
policy is a 256x256 MLP on a 14-dim input — roughly 70k parameters. A gradient
step on that is microseconds. What costs real time, per env step, is:

    frame_skip mj_step calls   120 of them at 5 Hz
    one offscreen render       640x480 = 307k pixels

Both happen in the 8 worker processes, not in torch. So the levers are n_envs,
render resolution, and thread oversubscription — not device.

The numbers below are from YOUR machine, which is the only ones that matter.

WHAT TO DO WITH THE RESULT
--------------------------
n_envs: take the knee of the curve, not the peak. Throughput usually climbs to
about your physical core count then flattens or dips as workers fight for
cores. Past the knee you are paying memory for nothing.

resolution: this one is a TRADE, not a free win. The detector was tuned at
640x480 with min_area_px 25. Area scales with pixel count, so if you drop to
160x120 (1/16 the pixels) you must divide min_area_px by 16 -> 2, and re-run
tools/tune_hsv.py to confirm the detection rate holds. A 30 mm cube at 30 cm
is still ~8 px across at 160x120, which is enough to centroid, but verify
rather than assume.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
# PyOpenGL selects its backend from PYOPENGL_PLATFORM, NOT from MUJOCO_GL.
# On a headless box with no display it otherwise resolves to None and fails
# with "'NoneType' object has no attribute 'eglQueryString'" deep inside the
# mujoco import. Setting both keeps them consistent.
os.environ.setdefault("PYOPENGL_PLATFORM",
                      os.environ.get("MUJOCO_GL", "egl"))
# One thread per worker. Without this each of N workers spawns its own BLAS
# pool and they thrash; this alone is often worth 30%.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import yaml

from rate import resolve as resolve_rate


def measure(cfg, n_envs, steps=300, no_render=False):
    """Steps per second, summed across workers, after a warmup."""
    from stable_baselines3.common.vec_env import SubprocVecEnv
    from env_3dof import SO101Approach3DOF

    if no_render:
        # env.blind returns a zero detection WITHOUT calling the renderer, so
        # this measures physics only. The gap between the two numbers is the
        # entire cost of rendering — and therefore the most a GPU could
        # possibly buy you.
        cfg = {**cfg, "env": {**cfg["env"], "blind": True}}

    def mk(rank):
        def _init():
            return SO101Approach3DOF(cfg, seed=1000 + rank)
        return _init

    venv = SubprocVecEnv([mk(i) for i in range(n_envs)])
    try:
        venv.reset()
        # Width from the action space, never a literal: the joint count
        # follows env.trained_joints and changes with the config.
        a = np.zeros((n_envs, venv.action_space.shape[0]), dtype=np.float32)
        for _ in range(20):                      # warmup: JIT, caches, EGL
            venv.step(a)
        t0 = time.perf_counter()
        n = 0
        while n < steps:
            venv.step(a)
            n += n_envs
        dt = time.perf_counter() - t0
        return n / dt
    finally:
        venv.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--envs", type=int, nargs="*", default=None)
    ap.add_argument("--res", type=int, nargs="*", default=[640, 320, 160],
                    help="render widths to try (height scales 4:3)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="n_envs sweep only")
    a = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg0 = resolve_rate(yaml.safe_load((root / a.config).read_text()))

    cores = os.cpu_count() or 8
    envs = a.envs or sorted({2, 4, 8, cores // 2, cores, cores + cores // 2})
    envs = [e for e in envs if e >= 1]

    hz = cfg0["env"]["control_hz"]
    fs = cfg0["env"]["frame_skip"]
    w, h = cfg0["detector"]["width"], cfg0["detector"]["height"]
    total = cfg0["train"]["total_timesteps"]

    print(f"cores {cores}   control {hz:g} Hz   frame_skip {fs}   "
          f"render {w}x{h}   budget {total:,} steps")
    print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}  "
          f"MUJOCO_GL={os.environ['MUJOCO_GL']}\n")

    quiet_after = [False]
    print("0. where the time goes  (at n_envs = physical cores)")
    probe = max(1, cores // 2)
    try:
        full = measure(cfg0, probe, a.steps)
        phys = measure(cfg0, probe, a.steps, no_render=True)
        share = 1 - full / phys if phys > full else 0.0
        print(f"  physics + render  {full:7.0f} steps/s")
        print(f"  physics only      {phys:7.0f} steps/s   (rendering disabled)")
        print(f"  rendering is      {100*share:6.1f}% of the wall time")
        if share > 0.5:
            print("  -> rendering dominates. A GPU (hardware EGL) is the fix;")
            print("     lowering the resolution is the fallback.")
        else:
            print("  -> physics dominates. A GPU will NOT help much; you need")
            print("     more cores, or a larger frame_skip / lower control_hz.")
    except Exception as e:
        print(f"  probe failed: {type(e).__name__}: {e}")
    print()

    print("A. n_envs")
    print(f"{'n_envs':>7} {'steps/s':>10} {'per env':>9} {'run time':>12}")
    best = (0, 0.0)
    for n in envs:
        try:
            sps = measure(cfg0, n, a.steps)
        except Exception as e:                    # OOM, too many processes
            # Print the whole thing. A bare exception NAME tells you nothing,
            # and the first failure here is almost always a config or scene
            # problem rather than a resource limit.
            print(f"{n:>7}  failed: {type(e).__name__}: {e}")
            if not quiet_after[0]:
                import traceback
                traceback.print_exc()
                quiet_after[0] = True
                print("  (further tracebacks suppressed; the cause is above)")
            continue
        hrs = total / sps / 3600
        print(f"{n:>7} {sps:10.0f} {sps/n:9.0f} {hrs:10.1f} h")
        if sps > best[1]:
            best = (n, sps)
    if best[1] == 0:
        print("\n  Every configuration failed — this is not a hardware limit.")
        print("  Read the traceback above; then check that the scene, config")
        print("  and fixed_angles.yaml all copied over intact:")
        print("      python sanity_check.py --episodes 2 --all-visible --no-dr")
        return
    print(f"\n  best {best[0]} envs at {best[1]:.0f} steps/s")
    print("  take the KNEE, not the peak — a 3% gain for double the RAM and "
          "double the\n  renderer memory is not worth it.")

    if a.quick:
        return

    print("\nB. render resolution   (at the best n_envs above)")
    print(f"{'render':>10} {'px':>9} {'steps/s':>10} {'vs 640':>8} "
          f"{'min_area_px':>12} {'run time':>11}")
    base = None
    for width in a.res:
        c = yaml.safe_load((root / a.config).read_text())
        c["detector"]["width"] = width
        c["detector"]["height"] = int(round(width * 3 / 4))
        # min_area_px is an AREA, so it scales with the pixel count, not the
        # width. Leave it at 25 after shrinking and the detector rejects the
        # cube as noise.
        scale = (width * (width * 3 / 4)) / (w * h)
        c["detector"]["min_area_px"] = max(4, int(round(25 * scale)))
        resolve_rate(c)
        try:
            sps = measure(c, best[0], a.steps)
        except Exception as e:
            print(f"{width:>10}  failed: {type(e).__name__}: {e}")
            continue
        base = base or sps
        print(f"{width:>6}x{c['detector']['height']:<3} "
              f"{width*c['detector']['height']:>9,} {sps:10.0f} "
              f"{sps/base:7.2f}x {c['detector']['min_area_px']:>12} "
              f"{total/sps/3600:9.1f} h")

    print("\n  Resolution is a TRADE. Before adopting a smaller render, set")
    print("  detector.min_area_px to the value above and run:")
    print("      python tools/tune_hsv.py sim -n 64")
    print("  The detection rate must hold. A faster env that cannot see the")
    print("  cube trains a worse policy faster.")


if __name__ == "__main__":
    main()