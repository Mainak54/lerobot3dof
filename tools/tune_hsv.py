#!/usr/bin/env python3
"""
fit_spawn.py — derive spawn.radius_min / radius_max from the FIXED angles.

    python tools/fit_spawn.py                 # report only
    python tools/fit_spawn.py --write         # update the config in place

WHY THIS IS NOT OPTIONAL
------------------------
The radii in stage1.yaml (0.15 - 0.30 m) were measured for the SIX-DOF arm,
where wrist_flex and elbow_flex together could tilt the gripper to vertical at
any reachable point. With elbow_flex frozen that freedom is gone: two joints in
the vertical plane are exactly enough to hit a target (r, z), which means the
approach pitch is then DETERMINED, not chosen.

So there is a band of radii the arm can reach but cannot reach VERTICALLY
enough to satisfy align_min. Cubes spawned there are unwinnable, and the policy
gets punished for geometry it has no way to influence. In the scripted probe
that shows up as `approach_angle` failures; in training it shows up as a
success rate that plateaus for no visible reason.

This tool finds the band that actually works and writes it down.

EVERYTHING IS READ FROM YOUR CONFIG — no magic numbers:

    fixed angles       configs/fixed_angles.yaml, as-is. Not re-optimised.
    tilt tolerance     acos(reward.success.align_min), so it matches the
                       success predicate exactly rather than approximating it
    pre-grasp height   spawn.cube_half + reward.success.z_min .. z_max
                       (z_min/z_max are STANDOFF above the cube, and the cube
                       rests on the floor, so pad height = half + standoff)

WHAT IT REPORTS
---------------
For each radius: the best achievable tilt, the joint angles that achieve it,
and how many grid samples land there. That last column is the one people skip.
A radius reachable through exactly one configuration is a knife edge — a
stochastic policy cannot hold it, and it will read as a mysterious failure at
one specific distance. Radii below --min-slack are excluded from the band.
"""

from __future__ import annotations

import argparse
import os

# Headless-safe GL defaults; see the note in sanity_check.py. fit_spawn only
# renders offscreen, so egl (or osmesa) is always correct here.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import math
import re
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from setup_fixed_angles import Arm, evaluate           # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def contiguous_band(bins, err, red, tol, min_slack):
    """Longest run of radii that are both within tolerance and not knife-edge.

    Longest CONTIGUOUS run, not simply min/max of all passing radii: a band
    with a hole in the middle would spawn cubes in the hole, which is the
    failure this tool exists to prevent.
    """
    ok = np.isfinite(err) & (err <= tol) & (red >= min_slack)
    best = cur = None
    for i, good in enumerate(ok):
        if good:
            cur = (i, i) if cur is None else (cur[0], i)
            if best is None or (cur[1] - cur[0]) > (best[1] - best[0]):
                best = cur
        else:
            cur = None
    return (None, None, ok) if best is None else (bins[best[0]],
                                                  bins[best[1]], ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--scene", default=None, help="defaults to env.scene")
    ap.add_argument("--scan-min", type=float, default=0.05)
    ap.add_argument("--scan-max", type=float, default=0.40)
    ap.add_argument("--bins", type=int, default=40)
    ap.add_argument("--n-lift", type=int, default=301)
    ap.add_argument("--n-wrist", type=int, default=301)
    ap.add_argument("--min-slack", type=int, default=5,
                    help="grid samples needed at a radius for it to count as "
                         "robustly reachable rather than a knife edge")
    ap.add_argument("--margin", type=float, default=0.005,
                    help="shrink the band by this at each end, in metres. The "
                         "grid is discrete and the cube has width; spawning "
                         "exactly at the edge is asking for trouble.")
    ap.add_argument("--write", action="store_true",
                    help="update radius_min/radius_max in the config")
    a = ap.parse_args()

    cfg_path = ROOT / a.config if not Path(a.config).is_absolute() else Path(a.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    fa_path = ROOT / cfg["env"]["fixed_angles"]
    if not fa_path.exists():
        sys.exit(f"{fa_path} not found — run setup_fixed_angles.py first.")
    fa = yaml.safe_load(fa_path.read_text())["fixed_joints"]
    elbow, roll, grip = (float(fa["elbow_flex"]), float(fa["wrist_roll"]),
                         float(fa["gripper"]))

    sc = cfg["reward"]["success"]
    half = float(cfg["spawn"]["cube_half"])
    z_lo, z_hi = half + float(sc["z_min"]), half + float(sc["z_max"])
    tol = math.degrees(math.acos(float(sc["align_min"])))

    scene = a.scene or cfg["env"]["scene"]
    arm = Arm(str(ROOT / scene) if not Path(scene).is_absolute() else scene,
              cfg["env"].get("gripper_body", "gripper"))

    print(f"fixed angles   elbow {math.degrees(elbow):+.1f}d   "
          f"roll {math.degrees(roll):+.1f}d   grip {grip:.3f}   (unchanged)")
    print(f"pre-grasp band pad z {z_lo:.3f} .. {z_hi:.3f} m "
          f"(cube_half {half:.3f} + standoff {sc['z_min']:.3f}..{sc['z_max']:.3f})")
    print(f"tilt tolerance {tol:.1f} deg  (align_min {sc['align_min']})")
    print(f"scanning       {a.scan_min:.2f} .. {a.scan_max:.2f} m, "
          f"{a.bins} bins, {a.n_lift}x{a.n_wrist} grid\n")

    bins, err, bq, red = evaluate(arm, elbow, roll, grip,
                                  a.scan_min, a.scan_max, z_lo, z_hi,
                                  a.n_lift, a.n_wrist, a.bins)

    lo, hi, ok = contiguous_band(bins, err, red, tol, a.min_slack)

    print(f"{'radius':>8} {'tilt':>8} {'sh_lift':>9} {'wr_flex':>9} "
          f"{'slack':>7}  ")
    print("-" * 52)
    for i, r in enumerate(bins):
        if not np.isfinite(err[i]):
            mark = "unreachable"
            print(f"{r:8.3f} {'---':>8} {'---':>9} {'---':>9} {0:7d}  {mark}")
            continue
        if err[i] > tol:
            mark = "too tilted"
        elif red[i] < a.min_slack:
            mark = "knife edge"
        elif lo is not None and lo <= r <= hi:
            mark = "USE"
        else:
            mark = "ok, outside the contiguous band"
        print(f"{r:8.3f} {err[i]:7.1f}d {bq[i,0]:+9.3f} {bq[i,1]:+9.3f} "
              f"{red[i]:7d}  {mark}")

    if lo is None:
        sys.exit("\nNo usable band at these fixed angles. Either the pre-grasp "
                 "height band or align_min is unsatisfiable for this elbow — "
                 "re-run `setup_fixed_angles.py sweep`.")

    r_min, r_max = lo + a.margin, hi - a.margin
    if r_max <= r_min:
        sys.exit(f"\nBand {lo:.3f}..{hi:.3f} m is narrower than twice the "
                 f"{a.margin:.3f} m margin. Pass a smaller --margin, but a band "
                 f"this thin means almost every cube sits at the edge of what "
                 f"the frozen elbow can do.")

    cur = cfg["spawn"]
    print(f"\nreachable band   {lo:.3f} .. {hi:.3f} m")
    print(f"with margin      {r_min:.3f} .. {r_max:.3f} m")
    print(f"currently        {cur['radius_min']:.3f} .. {cur['radius_max']:.3f} m")
    lost = 1 - (r_max - r_min) / (cur["radius_max"] - cur["radius_min"])
    print(f"                 {'narrower' if lost > 0 else 'wider'} by "
          f"{abs(lost)*100:.0f}%")

    if not a.write:
        print("\nre-run with --write to apply, or edit by hand:")
        print(f"  spawn:\n    radius_min: {r_min:.3f}\n    radius_max: {r_max:.3f}")
        return

    # Line-level edit rather than a yaml round-trip: PyYAML would strip every
    # comment in the file, and those comments are the record of why each value
    # is what it is.
    text = cfg_path.read_text()
    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    backup.write_text(text)
    new, n = text, 0
    for key, val in (("radius_min", r_min), ("radius_max", r_max)):
        new, k = re.subn(rf"^(\s*){key}:\s*[-\d.eE+]+",
                         lambda m, v=val: f"{m.group(1)}{key}: {v:.3f}",
                         new, count=1, flags=re.M)
        n += k
    if n != 2:
        sys.exit(f"expected to rewrite 2 lines, rewrote {n}. Config left "
                 f"untouched; edit by hand.")
    cfg_path.write_text(new)
    print(f"\nwrote {cfg_path}   (backup at {backup.name})")
    print("next:  python sanity_check.py --episodes 30 --all-visible --no-dr")


if __name__ == "__main__":
    main()