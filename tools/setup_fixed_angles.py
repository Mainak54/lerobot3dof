#!/usr/bin/env python3
"""
setup_fixed_angles.py — choose the three FIXED joint angles for the reduced
3-DOF SO-101 approach task.

STRATEGY THIS SUPPORTS
----------------------
    TRAINED : shoulder_pan, shoulder_lift, wrist_flex
    FIXED   : elbow_flex, wrist_roll, gripper

WHY THIS TOOL EXISTS (read this before using it)
------------------------------------------------
shoulder_lift, elbow_flex and wrist_flex are three PARALLEL pitch axes, and
shoulder_pan merely rotates that plane about z. So:

  * (r, z) of the gripper pad is invariant to shoulder_pan  -> pan only
    handles azimuth, and it can always reach any azimuth in the spawn arc.
  * Freezing elbow_flex leaves TWO joints in the vertical plane. Two DOF is
    exactly enough to place the pad at a target (r, z) -- which means the
    approach pitch is then FULLY DETERMINED, not free:

        pitch = shoulder_lift + elbow_fixed + wrist_flex   (planar sum)

Consequence: the elbow angle you freeze *decides* how close to vertical your
approach can be at each spawn radius. There is no slider you can drag at
training time to fix a bad choice. Hence `sweep` before `tune`.

wrist_roll does not change (r, z) much -- the pad sits nearly on the roll axis
-- but it does set which way the jaws face. For a top-down grasp of a cube you
want the jaw axis perpendicular to the radial direction, which is roll = 0 for
the stock SO-101 zero. Sweep still evaluates it properly via full FK rather
than assuming that.

SUBCOMMANDS
-----------
  sweep   Grid-search elbow_flex (x wrist_roll) and rank candidates by how
          much of your cube spawn annulus admits a near-vertical pre-grasp.
          Headless. Run this FIRST.

  check   Full diagnostic for one specific candidate: per-radius table of
          achievable pitch error, required shoulder_lift / wrist_flex, and
          how much joint-space redundancy the policy has at each radius.

  tune    Launch the MuJoCo viewer with the fixed joints pinned. Drag the
          Control sliders, watch a live readout of pad height / approach
          angle / radius in the terminal, press 's' to save the current
          fixed-joint values to YAML.

  apply   Write a fixed_angles.yaml from explicit --elbow/--roll/--grip
          values without opening a viewer. Use this if you have no display at
          all: read the numbers off `check`, then apply them.

NO WINDOW APPEARS?
------------------
Run `python tools/setup_fixed_angles.py doctor` first. It tests each stage —
environment, packages, GL context, scene, and an actual 3-second viewer — and
tells you which one fails instead of leaving you to guess.

Common causes, in order:
In order of how often it is the cause:

  1. MUJOCO_GL is set to egl or osmesa. Both are OFFSCREEN backends - they
     render correctly and open no window, ever. If you exported egl for
     training it is still set. `tune` now overrides it with glfw
     automatically and says so; force it yourself with --gl glfw.

  2. macOS. The viewer must own the main thread, which python does not give
     it. Run `mjpython tools/setup_fixed_angles.py tune ...` instead of
     `python`. mjpython ships with the mujoco package.

  3. Headless machine, WSL without WSLg, or SSH without X forwarding. There
     is no display to draw on. Use `check` to read the numbers, then `apply`.
     `ssh -X` plus a local X server also works but is slow.

  4. glfw missing:  pip install glfw

Check which backend you actually got - `tune` prints it on startup.

USAGE
-----
    python setup_fixed_angles.py sweep  --scene mujoco/train_scene.xml
    python setup_fixed_angles.py check  --elbow 1.35 --roll 0.0
    python setup_fixed_angles.py tune   --elbow 1.35 --roll 0.0
    python setup_fixed_angles.py tune   --gl glfw          # force a window
    mjpython setup_fixed_angles.py tune                    # macOS
    python setup_fixed_angles.py apply  --elbow 1.35 --roll 0.0 --grip 0.5

OUTPUT
------
configs/fixed_angles.yaml, consumed later by the 3-DOF env. Format:

    fixed_joints:  {elbow_flex: 1.35, wrist_roll: 0.0, gripper: 0.5}
    trained_joints: [shoulder_pan, shoulder_lift, wrist_flex]

HOW THE FIXED JOINTS SHOULD BE ENFORCED IN SIM
----------------------------------------------
Do NOT weld them with <equality>. On the real arm those are STS3215 servos
held by position control -- they have finite stiffness and they sag under
load. Keeping the position actuator and simply never changing its ctrl
reproduces that. A weld gives you an infinitely rigid joint that exists only
in simulation, and the sim-to-real gap shows up as the gripper arriving a few
millimetres low. The env should therefore hold data.ctrl[fixed] constant and
only ever write to the three trained actuator indices.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# GL BACKEND. This must happen BEFORE `import mujoco`, because MuJoCo reads
# MUJOCO_GL once at import and ignores later changes.
#
# The usual reason no window appears: you have MUJOCO_GL=egl exported for
# training. egl and osmesa are OFFSCREEN backends - they render fine and open
# no window, ever. The interactive viewer needs a windowed backend (glfw), so
# `tune` overrides it unless you pass --gl explicitly.
# ---------------------------------------------------------------------------
_argv = sys.argv[1:]
if "--gl" in _argv:
    os.environ["MUJOCO_GL"] = _argv[_argv.index("--gl") + 1]
elif _argv[:1] in (["tune"], ["doctor"]):
    _prev = os.environ.get("MUJOCO_GL", "")
    if _prev.lower() in ("egl", "osmesa"):
        print(f"MUJOCO_GL={_prev} is an offscreen backend and will never open "
              f"a window; overriding with glfw for this run "
              f"(pass --gl {_prev} to keep it).")
    os.environ["MUJOCO_GL"] = "glfw"

try:
    import mujoco
except ImportError:
    sys.exit("mujoco is not installed.  pip install mujoco")

try:
    import yaml
except ImportError:
    yaml = None


# --------------------------------------------------------------------------
# Geometry constants, carried over from the 6-DOF project so the numbers stay
# comparable. All offsets are in the `gripper` BODY frame.
# --------------------------------------------------------------------------
PAD_OFFSET = np.array([-0.0069, 0.0, -0.0880])   # centre between finger pads
TIP_OFFSET = np.array([-0.0160, 0.0, -0.1044])   # fingertip
APPROACH_LOCAL = np.array([0.0, 0.0, -1.0])      # gripper -Z is the approach

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
              "wrist_flex", "wrist_roll", "gripper"]
TRAINED = ["shoulder_pan", "shoulder_lift", "wrist_flex"]
FIXED = ["elbow_flex", "wrist_roll", "gripper"]

DEFAULT_SCENE = "mujoco/train_scene.xml"


# --------------------------------------------------------------------------
# Model plumbing
# --------------------------------------------------------------------------
class Arm:
    """Thin wrapper holding the model plus every id we need."""

    def __init__(self, scene: str, gripper_body: str = "gripper"):
        path = Path(scene)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            sys.exit(f"scene not found: {path}\n"
                     f"pass --scene explicitly, e.g. --scene mujoco/train_scene.xml")

        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        self.scene_path = path

        self.jid, self.aid, self.qadr = {}, {}, {}
        for n in ARM_JOINTS:
            j = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
            a = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
            if j < 0:
                sys.exit(f"joint '{n}' not in the model -- is this the right scene?")
            self.jid[n], self.aid[n] = j, a
            self.qadr[n] = self.model.jnt_qposadr[j]

        self.gb = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                    gripper_body)
        if self.gb < 0:
            sys.exit(f"body '{gripper_body}' not found; pass --gripper-body")

        self.lo = {n: float(self.model.jnt_range[self.jid[n], 0]) for n in ARM_JOINTS}
        self.hi = {n: float(self.model.jnt_range[self.jid[n], 1]) for n in ARM_JOINTS}

    # -- kinematics --------------------------------------------------------
    def fk(self, q: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Set the named joints, run mj_forward, return (pad, tip, approach)."""
        for n, v in q.items():
            self.data.qpos[self.qadr[n]] = v
        mujoco.mj_forward(self.model, self.data)
        R = self.data.xmat[self.gb].reshape(3, 3)
        p = self.data.xpos[self.gb]
        return p + R @ PAD_OFFSET, p + R @ TIP_OFFSET, R @ APPROACH_LOCAL

    def limits_str(self) -> str:
        w = max(len(n) for n in ARM_JOINTS)
        out = []
        for n in ARM_JOINTS:
            tag = "TRAIN" if n in TRAINED else "FIXED"
            out.append(f"  {tag}  {n:<{w}}  [{self.lo[n]:+.3f}, {self.hi[n]:+.3f}] rad"
                       f"   [{math.degrees(self.lo[n]):+7.1f}, "
                       f"{math.degrees(self.hi[n]):+7.1f}] deg")
        return "\n".join(out)


# --------------------------------------------------------------------------
# Core evaluation
# --------------------------------------------------------------------------
def evaluate(arm: Arm, elbow: float, roll: float, grip: float,
             r_min: float, r_max: float, z_lo: float, z_hi: float,
             n_lift: int, n_wrist: int, n_bins: int):
    """
    For one fixed (elbow, roll, grip): sweep the two trained planar joints and
    report, per radius bin, the best achievable approach-pitch error.

    shoulder_pan is held at 0. It only rotates the plane, so (r, z) and the
    approach elevation are invariant to it -- the azimuth part of the problem
    is trivially solvable and is deliberately excluded here.

    Returns
    -------
    bins       : (n_bins,) bin centre radii
    best_err   : (n_bins,) minimum |approach - straight down| in degrees,
                 inf where the pre-grasp height band is unreachable at all
    best_q     : (n_bins, 2) the (shoulder_lift, wrist_flex) achieving it
    redundancy : (n_bins,) how many grid samples land in that bin within the
                 height band -- a proxy for how much slack the policy has
    """
    lifts = np.linspace(arm.lo["shoulder_lift"], arm.hi["shoulder_lift"], n_lift)
    wrists = np.linspace(arm.lo["wrist_flex"], arm.hi["wrist_flex"], n_wrist)

    edges = np.linspace(r_min, r_max, n_bins + 1)
    bins = 0.5 * (edges[:-1] + edges[1:])
    best_err = np.full(n_bins, np.inf)
    best_q = np.zeros((n_bins, 2))
    redundancy = np.zeros(n_bins, dtype=int)

    base = {"shoulder_pan": 0.0, "elbow_flex": elbow,
            "wrist_roll": roll, "gripper": grip}

    for sl in lifts:
        for wf in wrists:
            q = dict(base, shoulder_lift=sl, wrist_flex=wf)
            pad, tip, app = arm.fk(q)

            if not (z_lo <= pad[2] <= z_hi):
                continue
            # tip must not be through the floor -- catches poses that satisfy
            # the pad-height band only because the hand is pitched over.
            if tip[2] < 0.0:
                continue

            r = math.hypot(pad[0], pad[1])
            k = np.searchsorted(edges, r) - 1
            if k < 0 or k >= n_bins:
                continue

            redundancy[k] += 1
            # angle between approach vector and straight down (0,0,-1)
            c = float(np.clip(-app[2] / (np.linalg.norm(app) + 1e-12), -1.0, 1.0))
            err = math.degrees(math.acos(c))
            if err < best_err[k]:
                best_err[k] = err
                best_q[k] = (sl, wf)

    return bins, best_err, best_q, redundancy


def coverage(best_err: np.ndarray, tol_deg: float) -> float:
    """Fraction of radius bins where a within-tolerance approach exists."""
    return float(np.mean(best_err <= tol_deg))


# --------------------------------------------------------------------------
# sweep
# --------------------------------------------------------------------------
def cmd_sweep(a):
    arm = Arm(a.scene, a.gripper_body)
    print(f"scene: {arm.scene_path}")
    print(f"joints ({arm.model.njnt} total, {arm.model.nu} actuators):")
    print(arm.limits_str())
    print()

    elbows = np.linspace(a.elbow_min, a.elbow_max, a.n_elbow)
    rolls = np.array([a.roll]) if a.roll is not None else \
        np.linspace(arm.lo["wrist_roll"], arm.hi["wrist_roll"], a.n_roll)

    print(f"sweeping {len(elbows)} elbow x {len(rolls)} roll candidates, "
          f"{a.n_lift}x{a.n_wrist} planar grid each")
    print(f"pre-grasp band: pad z in [{a.z_lo:.3f}, {a.z_hi:.3f}] m   "
          f"spawn radius [{a.r_min:.2f}, {a.r_max:.2f}] m   "
          f"tol {a.tol:.0f} deg\n")

    results = []
    for e in elbows:
        for rl in rolls:
            bins, err, bq, red = evaluate(
                arm, e, rl, a.grip, a.r_min, a.r_max, a.z_lo, a.z_hi,
                a.n_lift, a.n_wrist, a.n_bins)
            cov = coverage(err, a.tol)
            finite = err[np.isfinite(err)]
            worst = float(finite.max()) if finite.size else float("inf")
            mean = float(finite.mean()) if finite.size else float("inf")
            slack = int(red.min())
            results.append(dict(elbow=e, roll=rl, cov=cov, mean=mean,
                                worst=worst, slack=slack))

    # Rank by coverage, break ties by mean pitch error, then by the *minimum*
    # redundancy across bins -- a candidate whose coverage relies on a single
    # knife-edge joint configuration is not one a stochastic policy can hold.
    results.sort(key=lambda d: (-d["cov"], d["mean"], -d["slack"]))

    print(f"{'elbow(deg)':>11} {'roll(deg)':>10} {'coverage':>9} "
          f"{'mean err':>9} {'worst':>8} {'min slack':>10}")
    print("-" * 62)
    for d in results[:a.top]:
        w = "  inf" if not math.isfinite(d["worst"]) else f"{d['worst']:6.1f}"
        m = "  inf" if not math.isfinite(d["mean"]) else f"{d['mean']:6.1f}"
        print(f"{math.degrees(d['elbow']):11.1f} {math.degrees(d['roll']):10.1f} "
              f"{d['cov']*100:8.0f}% {m:>9} {w:>8} {d['slack']:10d}")

    best = results[0]
    print()
    if best["cov"] < 0.85:
        print("!! No candidate covers 85% of the spawn annulus.")
        print("   The elbow freeze is too restrictive for this radius range.")
        print("   Options, in order of how much I'd prefer them:")
        print(f"     1. Shrink the spawn annulus. Try --r-min {a.r_min+0.03:.2f} "
              f"--r-max {a.r_max-0.03:.2f} and re-run; a smaller reachable")
        print("        workspace is far better than an unreachable one.")
        print(f"     2. Loosen the approach tolerance (--tol {a.tol+10:.0f}). "
              "Only if your gripper opening")
        print("        actually tolerates a tilted approach on a 30 mm cube.")
        print("     3. Train wrist_flex AND elbow_flex, freeze wrist_roll and")
        print("        gripper only. Still 4 DOF, and it restores pitch as a")
        print("        free variable.")
    else:
        print(f"best: elbow_flex = {best['elbow']:.4f} rad "
              f"({math.degrees(best['elbow']):.1f} deg), "
              f"wrist_roll = {best['roll']:.4f} rad "
              f"({math.degrees(best['roll']):.1f} deg)")
        print(f"      covers {best['cov']*100:.0f}% of the annulus, "
              f"mean pitch error {best['mean']:.1f} deg")
        print()
        print("next:")
        print(f"  python {Path(sys.argv[0]).name} check --elbow {best['elbow']:.4f} "
              f"--roll {best['roll']:.4f}")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------
def cmd_check(a):
    arm = Arm(a.scene, a.gripper_body)
    bins, err, bq, red = evaluate(
        arm, a.elbow, a.roll, a.grip, a.r_min, a.r_max, a.z_lo, a.z_hi,
        a.n_lift, a.n_wrist, a.n_bins)

    print(f"elbow_flex = {a.elbow:+.4f} rad ({math.degrees(a.elbow):+.1f} deg)")
    print(f"wrist_roll = {a.roll:+.4f} rad ({math.degrees(a.roll):+.1f} deg)")
    print(f"gripper    = {a.grip:+.4f}\n")
    print(f"{'radius(m)':>10} {'pitch err':>10} {'sh_lift':>9} {'wr_flex':>9} "
          f"{'slack':>7}  verdict")
    print("-" * 64)
    for i, r in enumerate(bins):
        if not math.isfinite(err[i]):
            print(f"{r:10.3f} {'---':>10} {'---':>9} {'---':>9} {0:7d}  "
                  f"UNREACHABLE at pre-grasp height")
            continue
        v = "ok" if err[i] <= a.tol else "TOO TILTED"
        if red[i] < 5 and err[i] <= a.tol:
            v = "ok but knife-edge"
        print(f"{r:10.3f} {err[i]:9.1f}d {bq[i,0]:+9.3f} {bq[i,1]:+9.3f} "
              f"{red[i]:7d}  {v}")

    cov = coverage(err, a.tol)
    print(f"\ncoverage at {a.tol:.0f} deg tolerance: {cov*100:.0f}% "
          f"of [{a.r_min:.2f}, {a.r_max:.2f}] m")

    ok = np.isfinite(err) & (err <= a.tol)
    if ok.any():
        lo_r, hi_r = bins[ok].min(), bins[ok].max()
        contiguous = ok[np.searchsorted(bins, lo_r):
                        np.searchsorted(bins, hi_r) + 1].all()
        print(f"usable radii: {lo_r:.3f} .. {hi_r:.3f} m"
              f"{'' if contiguous else '  (WITH GAPS -- see table)'}")
        print(f"\nSet these in your spawn config, otherwise you will spawn "
              f"cubes the arm\nphysically cannot pre-grasp and the policy will "
              f"be punished for geometry:")
        print(f"  spawn:\n    radius_min: {lo_r:.3f}\n    radius_max: {hi_r:.3f}")
    else:
        print("NOTHING is reachable within tolerance. Re-run sweep.")

    if a.save:
        write_yaml(a, Path(a.save))


# --------------------------------------------------------------------------
# tune  (interactive viewer)
# --------------------------------------------------------------------------
def cmd_tune(a):
    """Pose the arm by hand in the MuJoCo viewer and save the fixed angles.

    KINEMATIC BY DEFAULT. The loop calls mj_forward, not mj_step, so nothing
    is simulated and nothing fights you: whatever you drag a joint slider to
    is where the joint stays. That is what you want for choosing angles.

    Pass --dynamic to step physics instead. Then the position actuators hold
    the arm, so you must drag the CONTROL sliders rather than the Joint
    sliders - dragging a joint under a position actuator just snaps back.

    NOTHING IS PINNED. An earlier version of this tool re-pinned elbow_flex,
    wrist_roll and gripper every step, which made the three joints you are
    here to adjust the only three you could not move. Use --pin only if you
    have already chosen the fixed angles and want to explore the remaining
    workspace with them held.
    """
    import threading
    import time
    try:
        import mujoco.viewer as mjv
    except ImportError:
        sys.exit("mujoco.viewer is unavailable. Use `check` instead, or see "
                 "the troubleshooting notes at the bottom of --help.")

    arm = Arm(a.scene, a.gripper_body)
    m, d = arm.model, arm.data

    # Seed pose. --elbow/--roll/--grip are starting points, not constraints.
    seed = {"elbow_flex": a.elbow, "wrist_roll": a.roll, "gripper": a.grip}
    for n, v in seed.items():
        if v is not None:
            d.qpos[arm.qadr[n]] = float(np.clip(v, arm.lo[n], arm.hi[n]))
    for n in ARM_JOINTS:
        if arm.aid[n] >= 0:
            d.ctrl[arm.aid[n]] = d.qpos[arm.qadr[n]]
    mujoco.mj_forward(m, d)

    # ------------------------------------------------------------------
    # Bidirectional slider sync (kinematic mode only).
    #
    # The viewer has two slider panels and they drive different things:
    # "Control" writes data.ctrl, "Joint" writes data.qpos. In kinematic mode
    # there is no mj_step, so an actuator setpoint never reaches the joint and
    # the Control sliders appear dead - the slider moves, the arm does not.
    #
    # Rather than making you remember which panel is live, mirror whichever
    # one you touched onto the other. Both panels now move the arm.
    # ------------------------------------------------------------------
    q_idx = np.array([arm.qadr[n] for n in ARM_JOINTS])
    a_idx = np.array([arm.aid[n] for n in ARM_JOINTS])
    last_q = d.qpos[q_idx].copy()
    last_c = d.ctrl[a_idx].copy()

    def sync_sliders():
        """Return True if the pose changed this frame."""
        nonlocal last_q, last_c
        q, c = d.qpos[q_idx], d.ctrl[a_idx]
        # Control checked first: it is the panel people reach for.
        if np.any(np.abs(c - last_c) > 1e-9):
            d.qpos[q_idx] = c
        elif np.any(np.abs(q - last_q) > 1e-9):
            d.ctrl[a_idx] = q
        last_q = d.qpos[q_idx].copy()
        last_c = d.ctrl[a_idx].copy()

    state = {"save": False, "quit": False, "cover": False}

    def on_key(keycode):
        # The viewer only forwards keys its own UI did not consume, and it
        # consumes everything while the cursor is over a slider panel. So this
        # callback is best-effort; the stdin watcher below is the reliable
        # path. Unknown keys are echoed so you can tell whether the callback
        # is firing at all.
        ch = chr(keycode) if 0 < keycode < 0x110000 else ""
        if ch in ("s", "S"):
            state["save"] = True
        elif ch in ("c", "C"):
            state["cover"] = True
        elif ch in ("q", "Q"):
            state["quit"] = True
        elif ch.strip():
            print(f"\n[key '{ch}' received — callback is working; "
                  f"use s / c / q]")

    print("=" * 72)
    print("TUNE MODE — pose the arm, then press 's' to save the fixed angles")
    print("=" * 72)
    print(f"  GL backend      MUJOCO_GL={os.environ.get('MUJOCO_GL', '(default)')}")
    print(f"  mode            {'dynamic (physics stepping)' if a.dynamic else 'kinematic (mj_forward only)'}")
    print(f"  pinned joints   {'elbow_flex, wrist_roll, gripper' if a.pin else 'NONE — every joint is draggable'}")
    print()
    if a.dynamic:
        print("  Drag the CONTROL sliders in the left panel. In dynamic mode")
        print("  the position actuators hold the arm, so the Joint sliders")
        print("  snap back — that is the actuators working, not a bug.")
    else:
        print("  Drag EITHER slider group in the left panel — Control or")
        print("  Joint. They are mirrored, so both move the arm. Physics is")
        print("  not running, so joints stay exactly where you put them.")
        print("  (Press Tab if the panel is hidden.)")
    print()
    print("  s  save elbow_flex / wrist_roll / gripper to YAML")
    print("  c  score the CURRENT elbow+roll over the spawn annulus")
    print("  q  quit")
    print("=" * 72)
    print("  If the keys do nothing (the viewer swallows them while the")
    print("  cursor is over a slider panel), type the same letter into THIS")
    print("  terminal and press Enter. That path always works.")
    print("=" * 72)
    print(f"{'pad r':>8} {'pad z':>8} {'tip z':>8} {'tilt':>7}  "
          f"{'elbow':>8} {'roll':>8} {'grip':>7}  status")

    def stdin_watch():
        """Second, reliable input path: type s / c / q + Enter."""
        for line in sys.stdin:
            cmd = line.strip().lower()
            if cmd in ("s", "save"):
                state["save"] = True
            elif cmd in ("c", "cover", "coverage"):
                state["cover"] = True
            elif cmd in ("q", "quit", "exit"):
                state["quit"] = True

    threading.Thread(target=stdin_watch, daemon=True).start()

    last = 0.0
    with mjv.launch_passive(m, d, key_callback=on_key) as v:
        # Main thread drives the loop. Threading a passive viewer is fragile
        # and on macOS it silently fails to draw.
        while v.is_running() and not state["quit"]:
            if a.pin:
                for n, val in seed.items():
                    if val is not None:
                        d.qpos[arm.qadr[n]] = val
                        if arm.aid[n] >= 0:
                            d.ctrl[arm.aid[n]] = val

            if a.dynamic:
                mujoco.mj_step(m, d)
            else:
                sync_sliders()
                mujoco.mj_forward(m, d)

            now = time.time()
            if now - last > 0.2:
                last = now
                R = d.xmat[arm.gb].reshape(3, 3)
                p = d.xpos[arm.gb]
                pad, tip = p + R @ PAD_OFFSET, p + R @ TIP_OFFSET
                app = R @ APPROACH_LOCAL
                r = math.hypot(pad[0], pad[1])
                tilt = math.degrees(math.acos(float(np.clip(-app[2], -1.0, 1.0))))
                e = d.qpos[arm.qadr["elbow_flex"]]
                rl = d.qpos[arm.qadr["wrist_roll"]]
                g = d.qpos[arm.qadr["gripper"]]
                ok = (a.z_lo <= pad[2] <= a.z_hi and a.r_min <= r <= a.r_max
                      and tilt <= a.tol)
                status = "PRE-GRASP OK" if ok else " / ".join(filter(None, [
                    "" if a.z_lo <= pad[2] <= a.z_hi else "height",
                    "" if a.r_min <= r <= a.r_max else "radius",
                    "" if tilt <= a.tol else "tilt"])) + " out"
                print(f"\r{r:8.3f} {pad[2]:8.3f} {tip[2]:8.3f} {tilt:6.1f}d "
                      f"{math.degrees(e):7.1f}d {math.degrees(rl):7.1f}d "
                      f"{g:7.3f}  {status:<26}", end="", flush=True)

            if state["cover"]:
                state["cover"] = False
                e = float(d.qpos[arm.qadr["elbow_flex"]])
                rl = float(d.qpos[arm.qadr["wrist_roll"]])
                g = float(d.qpos[arm.qadr["gripper"]])
                print()
                print(f"scoring elbow={math.degrees(e):.1f}d "
                      f"roll={math.degrees(rl):.1f}d ...")
                # Uses a scratch MjData internally? No - evaluate() writes into
                # arm.data, so snapshot and restore the pose the user built.
                snap = d.qpos.copy()
                bins, err, bq, red = evaluate(
                    arm, e, rl, g, a.r_min, a.r_max, a.z_lo, a.z_hi,
                    a.n_lift, a.n_wrist, a.n_bins)
                d.qpos[:] = snap
                d.ctrl[a_idx] = d.qpos[q_idx]
                mujoco.mj_forward(m, d)
                last_q = d.qpos[q_idx].copy()
                last_c = d.ctrl[a_idx].copy()
                cov = coverage(err, a.tol)
                print(f"  coverage {cov*100:.0f}% of "
                      f"[{a.r_min:.2f}, {a.r_max:.2f}] m at {a.tol:.0f} deg")
                okb = np.isfinite(err) & (err <= a.tol)
                if okb.any():
                    print(f"  usable radii {bins[okb].min():.3f} .. "
                          f"{bins[okb].max():.3f} m")
                else:
                    print("  NOTHING reachable within tolerance at this elbow")

            if state["save"]:
                state["save"] = False
                _save_current(arm, d, a)

            v.sync()
            time.sleep(m.opt.timestep if a.dynamic else 0.01)

    # Autosave on close. Losing a pose you spent ten minutes on because a
    # keypress went to the wrong widget is not an acceptable failure mode.
    print("\nviewer closed")
    if not a.no_autosave:
        print("autosaving the final pose (--no-autosave to disable)")
        _save_current(arm, d, a)


def _save_current(arm, d, a):
    """Read the three fixed joints out of the live pose and write the YAML."""
    a.elbow = float(d.qpos[arm.qadr["elbow_flex"]])
    a.roll = float(d.qpos[arm.qadr["wrist_roll"]])
    a.grip = float(d.qpos[arm.qadr["gripper"]])
    out = Path(a.save or "configs/fixed_angles.yaml").resolve()
    print()
    try:
        write_yaml(a, out)
    except Exception as exc:                      # noqa: BLE001
        print(f"SAVE FAILED: {type(exc).__name__}: {exc}")
        print(f"  target was {out}")
        print("  pass --save <path> to write somewhere you can definitely "
              "write to.")


# --------------------------------------------------------------------------
# doctor
# --------------------------------------------------------------------------
def cmd_doctor(a):
    """Test every stage between here and a window, and say which one fails."""
    import platform
    import subprocess

    def ok(label, value=""):
        print(f"  [ ok ] {label}{('  ' + str(value)) if value else ''}")

    def bad(label, hint=""):
        print(f"  [FAIL] {label}")
        if hint:
            for line in hint.splitlines():
                print(f"         {line}")

    print("=" * 72)
    print("MUJOCO VIEWER DOCTOR")
    print("=" * 72)

    print("\n1. environment")
    ok("python", sys.version.split()[0])
    ok("platform", f"{platform.system()} {platform.release()}")
    ok("MUJOCO_GL", os.environ.get("MUJOCO_GL", "(unset)"))
    for var in ("DISPLAY", "WAYLAND_DISPLAY", "XDG_SESSION_TYPE", "SSH_CONNECTION"):
        v = os.environ.get(var)
        print(f"  [info] {var} = {v if v else '(unset)'}")
    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        bad("no DISPLAY and no WAYLAND_DISPLAY",
            "Nothing can open a window. If this is SSH, reconnect with ssh -X.\n"
            "Otherwise use `sweep` + `check` + `apply`, which need no display.")
    if os.environ.get("SSH_CONNECTION"):
        print("         (SSH session detected — a window will only appear with "
              "X forwarding)")

    print("\n2. packages")
    try:
        import mujoco as _mj
        ok("mujoco", _mj.__version__)
    except Exception as e:
        bad("import mujoco", f"{e}\npip install mujoco")
        return
    try:
        import glfw
        ok("glfw", getattr(glfw, "__version__", "?"))
    except Exception as e:
        bad("import glfw", f"{e}\npip install glfw\n"
                           "mujoco.viewer needs it for a windowed backend.")
    try:
        import mujoco.viewer as _v
        ok("mujoco.viewer importable")
    except Exception as e:
        bad("import mujoco.viewer", str(e))
        return

    print("\n3. GL context")
    try:
        import glfw as _g
        if not _g.init():
            bad("glfw.init() returned False",
                "No usable display. On Wayland try:  export GDK_BACKEND=x11\n"
                "or run from an Xorg session.")
        else:
            _g.window_hint(_g.VISIBLE, _g.FALSE)
            w = _g.create_window(64, 64, "probe", None, None)
            if w:
                ok("glfw window created")
                _g.destroy_window(w)
            else:
                bad("glfw.create_window() returned None",
                    "A display exists but no GL context could be made.\n"
                    "Common on Wayland, in containers, and over plain SSH.")
            _g.terminate()
    except Exception as e:
        bad("glfw probe", str(e))

    print("\n4. scene")
    try:
        arm = Arm(a.scene, a.gripper_body)
        ok("model loaded", f"{arm.model.njnt} joints, {arm.model.nu} actuators")
        ok("timestep", f"{arm.model.opt.timestep:g} s "
                       f"= {1/arm.model.opt.timestep:g} Hz")
    except SystemExit as e:
        bad("scene", str(e))
        return

    print("\n5. launch_passive")
    print("     A window titled 'MuJoCo' should appear now. It stays open")
    print("     until you close it or press Enter here. The arm is driven in")
    print("     a slow sine so a FROZEN window is obvious from a live one.")
    try:
        import math as _m
        import threading as _t
        import time
        import mujoco.viewer as mjv

        stop = {"go": False}
        _t.Thread(target=lambda: (sys.stdin.readline(),
                                  stop.__setitem__("go", True)),
                  daemon=True).start()

        j0 = arm.qadr["shoulder_pan"]
        base = float(arm.data.qpos[j0])
        with mjv.launch_passive(arm.model, arm.data) as v:
            t0, n = time.time(), 0
            while v.is_running() and not stop["go"]:
                el = time.time() - t0
                arm.data.qpos[j0] = base + 0.5 * _m.sin(el)
                mujoco.mj_forward(arm.model, arm.data)
                v.sync()
                n += 1
                if n % 50 == 0:
                    print(f"\r     {el:5.1f} s, {n} frames drawn — "
                          f"see a moving arm? (Enter to stop)", end="",
                          flush=True)
                time.sleep(0.02)
            arm.data.qpos[j0] = base
            closed_by_user = not v.is_running()
        print()
        if n == 0:
            bad("viewer closed immediately",
                "is_running() was False on the first check.")
        elif closed_by_user:
            ok(f"you closed the window after {n} frames — "
               "the viewer works")
        else:
            ok(f"viewer ran {n} frames")
            print("         If NO window ever appeared while that ran, the")
            print("         window is being created but not shown. Check:")
            print("           wmctrl -l            does it exist at all?")
            print("           another workspace    some WMs place it there")
            print("           picom/compiz off     compositors can eat it")
    except Exception as e:
        bad("launch_passive", f"{type(e).__name__}: {e}")
        print("\n  Full traceback:")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 72)
    print("If every stage passed but you still see nothing, paste this whole")
    print("output — the failure is outside anything this script controls.")
    print("=" * 72)


# --------------------------------------------------------------------------
# apply / yaml
# --------------------------------------------------------------------------
def write_yaml(a, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "fixed_joints": {
            "elbow_flex": round(float(a.elbow), 5),
            "wrist_roll": round(float(a.roll), 5),
            "gripper": round(float(a.grip), 5),
        },
        "trained_joints": list(TRAINED),
        "pregrasp": {
            "pad_z_min": float(a.z_lo),
            "pad_z_max": float(a.z_hi),
            "approach_tol_deg": float(a.tol),
        },
        "spawn_hint": {
            "radius_min": float(a.r_min),
            "radius_max": float(a.r_max),
        },
        "_note": ("Hold the fixed joints with their position actuators at a "
                  "constant ctrl. Do not weld them with <equality> -- the real "
                  "STS3215 servos have finite holding stiffness."),
    }
    if yaml is None:
        out.write_text(repr(doc))
        print(f"wrote {out}  (PyYAML missing, dumped as python repr)")
    else:
        out.write_text(yaml.safe_dump(doc, sort_keys=False))
        print(f"wrote {out}")
        print(yaml.safe_dump(doc, sort_keys=False))


def cmd_apply(a):
    Arm(a.scene, a.gripper_body)          # validate the angles are in range
    write_yaml(a, Path(a.save or "configs/fixed_angles.yaml"))


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Choose the fixed joint angles for the 3-DOF SO-101 task.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--scene", default=DEFAULT_SCENE)
        sp.add_argument("--gripper-body", default="gripper")
        sp.add_argument("--grip", type=float, default=0.5,
                        help="gripper opening (0.5 ~= 46.4 mm pad gap)")
        sp.add_argument("--r-min", type=float, default=0.15)
        sp.add_argument("--r-max", type=float, default=0.30)
        sp.add_argument("--z-lo", type=float, default=0.060,
                        help="pad height lower bound (cube top 0.03 + 30 mm)")
        sp.add_argument("--z-hi", type=float, default=0.090,
                        help="pad height upper bound (cube top 0.03 + 60 mm)")
        sp.add_argument("--tol", type=float, default=25.0,
                        help="max approach tilt off vertical, degrees")
        sp.add_argument("--n-lift", type=int, default=241)
        sp.add_argument("--n-wrist", type=int, default=241)
        sp.add_argument("--n-bins", type=int, default=16)
        sp.add_argument("--save", default=None)
        # Accepted everywhere so it never errors out, but it is only read by
        # the pre-import block at the top of this file, and only `tune` opens
        # a window at all.
        sp.add_argument("--gl", default=None,
                        help="force a MUJOCO_GL backend, e.g. glfw. Handled "
                             "before mujoco is imported. Only affects `tune`.")

    s = sub.add_parser("sweep", help="rank candidate fixed angles")
    common(s)
    s.add_argument("--elbow-min", type=float, default=0.0)
    s.add_argument("--elbow-max", type=float, default=2.6)
    s.add_argument("--n-elbow", type=int, default=27)
    s.add_argument("--roll", type=float, default=0.0,
                   help="fix roll at this value; omit to sweep it too")
    s.add_argument("--n-roll", type=int, default=5)
    s.add_argument("--top", type=int, default=12)
    s.set_defaults(func=cmd_sweep)

    c = sub.add_parser("check", help="diagnose one candidate")
    common(c)
    c.add_argument("--elbow", type=float, required=True)
    c.add_argument("--roll", type=float, default=0.0)
    c.set_defaults(func=cmd_check)

    t = sub.add_parser("tune", help="pose the arm by hand in the viewer")
    common(t)
    # Starting points, NOT constraints. Omit them to start from the model zero.
    t.add_argument("--elbow", type=float, default=None)
    t.add_argument("--roll", type=float, default=None)
    t.add_argument("--dynamic", action="store_true",
                   help="step physics instead of pure kinematics; drag the "
                        "CONTROL sliders in this mode")
    t.add_argument("--no-autosave", action="store_true",
                   help="do not write the YAML when the viewer closes")
    t.add_argument("--pin", action="store_true",
                   help="hold elbow/roll/gripper at the passed values. Only "
                        "useful once you have chosen them.")
    t.set_defaults(func=cmd_tune)

    dr = sub.add_parser("doctor", help="diagnose why no window appears")
    common(dr)
    dr.set_defaults(func=cmd_doctor)

    ap = sub.add_parser("apply", help="write yaml from explicit values")
    common(ap)
    ap.add_argument("--elbow", type=float, required=True)
    ap.add_argument("--roll", type=float, default=0.0)
    ap.set_defaults(func=cmd_apply)

    a = p.parse_args()
    if a.z_lo >= a.z_hi:
        sys.exit("--z-lo must be below --z-hi")
    a.func(a)


if __name__ == "__main__":
    main()