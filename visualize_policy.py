#!/usr/bin/env python3
"""
Watch a trained 3-DOF policy run.

    python visualize_policy.py --snapshot runs/approach_3dof/best/best_model.zip
    python visualize_policy.py --scripted        # no policy needed
    python visualize_policy.py --snapshot ... --speed 0.25 --no-viewer --record out/

Opens the interactive MuJoCo viewer plus a second window showing the wrist
camera with the live detection drawn on it, the HSV mask beside it, and the
success criteria as a colour-coded readout. When the policy does something
odd, this tells you within seconds whether it is a perception problem or a
control problem — which the scalar logs cannot.

Keys in the camera window:  R reset episode,  M toggle mask,  S save frame,
                            Q or Esc quit

ONE CHANGE FROM THE 6-DOF VERSION WORTH KNOWING
------------------------------------------------
1. --scripted USES POSITION-ONLY IK. With the elbow frozen there are three
   joints for three position constraints, so orientation is not commanded.
   The readout shows the resulting tilt so you can see the geometry the frozen
   elbow handed you.

GL BACKENDS — the one thing that bites here
--------------------------------------------
This script needs TWO kinds of rendering at once: an offscreen render for the
wrist camera, and the interactive 3D viewer. MUJOCO_GL controls only the
offscreen one; mujoco.viewer opens its own GLFW window regardless.

    MUJOCO_GL=egl     correct. Offscreen on the GPU, viewer in its own window.
    MUJOCO_GL=glfw    SEGFAULTS. Two GLFW contexts in one process.
    MUJOCO_GL=osmesa  works, CPU offscreen, slow but fine for a look.

The script defaults to egl, so just run it without setting the variable. If
your shell exports MUJOCO_GL=glfw, override it:  MUJOCO_GL=egl python ...

Headless box: no viewer window is possible, so use --no-viewer --record.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent

GREEN = (60, 220, 90)
RED = (235, 70, 70)
WHITE = (245, 245, 245)
AMBER = (250, 190, 60)
DOWN = np.array([0.0, 0.0, -1.0])


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


def overlay(frame, mask, det, status, cv2, show_mask=True):
    """Draw the detection and criteria readout; optionally append the mask."""
    img = frame.copy()
    h, w = img.shape[:2]

    cv2.drawMarker(img, (w // 2, h // 2), (110, 110, 110),
                   cv2.MARKER_CROSS, 18, 1)

    if det[3] > 0.5:
        # u, v are both in [-1, 1] with the SAME sign convention as pixel
        # indices: +u right, +v DOWN. See the note in the module docstring.
        cx = int((det[0] + 1) * w / 2)
        # v is +UP, so pixel row counts DOWN from it.
        cy = int((1 - det[1]) * h / 2)
        # det[2] is sqrt(bbox_area / image_area), so the equivalent square
        # side in pixels is det[2] * sqrt(w*h).
        half = max(int(0.5 * det[2] * math.sqrt(w * h)), 3)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half),
                      GREEN, 2)
        cv2.drawMarker(img, (cx, cy), GREEN, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, f"u{det[0]:+.2f} v{det[1]:+.2f} s{det[2]:.3f}",
                    (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREEN, 1,
                    cv2.LINE_AA)
    else:
        cv2.putText(img, "NO DETECTION", (10, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)

    if show_mask and mask is not None:
        img = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)])

    y = img.shape[0] - 8 - 16 * len(status)
    for name, ok, value in status:
        cv2.putText(img, f"{'OK ' if ok else '   '}{name}: {value}",
                    (img.shape[1] // 2 + 6 if show_mask and mask is not None else 10,
                     y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                    GREEN if ok else AMBER, 1, cv2.LINE_AA)
        y += 16
    return img


def criteria_status(env, det):
    """The six success conditions, plus the hold counter, as display rows."""
    c = env.reward_fn.c
    pinch, cube = env.pinch_pos, env.cube_pos
    dxy = float(np.linalg.norm(pinch[:2] - cube[:2]))
    dz = float(pinch[2] - cube[2])
    align = float(np.dot(env.approach_axis, DOWN))
    seen = det[3] > 0.5
    centred = seen and abs(det[0]) <= c.center_tol and abs(det[1]) <= c.center_tol
    speed = float(np.linalg.norm(env.last_pinch_vel))
    return [
        ("seen", seen, f"{int(seen)}"),
        ("centred", centred,
         f"{abs(det[0]):.2f},{abs(det[1]):.2f}<{c.center_tol}"),
        ("xy", dxy <= c.xy_tol, f"{dxy*1000:.0f}mm<{c.xy_tol*1000:.0f}"),
        ("standoff", c.z_min <= dz <= c.z_max, f"{dz*1000:.0f}mm"),
        ("align", align >= c.align_min, f"{align:.3f}"),
        ("slow", speed <= c.speed_max, f"{speed*100:.1f}cm/s"),
        ("hold", env.reward_fn.hold_counter >= c.hold_steps,
         f"{env.reward_fn.hold_counter}/{c.hold_steps}"),
        ("dwell", False,
         f"{env.reward_fn.dwell_steps}/{max(env.step_count,1)} "
         f"= {100*env.reward_fn.dwell_steps/max(env.step_count,1):.0f}%"),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=None)
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--scripted", action="store_true",
                    help="run the sanity_check IK controller instead of a policy")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no-dr", action="store_true")
    ap.add_argument("--all-visible", action="store_true")
    ap.add_argument("--no-viewer", action="store_true",
                    help="camera window only (or use with --record when headless)")
    ap.add_argument("--record", default=None, help="write frames to this directory")
    ap.add_argument("--decimate", type=int, default=1,
                    help="issue a new command every N control steps and hold "
                         "the target in between. 1 = 30 Hz (as trained), "
                         "3 = 10 Hz. OFF-POLICY: see the warning it prints.")
    ap.add_argument("--action-scale", type=float, default=1.0,
                    help="multiply every action by this. Also off-policy.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate. 1.0 = real time, 0.25 = quarter "
                         "speed, 0 = as fast as possible")
    ap.add_argument("--device", default="cpu",
                    help="torch device for inference. cpu is right for a "
                         "256x256 MLP and avoids fighting a running trainer "
                         "for the GPU.")
    ap.add_argument("--window-width", type=int, default=1280,
                    help="width in pixels of the camera window. The frame is "
                         "scaled to fit, up or down, whatever the render "
                         "resolution is.")
    ap.add_argument("--mask", action="store_true",
                    help="also show the HSV mask panel beside the camera "
                         "(M toggles it live either way)")
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0,
                    help="base seed; episode N uses seed+N, so the same cube "
                         "sequence replays every run. Pass -1 for a different "
                         "random sequence each time.")
    args = ap.parse_args()

    # DO NOT force glfw here. This script needs an OFFSCREEN renderer (the
    # wrist camera) as well as the interactive viewer. MUJOCO_GL selects the
    # backend for the offscreen renderer; set it to glfw and you get two GLFW
    # contexts in one process, which segfaults. mujoco.viewer opens its own
    # window independently of MUJOCO_GL, so egl for offscreen + the viewer's
    # own GLFW window is the combination that works.
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM",
                          os.environ.get("MUJOCO_GL", "egl"))

    import mujoco
    from env_3dof import SO101Approach3DOF

    if not args.scripted and not args.snapshot:
        raise SystemExit("pass --snapshot, or --scripted to run without a policy")

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    if args.snapshot:
        cand = Path(args.snapshot).parent / "config.yaml"
        if cand.exists():
            cfg_path = cand
    from rate import resolve as resolve_rate
    cfg = resolve_rate(yaml.safe_load(cfg_path.read_text()),
                       verbose=True)

    if args.snapshot:
        fa = Path(args.snapshot).parent / "fixed_angles.yaml"
        if fa.exists():
            cfg["env"]["fixed_angles"] = str(fa.resolve())
    if args.no_dr:
        cfg["randomization"]["enabled"] = False
    if args.all_visible:
        cfg["spawn"]["hidden_fraction"] = 0.0

    env = SO101Approach3DOF(cfg, seed=args.seed)

    predict = None
    if not args.scripted:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        snap = Path(args.snapshot)
        model = load_policy(PPO, snap, args.device)

        vecnorm = snap.parent / f"vecnorm_{snap.stem}.pkl"
        normaliser = None
        if vecnorm.exists():
            shell = VecNormalize.load(str(vecnorm), DummyVecEnv([lambda: env]))
            shell.training = False
            normaliser = shell
        else:
            print("WARNING: no vecnorm_*.pkl beside the snapshot. "
                  "normalize_obs is true, so the policy will see unnormalised "
                  "observations and behave randomly.")

        def predict(obs):
            o = normaliser.normalize_obs(obs) if normaliser else obs
            action, _ = model.predict(o, deterministic=args.deterministic)
            return action
    else:
        from sanity_check import solve_ik, track_step

    try:
        import cv2
    except ImportError:
        raise SystemExit("needs opencv:  pip install opencv-python")

    # Reuse the env's renderer. A second Renderer means a second GL context
    # for no benefit — and the frame must be the one the detector saw anyway,
    # so rendering it twice could only introduce a discrepancy.
    renderer = env.camera.renderer
    markers = [i for i in range(env.model.ngeom)
               if (mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, i)
                   or "").startswith("wrist_cam_")]

    def render_wrist():
        """Hide the camera housing, which is centred on the lens origin."""
        saved = env.model.geom_rgba[markers].copy() if markers else None
        if markers:
            env.model.geom_rgba[markers, 3] = 0.0
        try:
            renderer.update_scene(env.data,
                                  camera=cfg["env"].get("camera", "wrist_cam"))
            return renderer.render()
        finally:
            if markers:
                env.model.geom_rgba[markers] = saved

    viewer = None
    if not args.no_viewer:
        try:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
        except Exception as exc:                          # noqa: BLE001
            print(f"could not open the 3D viewer ({type(exc).__name__}: "
                  f"{exc}).\nContinuing with the camera window only — pass "
                  f"--no-viewer to skip this attempt.")

    record_dir = Path(args.record) if args.record else None
    if record_dir:
        record_dir.mkdir(parents=True, exist_ok=True)

    title = "wrist camera + detection | HSV mask"

    def fit(bgr):
        """Scale the composite to --window-width, up OR down.

        The old code multiplied by a fixed 3x, which was written for a 128x128
        render. At 640x480 with the mask panel alongside that is 1280x480 ->
        3840x1440, larger than most screens. Scaling to a target WIDTH works at
        any render resolution: small renders are magnified, large ones shrunk.
        INTER_NEAREST when magnifying so you can see individual mask pixels,
        INTER_AREA when shrinking because nearest-neighbour downsampling drops
        thin features like a one-pixel-wide detection box edge.
        """
        h, w = bgr.shape[:2]
        k = args.window_width / float(w)
        if abs(k - 1.0) < 0.02:
            return bgr
        interp = cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA
        return cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=interp)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)   # user can resize/maximise
    saved_frames = frame_idx = 0
    show_mask = args.mask

    # Wall-clock pacing. Without this the viewer runs as fast as the CPU
    # allows — one control step advances 33.3 ms of simulated time but costs a
    # couple of ms to compute, so it looks like the motors are flying. They
    # are not; the clock is.
    step_period = (1.0 / env.control_hz) / args.speed if args.speed > 0 else 0.0

    cmd_hz = env.control_hz / max(args.decimate, 1)
    print(f"playback {args.speed}x real time  ({env.control_hz:.0f} Hz control "
          f"loop, {cmd_hz:.1f} Hz commands)")
    print("fixed joints " + "  ".join(f"{k}={v:+.3f}"
                                      for k, v in env.fixed_vals.items()))
    if args.decimate > 1 or args.action_scale != 1.0:
        print("WARNING: --decimate / --action-scale run the policy OUTSIDE the "
              "conditions it was\n"
              f"         trained under. It was trained closed-loop at "
              f"{env.control_hz:.0f} Hz and cannot correct\n"
              "         during held steps, so expect overshoot and a lower "
              "success rate. Use this to\n"
              "         LOOK at the motion, not to judge the policy. To "
              "genuinely run slower, change\n"
              "         env.frame_skip in the config and retrain.")
    print("R reset   M mask   S save frame   Q/Esc quit")

    try:
        for ep in range(args.episodes):
            obs, _ = env.reset(seed=None if args.seed < 0 else args.seed + ep)
            q_target = None
            if args.scripted:
                target = env.cube_pos.copy()
                target[2] = 0.5 * (env.reward_fn.c.z_min + env.reward_fn.c.z_max)
                q_target, res, tilt = solve_ik(env, target)
                print(f"ep {ep}: IK residual {res*1000:.2f} mm, "
                      f"resulting tilt {tilt:.1f} deg")

            done = quit_all = False
            info = {}
            n_act = env.action_space.shape[0]
            held = np.zeros(n_act)

            while not done:
                tic = time.perf_counter()
                # Only query the controller every `decimate` steps. In between,
                # send a ZERO delta so the position target holds, rather than
                # repeating the action — repeating would keep the arm moving at
                # the same speed, just blind.
                if env.step_count % max(args.decimate, 1) == 0:
                    held = (track_step(env, q_target) if args.scripted
                            else predict(obs[None, :])[0])
                    action = held * args.action_scale
                else:
                    action = np.zeros(n_act)

                obs, reward, term, trunc, info = env.step(action)
                det = env.last_detection
                done = term or trunc

                if viewer is not None:
                    if not viewer.is_running():
                        quit_all = True
                        break
                    viewer.sync()

                rgb = render_wrist()
                mask = env.detector.mask(rgb) if show_mask else None
                img = overlay(rgb, mask, det, criteria_status(env, det), cv2,
                              show_mask)
                banner = f"ep {ep}  step {env.step_count}/{env.max_steps}"
                if info.get("in_pose"):
                    banner += "  IN POSE"
                elif info.get("is_success"):
                    banner += "  (reached earlier)"
                cv2.putText(img, banner, (6, 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, WHITE, 1, cv2.LINE_AA)

                cv2.imshow(title, fit(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    quit_all = True
                    break
                if key == ord("r"):
                    break
                if key == ord("m"):
                    show_mask = not show_mask
                if key == ord("s"):
                    from PIL import Image
                    name = f"frame_{saved_frames:03d}.png"
                    Image.fromarray(img).save(name)
                    print(f"  saved {name}")
                    saved_frames += 1

                if record_dir:
                    from PIL import Image
                    Image.fromarray(img).save(record_dir / f"{frame_idx:05d}.png")
                    frame_idx += 1

                if step_period:
                    lag = step_period - (time.perf_counter() - tic)
                    if lag > 0:
                        time.sleep(lag)

            if quit_all:
                break
            print(f"episode {ep}: "
                  f"{'reached' if info.get('is_success') else 'never reached'}"
                  f"  dwell {100*info.get('dwell_frac', 0.0):.0f}%"
                  f"  ttf {info.get('time_to_first', -1.0):.2f}s"
                  f"  ({env.step_count} steps)")
    finally:
        cv2.destroyAllWindows()
        # renderer belongs to env; env.close() disposes of it.
        if viewer is not None:
            viewer.close()
        env.close()
        if record_dir:
            print(f"wrote {frame_idx} frames to {record_dir}")
            print(f"ffmpeg -framerate 30 -i {record_dir}/%05d.png "
                  "-pix_fmt yuv420p out.mp4")


if __name__ == "__main__":
    main()