#!/usr/bin/env python3
"""
Show what the SO-101 wrist camera sees.

Opens the normal interactive MuJoCo viewer plus a second live window rendering
the `wrist_cam` view, so you can drag the actuator sliders and watch the
gripper camera follow.

    python view_wrist_cam.py                      # live, both windows
    python view_wrist_cam.py --camera front       # any camera in the model
    python view_wrist_cam.py --list               # list available cameras
    python view_wrist_cam.py --snapshot out.png   # one frame, no window
    python view_wrist_cam.py --headless           # sweep the joints, save PNGs

Press S in the camera window to save the current frame, Q or Esc to quit.

Note on the marker geoms: the red housing box in camera.xml is centred exactly
on the lens origin, so the camera sits *inside* it and would otherwise see only
the inside of the box. This script hides any geom whose name starts with
"wrist_cam_" while rendering that camera, then restores it. The markers stay
fully visible in the interactive viewer.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError:
    sys.exit("mujoco is not installed:  pip install mujoco")

DEFAULT_SCENE = "mujoco/scene.xml"
MARKER_PREFIX = "wrist_cam_"


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def load(scene_path):
    path = Path(scene_path)
    if not path.exists():
        sys.exit(f"scene not found: {path.resolve()}\n"
                 f"run this from the folder containing 'mujoco/', or pass --scene")
    model = mujoco.MjModel.from_xml_path(str(path))
    return model, mujoco.MjData(model)


def camera_names(model):
    return [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)]


def resolve_camera(model, name):
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
    if cid < 0:
        sys.exit(f"no camera named '{name}'. available: {camera_names(model)}")
    return cid


def marker_geom_ids(model):
    """Geoms that would occlude the lens and must be hidden while rendering."""
    ids = []
    for i in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if gname and gname.startswith(MARKER_PREFIX):
            ids.append(i)
    return ids


class MarkerHider:
    """Temporarily set marker geoms fully transparent (MuJoCo skips alpha 0)."""

    def __init__(self, model, ids):
        self.model = model
        self.ids = ids
        self.saved = model.geom_rgba[ids].copy() if ids else None

    def __enter__(self):
        if self.ids:
            self.model.geom_rgba[self.ids, 3] = 0.0
        return self

    def __exit__(self, *exc):
        if self.ids:
            self.model.geom_rgba[self.ids] = self.saved
        return False


def home_pose(model, data):
    """Bend the arm into a reachable pose so the camera has something to see."""
    q = np.array([0.0, -0.9, 1.1, 0.5, 0.0, 1.0])
    n = min(len(q), model.nq)
    data.qpos[:n] = q[:n]
    if model.nu:
        data.ctrl[:min(model.nu, n)] = q[:min(model.nu, n)]
    mujoco.mj_forward(model, data)


# --------------------------------------------------------------------------
# display backends
# --------------------------------------------------------------------------
def get_display(width, height, title):
    """Return (show_fn, close_fn, backend_name). Prefers cv2, falls back to mpl."""
    try:
        import cv2

        def show(rgb):
            cv2.imshow(title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                return "quit"
            if key == ord("s"):
                return "save"
            return None

        return show, cv2.destroyAllWindows, "opencv"

    except ImportError:
        pass

    try:
        import matplotlib
        import matplotlib.pyplot as plt

        plt.ion()
        fig, ax = plt.subplots(figsize=(width / 100, height / 100), num=title)
        ax.axis("off")
        fig.tight_layout(pad=0)
        im = ax.imshow(np.zeros((height, width, 3), dtype=np.uint8))
        state = {"cmd": None}

        def on_key(event):
            if event.key in ("q", "escape"):
                state["cmd"] = "quit"
            elif event.key == "s":
                state["cmd"] = "save"

        fig.canvas.mpl_connect("key_press_event", on_key)

        def show(rgb):
            im.set_data(rgb)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            if not plt.fignum_exists(fig.number):
                return "quit"
            cmd, state["cmd"] = state["cmd"], None
            return cmd

        return show, lambda: plt.close(fig), "matplotlib"

    except ImportError:
        sys.exit("need either opencv-python or matplotlib for the live window:\n"
                 "  pip install opencv-python\n"
                 "or use --snapshot / --headless instead")


# --------------------------------------------------------------------------
# modes
# --------------------------------------------------------------------------
def snapshot(model, data, cam, width, height, out_path):
    from PIL import Image

    home_pose(model, data)
    with MarkerHider(model, marker_geom_ids(model)):
        with mujoco.Renderer(model, height, width) as r:
            r.update_scene(data, camera=cam)
            pixels = r.render()
    Image.fromarray(pixels).save(out_path)
    print(f"saved {out_path}  ({width}x{height}, camera '{cam}')")


def headless_sweep(model, data, cam, width, height, out_dir, frames):
    """No display: step a slow gripper open/close and dump PNGs."""
    from PIL import Image

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    home_pose(model, data)

    grip = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
    lo, hi = (model.actuator_ctrlrange[grip] if grip >= 0 else (0.0, 1.0))

    with MarkerHider(model, marker_geom_ids(model)):
        with mujoco.Renderer(model, height, width) as r:
            for k in range(frames):
                if grip >= 0:
                    phase = 0.5 * (1 - np.cos(2 * np.pi * k / frames))
                    data.ctrl[grip] = lo + phase * (hi - lo)
                for _ in range(25):
                    mujoco.mj_step(model, data)
                r.update_scene(data, camera=cam)
                Image.fromarray(r.render()).save(out / f"frame_{k:03d}.png")

    print(f"wrote {frames} frames to {out.resolve()}")


def live(model, data, cam, width, height, realtime):
    show, close, backend = get_display(width, height, f"MuJoCo camera: {cam}")
    print(f"camera window backend: {backend}")
    print("interactive viewer: open the Control panel to drive the joints")
    print("camera window: S = save frame, Q/Esc = quit")

    home_pose(model, data)
    hidden = marker_geom_ids(model)
    saved = 0

    with mujoco.Renderer(model, height, width) as renderer, \
            mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():
            tic = time.time()

            mujoco.mj_step(model, data)
            viewer.sync()

            with MarkerHider(model, hidden):
                renderer.update_scene(data, camera=cam)
                pixels = renderer.render()

            cmd = show(pixels)
            if cmd == "quit":
                break
            if cmd == "save":
                from PIL import Image
                name = f"{cam}_{saved:03d}.png"
                Image.fromarray(pixels).save(name)
                print(f"  saved {name}")
                saved += 1

            if realtime:
                lag = model.opt.timestep - (time.time() - tic)
                if lag > 0:
                    time.sleep(lag)

    close()


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scene", default=DEFAULT_SCENE)
    p.add_argument("--camera", default="wrist_cam")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--list", action="store_true", help="list cameras and exit")
    p.add_argument("--snapshot", metavar="PNG", help="save one frame and exit")
    p.add_argument("--headless", action="store_true",
                   help="no windows; sweep the gripper and dump PNGs")
    p.add_argument("--frames", type=int, default=60, help="frames for --headless")
    p.add_argument("--out-dir", default="camera_frames")
    p.add_argument("--no-realtime", action="store_true",
                   help="run as fast as possible instead of wall-clock pacing")
    args = p.parse_args()

    model, data = load(args.scene)

    if args.list:
        print(f"{model.ncam} cameras in {args.scene}:")
        for name in camera_names(model):
            print("   ", name)
        return

    resolve_camera(model, args.camera)

    if args.snapshot:
        snapshot(model, data, args.camera, args.width, args.height, args.snapshot)
    elif args.headless:
        headless_sweep(model, data, args.camera, args.width, args.height,
                       args.out_dir, args.frames)
    else:
        live(model, data, args.camera, args.width, args.height,
             realtime=not args.no_realtime)


if __name__ == "__main__":
    main()