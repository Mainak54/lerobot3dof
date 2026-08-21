#!/usr/bin/env python3
"""
Deploy a trained policy to the physical SO-100 arm with parallel 3D MuJoCo
viewer and wrist camera detection HUD.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import serial
import yaml
from PIL import Image

import mujoco
import mujoco.viewer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from vision.cv_detector import CubeDetector, DetectorConfig

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

ROOT = Path(__file__).resolve().parent

GREEN = (60, 220, 90)
RED = (235, 70, 70)
WHITE = (245, 245, 245)
AMBER = (250, 190, 60)

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

# --- Feetech STS Protocol Helpers ---

def set_torque(ser: serial.Serial, motor_ids: list[int], enable: bool = True):
    for mid in motor_ids:
        address, value, length, instruction = 41, (1 if enable else 0), 4, 0x03
        checksum = ~(mid + length + instruction + address + value) & 0xFF
        packet = bytes([0xFF, 0xFF, mid, length, instruction, address, value, checksum])
        ser.write(packet)
        time.sleep(0.005)

def sync_read_positions(ser: serial.Serial, motor_ids: list[int]):
    address, data_len = 56, 2
    length = len(motor_ids) + 4
    checksum_sum = 254 + length + 0x82 + address + data_len + sum(motor_ids)
    packet = [0xFF, 0xFF, 254, length, 0x82, address, data_len] + motor_ids + [(~checksum_sum) & 0xFF]
    
    ser.reset_input_buffer()
    ser.write(bytes(packet))
    
    expected_bytes = len(motor_ids) * 8
    start_time = time.perf_counter()
    while ser.in_waiting < expected_bytes:
        if time.perf_counter() - start_time > 0.04:
            return None
            
    raw_data = ser.read(expected_bytes)
    positions = []
    for i in range(len(motor_ids)):
        chunk = raw_data[i*8 : (i+1)*8]
        if len(chunk) == 8 and chunk[0] == 0xFF and chunk[1] == 0xFF and chunk[4] == 0:
            raw_pos = chunk[5] + (chunk[6] << 8)
            positions.append((raw_pos / 4096.0) * (2 * math.pi))
        else:
            positions.append(None)
    return positions

def sync_write_positions(ser: serial.Serial, motor_ids: list[int], raw_positions: list[float]):
    address, data_len = 42, 2
    length = (data_len + 1) * len(motor_ids) + 4
    parameters = []
    for mid, pos in zip(motor_ids, raw_positions):
        clamped_pos = max(0, min(int(pos), 4095))
        parameters.extend([mid, clamped_pos & 0xFF, (clamped_pos >> 8) & 0xFF])
        
    checksum_sum = 254 + length + 0x83 + address + data_len + sum(parameters)
    packet = [0xFF, 0xFF, 254, length, 0x83, address, data_len] + parameters + [(~checksum_sum) & 0xFF]
    ser.reset_input_buffer()
    ser.write(bytes(packet))

# --- HUD and Overlay ---

def overlay_hud(frame_bgr, mask, det, status_lines, show_mask=True):
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    cv2.drawMarker(img, (w // 2, h // 2), (110, 110, 110), cv2.MARKER_CROSS, 18, 1)

    if det[3] > 0.5:
        cx = int((det[0] + 1) * w / 2)
        cy = int((1 - det[1]) * h / 2)
        half = max(int(0.5 * det[2] * math.sqrt(w * h)), 4)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), GREEN, 2)
        cv2.drawMarker(img, (cx, cy), GREEN, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, f"u:{det[0]:+.2f} v:{det[1]:+.2f} s:{det[2]:.3f}",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "NO DETECTION", (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)

    if show_mask and mask is not None:
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        img = np.hstack([img, mask_bgr])

    y = img.shape[0] - 10 - 18 * len(status_lines)
    x_offset = (w + 10) if (show_mask and mask is not None) else 10
    for name, ok, value in status_lines:
        cv2.putText(img, f"{'OK ' if ok else '   '}{name}: {value}",
                    (x_offset, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    GREEN if ok else AMBER, 1, cv2.LINE_AA)
        y += 18
    return img

def fit_window(bgr, target_width):
    h, w = bgr.shape[:2]
    k = target_width / float(w)
    if abs(k - 1.0) < 0.02:
        return bgr
    interp = cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA
    return cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=interp)

def main():
    ap = argparse.ArgumentParser()
    # Defaults mapped to your file tree exactly
    ap.add_argument("--snapshot", default="policy/snapshots/stage1_approach_3dof/20260819-123637/best.pt")
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--calib", default="calib_config.json")
    ap.add_argument("--camera", type=int, default=2, help="Camera index (/dev/videoX)")
    ap.add_argument("--serial", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1000000)
    ap.add_argument("--no-viewer", action="store_true", help="Disable 3D MuJoCo window")
    ap.add_argument("--mask", action="store_true", help="Show HSV mask pane on startup")
    ap.add_argument("--window-width", type=int, default=1280)
    args = ap.parse_args()

    # 1. Load Calibration and Configs
    calib_path = ROOT / args.calib
    if not calib_path.exists():
        raise SystemExit(f"\nERROR: {calib_path} not found. Run calibrate script first!\n")
            
    with open(calib_path, 'r') as f:
        calib = json.load(f)

    cfg_path = ROOT / args.config
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # 2. Check snapshot
    snap = ROOT / args.snapshot
    if not snap.exists():
        raise SystemExit(f"Checkpoint {snap} does not exist.")
    print(f"Loading policy: {snap}")

    # Resolve fixed angles (Look in snapshot folder first, fallback to root config folder)
    fa_path_snapshot = snap.parent / "fixed_angles.yaml"
    if fa_path_snapshot.exists():
        fa_path = fa_path_snapshot
    else:
        fa_path_raw = cfg["env"]["fixed_angles"]
        fa_path = Path(fa_path_raw) if Path(fa_path_raw).is_absolute() else ROOT / fa_path_raw

    with open(fa_path, 'r') as f:
        fixed_angles = yaml.safe_load(f)["fixed_joints"]

    trained_joints = cfg["env"]["trained_joints"]
    max_delta = np.array(cfg["env"]["max_delta"], dtype=np.float32)
    a_filt = float(cfg["env"].get("action_filter", 0.5))
    control_hz = float(cfg["env"].get("control_hz", 30.0))
    max_steps = int(round(cfg["env"].get("episode_seconds", 30.0) * control_hz))
    has_memory = bool(cfg["env"].get("last_seen_memory", True))

    # 3. Load Policy & VecNormalize
    model = PPO.load(str(snap), device="cpu")

    vecnorm_path = snap.parent / f"vecnorm_{snap.stem}.pkl"
    normaliser = None
    if vecnorm_path.exists():
        import gymnasium as gym
        from gymnasium import spaces
        
        class DummyEnv(gym.Env):
            def __init__(self):
                # Reconstruct the exact observation and action space dimensions
                obs_dim = 3 * len(trained_joints) + 5 + (3 if has_memory else 0)
                self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
                self.action_space = spaces.Box(-1.0, 1.0, (len(trained_joints),), np.float32)
                
        normaliser = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda: DummyEnv()]))
        normaliser.training = False
    else:
        print("WARNING: vecnorm pickle not found. Policy will receive unnormalized observations.")
    # 4. Setup MuJoCo Digital Twin
    scene_path = ROOT / cfg["env"].get("scene", "mujoco/scene.xml")
    mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
    mj_data = mujoco.MjData(mj_model)

    qpos_map = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            qpos_map[name] = mj_model.jnt_qposadr[jid]

    viewer = None
    if not args.no_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
        except Exception as e:
            print(f"Could not open MuJoCo 3D viewer: {e}")

    # 5. Hardware Setup
    print(f"Connecting to serial port {args.serial}...")
    ser = serial.Serial(args.serial, args.baud, timeout=0.01)
    motor_ids = calib["motor_ids"]

    print("Enabling torque on servos...")
    set_torque(ser, motor_ids, enable=True)

    print(f"Opening physical wrist camera at index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(cfg["detector"]["width"]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cfg["detector"]["height"]))

    det_cfg = DetectorConfig(**cfg["detector"])
    det_cfg.randomize_thresholds = False
    detector = CubeDetector(det_cfg)

    # Safe Startup & State Initialization
    initial_phys = sync_read_positions(ser, motor_ids)
    while initial_phys is None or None in initial_phys:
        initial_phys = sync_read_positions(ser, motor_ids)
        time.sleep(0.01)

    target_qpos_trained = np.zeros(len(trained_joints), dtype=np.float32)
    prev_qpos_trained = np.zeros(len(trained_joints), dtype=np.float32)
    for i, name in enumerate(JOINT_NAMES):
        if name in trained_joints:
            idx = trained_joints.index(name)
            urdf_angle = calib["joint_directions"][i] * (initial_phys[i] - calib["joint_offsets"][i])
            target_qpos_trained[idx] = urdf_angle
            prev_qpos_trained[idx] = urdf_angle

    prev_action = np.zeros(len(trained_joints), dtype=np.float32)
    filt_action = np.zeros(len(trained_joints), dtype=np.float32)
    seen_mem = np.zeros(3, dtype=np.float32)
    steps_unseen = 0
    step_count = 0
    saved_frames = 0
    show_mask = args.mask
    window_title = "Wrist Camera + Live Detection | HSV Mask"
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)

    step_period = 1.0 / control_hz
    last_time = time.perf_counter()
    print("\n--- Live Policy Deployed ---")
    print("Controls:  R: Reset progress   M: Toggle Mask   S: Save frame   Q/Esc: Quit\n")

    try:
        while True:
            loop_start = time.perf_counter()
            dt = loop_start - last_time
            last_time = loop_start

            # --- A. Hardware Feedback & MuJoCo Twin Sync ---
            hw_angles = sync_read_positions(ser, motor_ids)
            if hw_angles and None not in hw_angles:
                curr_qpos_trained = np.zeros(len(trained_joints), dtype=np.float32)
                for i, name in enumerate(JOINT_NAMES):
                    urdf_val = calib["joint_directions"][i] * (hw_angles[i] - calib["joint_offsets"][i])
                    if name in qpos_map:
                        mj_data.qpos[qpos_map[name]] = urdf_val
                    if name in trained_joints:
                        curr_qpos_trained[trained_joints.index(name)] = urdf_val

                qvel_trained = (curr_qpos_trained - prev_qpos_trained) / max(dt, 0.001)
                prev_qpos_trained = curr_qpos_trained.copy()

                mujoco.mj_forward(mj_model, mj_data)
                if viewer is not None and viewer.is_running():
                    viewer.sync()
            else:
                curr_qpos_trained = prev_qpos_trained.copy()
                qvel_trained = np.zeros(len(trained_joints), dtype=np.float32)

            # --- B. Physical Camera & Detection ---
            ret, frame = cap.read()
            if not ret:
                continue

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = detector.detect(frame_rgb)
            mask = detector.mask(frame_rgb) if show_mask else None

            if has_memory:
                if det[3] > 0.5:
                    seen_mem[:2] = det[:2]
                    steps_unseen = 0
                else:
                    steps_unseen += 1
                seen_mem[2] = min(steps_unseen / 30.0, 2.0)

            # --- C. Assemble Observation & Predict ---
            obs_parts = [
                curr_qpos_trained,
                qvel_trained,
                prev_action,
                det,
                [min(step_count / max_steps, 1.0)],
            ]
            if has_memory:
                obs_parts.append(seen_mem)

            obs = np.concatenate(obs_parts).astype(np.float32)
            obs_norm = normaliser.normalize_obs(obs) if normaliser else obs

            action, _ = model.predict(obs_norm, deterministic=True)
            filt_action = a_filt * filt_action + (1.0 - a_filt) * action
            target_qpos_trained += filt_action * max_delta

            # --- D. Command Servos ---
            raw_targets = []
            for i, name in enumerate(JOINT_NAMES):
                if name in trained_joints:
                    cmd_rad = target_qpos_trained[trained_joints.index(name)]
                else:
                    cmd_rad = fixed_angles[name]
                phys_rad = (cmd_rad * calib["joint_directions"][i]) + calib["joint_offsets"][i]
                phys_rad = max(calib["min_limits"][i], min(phys_rad, calib["max_limits"][i]))
                raw_targets.append((phys_rad / (2 * math.pi)) * 4096.0)

            sync_write_positions(ser, motor_ids, raw_targets)
            prev_action = action.copy()
            step_count += 1

            # --- E. Render OpenCV HUD ---
            seen = det[3] > 0.5
            c_tol = float(cfg["reward"]["success"].get("center_tol", 0.40))
            centered = seen and abs(det[0]) <= c_tol and abs(det[1]) <= c_tol

            status_lines = [
                ("seen", seen, f"{int(seen)}"),
                ("centered", centered, f"{abs(det[0]):.2f}, {abs(det[1]):.2f} <= {c_tol}"),
                ("step", True, f"{step_count}/{max_steps}"),
            ]

            hud_img = overlay_hud(frame, mask, det, status_lines, show_mask)
            cv2.imshow(window_title, fit_window(hud_img, args.window_width))

            # --- F. Handle Keystrokes & Pacing ---
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            elif key == ord("m"):
                show_mask = not show_mask
            elif key == ord("r"):
                step_count = 0
                seen_mem[:] = 0.0
                steps_unseen = 0
                print("Episode step progress reset.")
            elif key == ord("s"):
                fname = f"real_frame_{saved_frames:03d}.png"
                Image.fromarray(cv2.cvtColor(hud_img, cv2.COLOR_BGR2RGB)).save(fname)
                print(f"Saved snapshot: {fname}")
                saved_frames += 1

            elapsed = time.perf_counter() - loop_start
            sleep_time = step_period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    finally:
        print("\nStopping policy. Disabling motor torque...")
        set_torque(ser, motor_ids, enable=False)
        cap.release()
        cv2.destroyAllWindows()
        if viewer is not None:
            viewer.close()
        ser.close()
        print("Hardware safely shutdown.")

if __name__ == "__main__":
    main()