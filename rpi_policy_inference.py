#!/usr/bin/env python3
"""
Distributed Policy Deployment.
Laptop receives video/encoder streams from RPi, runs PPO inference, 
updates MuJoCo Viewer, and sends control targets back to RPi.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import struct
import threading
import time
from pathlib import Path

import cv2
import numpy as np
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

GREEN, RED, WHITE, AMBER = (60, 220, 90), (235, 70, 70), (245, 245, 245), (250, 190, 60)

JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]

# --- Global Video State ---
latest_frame = None

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def video_receiver_thread(ip, port):
    global latest_frame
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, port))
        print(f"[Video] Connected to RPi stream.")
    except Exception as e:
        print(f"[Video] Connection failed: {e}")
        return

    while True:
        try:
            length_bytes = recvall(s, 4)
            if not length_bytes: break
            msglen = struct.unpack('>L', length_bytes)[0]
            
            frame_data = recvall(s, msglen)
            if not frame_data: break
            
            # Decode JPEG and update global frame
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            latest_frame = frame
        except Exception:
            break
    s.close()

# --- Network Motor Helpers ---

def set_torque_tcp(conn_file, motor_ids, enable=True):
    req = json.dumps({"motor_ids": motor_ids, "torque": enable})
    conn_file.write(req + '\n')
    conn_file.flush()
    conn_file.readline()

def sync_read_write_tcp(conn_file, motor_ids, targets=None):
    req_dict = {"motor_ids": motor_ids}
    if targets is not None:
        req_dict["targets"] = targets
        
    conn_file.write(json.dumps(req_dict) + '\n')
    conn_file.flush()
    
    resp_line = conn_file.readline()
    if not resp_line: return None
    return json.loads(resp_line).get("positions", [])

# --- HUD ---

def overlay_hud(frame_bgr, mask, det, status_lines, show_mask=True):
    img = frame_bgr.copy()
    h, w = img.shape[:2]

    cv2.drawMarker(img, (w // 2, h // 2), (110, 110, 110), cv2.MARKER_CROSS, 18, 1)

    if det[3] > 0.5:
        cx, cy = int((det[0] + 1) * w / 2), int((1 - det[1]) * h / 2)
        half = max(int(0.5 * det[2] * math.sqrt(w * h)), 4)
        cv2.rectangle(img, (cx - half, cy - half), (cx + half, cy + half), GREEN, 2)
        cv2.drawMarker(img, (cx, cy), GREEN, cv2.MARKER_CROSS, 12, 1)
        cv2.putText(img, f"u:{det[0]:+.2f} v:{det[1]:+.2f} s:{det[2]:.3f}",
                    (6, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40, GREEN, 1, cv2.LINE_AA)
    else:
        cv2.putText(img, "NO DETECTION", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)

    if show_mask and mask is not None:
        img = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)])

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
    if abs(k - 1.0) < 0.02: return bgr
    interp = cv2.INTER_NEAREST if k > 1 else cv2.INTER_AREA
    return cv2.resize(bgr, (int(w * k), int(h * k)), interpolation=interp)

def main():
    global latest_frame
    ap = argparse.ArgumentParser()
    ap.add_argument("--ip", required=True, help="IP Address of Raspberry Pi")
    ap.add_argument("--snapshot", default="policy/snapshots/stage1_approach_3dof/20260819-123637/best.pt")
    ap.add_argument("--config", default="configs/stage1_3dof.yaml")
    ap.add_argument("--calib", default="calib_config.json")
    ap.add_argument("--no-viewer", action="store_true", help="Disable 3D MuJoCo window")
    ap.add_argument("--mask", action="store_true", help="Show HSV mask pane on startup")
    ap.add_argument("--window-width", type=int, default=1280)
    args = ap.parse_args()

    # 1. Load Calibration & Config
    calib_path = ROOT / args.calib
    if not calib_path.exists():
        calib_path = Path("Real") / args.calib
    with open(calib_path, 'r') as f: calib = json.load(f)

    with open(ROOT / args.config, 'r') as f: cfg = yaml.safe_load(f)

    snap = ROOT / args.snapshot
    print(f"Loading policy: {snap}")
    
    fa_path = snap.parent / "fixed_angles.yaml"
    if not fa_path.exists(): fa_path = ROOT / cfg["env"]["fixed_angles"]
    with open(fa_path, 'r') as f: fixed_angles = yaml.safe_load(f)["fixed_joints"]

    trained_joints = cfg["env"]["trained_joints"]
    max_delta = np.array(cfg["env"]["max_delta"], dtype=np.float32)
    a_filt = float(cfg["env"].get("action_filter", 0.5))
    control_hz = float(cfg["env"].get("control_hz", 30.0))
    max_steps = int(round(cfg["env"].get("episode_seconds", 30.0) * control_hz))
    has_memory = bool(cfg["env"].get("last_seen_memory", True))

    # 2. Setup AI Policy
    model = PPO.load(str(snap), device="cpu")
    vecnorm_path = snap.parent / f"vecnorm_{snap.stem}.pkl"
    normaliser = None
    if vecnorm_path.exists():
        import gymnasium as gym
        from gymnasium import spaces
        class DummyEnv(gym.Env):
            def __init__(self):
                obs_dim = 3 * len(trained_joints) + 5 + (3 if has_memory else 0)
                self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), np.float32)
                self.action_space = spaces.Box(-1.0, 1.0, (len(trained_joints),), np.float32)
        normaliser = VecNormalize.load(str(vecnorm_path), DummyVecEnv([lambda: DummyEnv()]))
        normaliser.training = False

    # 3. Setup MuJoCo Twin
    mj_model = mujoco.MjModel.from_xml_path(str(ROOT / cfg["env"].get("scene", "mujoco/scene.xml")))
    mj_data = mujoco.MjData(mj_model)
    qpos_map = {n: mj_model.jnt_qposadr[mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)] 
                for n in JOINT_NAMES if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, n) >= 0}

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data) if not args.no_viewer else None

    # 4. Start Network Links
    threading.Thread(target=video_receiver_thread, args=(args.ip, 5001), daemon=True).start()
    
    print(f"[Motors] Connecting to {args.ip}:5000...")
    s_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s_motor.connect((args.ip, 5000))
    conn_file = s_motor.makefile('rw')
    motor_ids = calib["motor_ids"]

    print("Enabling remote torque...")
    set_torque_tcp(conn_file, motor_ids, enable=True)

    det_cfg = DetectorConfig(**cfg["detector"])
    det_cfg.randomize_thresholds = False
    detector = CubeDetector(det_cfg)

    # 5. Initialization
    hw_angles = sync_read_write_tcp(conn_file, motor_ids)
    while hw_angles is None or None in hw_angles:
        hw_angles = sync_read_write_tcp(conn_file, motor_ids)
        time.sleep(0.01)

    target_qpos_trained = np.zeros(len(trained_joints), dtype=np.float32)
    prev_qpos_trained = np.zeros(len(trained_joints), dtype=np.float32)
    
    for i, name in enumerate(JOINT_NAMES):
        phys_rad = (hw_angles[i] / 4096.0) * (2 * math.pi)
        urdf_angle = calib["joint_directions"][i] * (phys_rad - calib["joint_offsets"][i])
        if name in qpos_map: mj_data.qpos[qpos_map[name]] = urdf_angle
        if name in trained_joints:
            idx = trained_joints.index(name)
            target_qpos_trained[idx] = prev_qpos_trained[idx] = urdf_angle

    prev_action, filt_action = np.zeros(len(trained_joints), dtype=np.float32), np.zeros(len(trained_joints), dtype=np.float32)
    seen_mem = np.zeros(3, dtype=np.float32)
    steps_unseen, step_count, saved_frames = 0, 0, 0
    show_mask = args.mask
    
    cv2.namedWindow("Distributed Inference | HUD", cv2.WINDOW_NORMAL)
    step_period = 1.0 / control_hz
    last_time = time.perf_counter()

    print("\n--- Distributed Policy Deployed ---")
    print("Waiting for video stream...")
    while latest_frame is None: time.sleep(0.1)

    try:
        while True:
            loop_start = time.perf_counter()
            dt = loop_start - last_time
            last_time = loop_start

            # --- A. Grab Network Frame & Detect ---
            frame = latest_frame.copy()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            det = detector.detect(frame_rgb)
            mask = detector.mask(frame_rgb) if show_mask else None

            if has_memory:
                if det[3] > 0.5:
                    seen_mem[:2], steps_unseen = det[:2], 0
                else:
                    steps_unseen += 1
                seen_mem[2] = min(steps_unseen / 30.0, 2.0)

            # --- B. Process Kinematics & Infer ---
            qvel_trained = (target_qpos_trained - prev_qpos_trained) / max(dt, 0.001)
            prev_qpos_trained = target_qpos_trained.copy()

            obs_parts = [target_qpos_trained, qvel_trained, prev_action, det, [min(step_count / max_steps, 1.0)]]
            if has_memory: obs_parts.append(seen_mem)

            obs = np.concatenate(obs_parts).astype(np.float32)
            obs_norm = normaliser.normalize_obs(obs) if normaliser else obs

            action, _ = model.predict(obs_norm, deterministic=True)
            filt_action = a_filt * filt_action + (1.0 - a_filt) * action
            target_qpos_trained += filt_action * max_delta

            # --- C. Network Command ---
            raw_targets = []
            for i, name in enumerate(JOINT_NAMES):
                cmd_rad = target_qpos_trained[trained_joints.index(name)] if name in trained_joints else fixed_angles[name]
                phys_rad = max(calib["min_limits"][i], min((cmd_rad * calib["joint_directions"][i]) + calib["joint_offsets"][i], calib["max_limits"][i]))
                
                # Wrap it in float() so JSON can serialize it
                raw_targets.append(float((phys_rad / (2 * math.pi)) * 4096.0))
            hw_raw_angles = sync_read_write_tcp(conn_file, motor_ids, raw_targets)
            
            # --- D. Sync MuJoCo ---
            if hw_raw_angles and None not in hw_raw_angles:
                for i, name in enumerate(JOINT_NAMES):
                    if name in qpos_map:
                        phys_rad = (hw_raw_angles[i] / 4096.0) * (2 * math.pi)
                        mj_data.qpos[qpos_map[name]] = calib["joint_directions"][i] * (phys_rad - calib["joint_offsets"][i])
                mujoco.mj_forward(mj_model, mj_data)
                if viewer and viewer.is_running(): viewer.sync()

            prev_action = action.copy()
            step_count += 1

            # --- E. Render HUD ---
            seen = det[3] > 0.5
            c_tol = float(cfg["reward"]["success"].get("center_tol", 0.40))
            centered = seen and abs(det[0]) <= c_tol and abs(det[1]) <= c_tol

            status_lines = [
                ("seen", seen, f"{int(seen)}"),
                ("centered", centered, f"{abs(det[0]):.2f}, {abs(det[1]):.2f} <= {c_tol}"),
                ("step", True, f"{step_count}/{max_steps}"),
            ]

            hud_img = overlay_hud(frame, mask, det, status_lines, show_mask)
            cv2.imshow("Distributed Inference | HUD", fit_window(hud_img, args.window_width))

            # --- F. Pacing & Controls ---
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")): break
            elif key == ord("m"): show_mask = not show_mask
            elif key == ord("r"):
                step_count, steps_unseen, seen_mem[:] = 0, 0, 0.0

            elapsed = time.perf_counter() - loop_start
            if (step_period - elapsed) > 0: time.sleep(step_period - elapsed)

    finally:
        print("\nStopping policy. Disabling remote torque...")
        try: set_torque_tcp(conn_file, motor_ids, enable=False)
        except Exception: pass
        cv2.destroyAllWindows()
        if viewer: viewer.close()
        s_motor.close()

if __name__ == "__main__":
    main()