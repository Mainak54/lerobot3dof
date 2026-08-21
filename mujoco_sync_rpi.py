import socket
import json
import time
import math
import os
import mujoco
import mujoco.viewer

# --- Configuration ---
RPI_IP = '192.168.29.26'  # <-- CHANGE THIS TO YOUR PI'S IP ADDRESS
PORT = 5000
XML_PATH = 'mujoco/scene.xml' 
CONFIG_FILE = 'calib_config.json'

# These must match the order of motors [1, 2, 3, 4, 5, 6]
JOINT_NAMES = [
    "shoulder_pan", "shoulder_lift", "elbow_flex",
    "wrist_flex", "wrist_roll", "gripper"
]
# ---------------------

# Prevent GLFW conflicts
os.environ.setdefault("MUJOCO_GL", "egl")

def main():
    # 1. Load Calibration Data
    try:
        with open(CONFIG_FILE, 'r') as f:
            calib = json.load(f)
    except FileNotFoundError:
        print(f"Error: {CONFIG_FILE} not found. Ensure you copied it to the laptop.")
        return

    motor_ids = calib["motor_ids"]
    offsets = calib["joint_offsets"]
    directions = calib["joint_directions"]

    # 2. Setup MuJoCo Scene
    print(f"Loading MuJoCo scene from {XML_PATH}...")
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    # Map joint names to qpos indices dynamically
    qpos_map = {}
    for name in JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            qpos_map[name] = model.jnt_qposadr[jid]

    # 3. Connect to Raspberry Pi
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to Raspberry Pi at {RPI_IP}:{PORT}...")
    try:
        s.connect((RPI_IP, PORT))
    except Exception as e:
        print(f"Connection failed: {e}")
        print("Ensure rpi_motor_server.py is running on the Pi and the IP is correct.")
        return

    conn_file = s.makefile('rw')
    print("Connected! Launching MuJoCo Viewer...")

    # 4. Main Synchronization Loop
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()
            
            try:
                # Ask Pi for current angles
                req = json.dumps({"motor_ids": motor_ids})
                conn_file.write(req + '\n')
                conn_file.flush()

                # Read response
                resp_line = conn_file.readline()
                if not resp_line:
                    print("Connection dropped by Raspberry Pi.")
                    break

                resp = json.loads(resp_line)
                raw_positions = resp.get("positions", [])

                # Process positions and update MuJoCo
                for i, (name, raw_pos) in enumerate(zip(JOINT_NAMES, raw_positions)):
                    if raw_pos is not None and name in qpos_map:
                        # Convert raw ticks (0-4095) to Radians
                        phys_rad = (raw_pos / 4096.0) * (2 * math.pi)
                        
                        # Apply Calibration (Direction * (Hardware - Offset))
                        urdf_angle = directions[i] * (phys_rad - offsets[i])
                        
                        # Overwrite simulation state
                        data.qpos[qpos_map[name]] = urdf_angle

                # Compute forward kinematics so the meshes update visually
                mujoco.mj_forward(model, data)
                viewer.sync()

            except Exception as e:
                print(f"Network error during sync: {e}")
                break

            # Throttle loop to ~50Hz to keep network traffic stable
            elapsed = time.perf_counter() - step_start
            time.sleep(max(0, 0.02 - elapsed))

    s.close()
    print("Viewer closed.")

if __name__ == '__main__':
    main()