import serial
import time
import math
import json
import mujoco
import mujoco.viewer

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
XML_PATH = '../mujoco/scene.xml' 
CONFIG_FILE = 'calib_config.json'
QPOS_INDICES = [0, 1, 2, 3, 4, 5]
# ---------------------

def set_torque(ser, motor_id, enable=True):
    """Enables torque so the motors will hold position and follow commands."""
    address = 41
    value = 1 if enable else 0
    length = 4
    instruction = 0x03
    checksum = ~(motor_id + length + instruction + address + value) & 0xFF
    packet = bytes([0xFF, 0xFF, motor_id, length, instruction, address, value, checksum])
    ser.write(packet)
    time.sleep(0.01)

def sync_write_positions(ser, motor_ids, raw_positions):
    """Sends goal positions to all motors simultaneously using Sync Write."""
    address = 42 # Goal Position (2 bytes)
    data_len = 2
    length = (data_len + 1) * len(motor_ids) + 4
    
    parameters = []
    for mid, pos in zip(motor_ids, raw_positions):
        # Ensure position is an integer between 0 and 4095
        clamped_pos = max(0, min(int(pos), 4095))
        pos_low = clamped_pos & 0xFF
        pos_high = (clamped_pos >> 8) & 0xFF
        parameters.extend([mid, pos_low, pos_high])
        
    checksum_sum = 254 + length + 0x83 + address + data_len + sum(parameters)
    checksum = (~checksum_sum) & 0xFF
    
    packet = [0xFF, 0xFF, 254, length, 0x83, address, data_len] + parameters + [checksum]
    ser.reset_input_buffer()
    ser.write(bytes(packet))

def main():
    # 1. Load Calibration
    try:
        with open(CONFIG_FILE, 'r') as f:
            calib = json.load(f)
    except FileNotFoundError:
        print(f"Error: {CONFIG_FILE} not found. Run calibrate_limits.py first.")
        return

    motor_ids = calib["motor_ids"]
    offsets = calib["joint_offsets"]
    directions = calib["joint_directions"]
    mins = calib["min_limits"]
    maxs = calib["max_limits"]

    # 2. Connect Serial & Enable Torque
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
    except Exception as e:
        print(f"Failed to open port: {e}")
        return

    print("Enabling torque on all motors...")
    for mid in motor_ids:
        set_torque(ser, mid, enable=True)

    # 3. Setup MuJoCo
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("\nWARNING: The physical arm will immediately snap to match the MuJoCo start pose!")
    print("Stand clear of the arm.")
    input("Press ENTER to launch...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Step the simulation so physics/dragging works
            mujoco.mj_step(model, data)
            
            raw_targets = []
            
            for i in range(len(motor_ids)):
                # Read URDF angle from MuJoCo
                urdf_angle = data.qpos[QPOS_INDICES[i]]
                
                # Inverse calibration: URDF -> Physical Radian
                # Since direction is 1 or -1, multiplying by it reverses the mapping safely
                physical_angle_rad = (urdf_angle * directions[i]) + offsets[i]
                
                # Clamp safely to known limits
                clamped_rad = max(mins[i], min(physical_angle_rad, maxs[i]))
                
                # Convert Radians to Ticks (0-4095)
                raw_pos = (clamped_rad / (2 * math.pi)) * 4096.0
                raw_targets.append(raw_pos)
                
            # Send all commands instantly
            sync_write_positions(ser, motor_ids, raw_targets)
            
            viewer.sync()
            
            # Run at ~50Hz (20ms loop)
            elapsed = time.time() - step_start
            time.sleep(max(0, 0.02 - elapsed))

    # Disable torque safely on exit
    print("\nShutting down. Disabling torque...")
    for mid in motor_ids:
        set_torque(ser, mid, enable=False)
    ser.close()

if __name__ == '__main__':
    main()