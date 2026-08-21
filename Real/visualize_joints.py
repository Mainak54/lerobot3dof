import serial
import time
import math
import mujoco
import mujoco.viewer

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]

# Using your MuJoCo scene file
XML_PATH = '../mujoco/scene.xml' 

# Physical to URDF Mapping
# Hardware servos are usually 0-360° (0-4096). URDF joints are usually centered at 0 radians.
# You will likely need to tune these offsets so the physical zero matches the simulation zero.
# Assuming physical 180° (Pi radians) is URDF 0 for this example.
JOINT_OFFSETS = [math.pi, math.pi, math.pi, math.pi, math.pi, math.pi]

# Multiply by -1 if a URDF joint rotates the opposite way of the physical servo
JOINT_DIRECTIONS = [1, 1, 1, 1, 1, 1] 

# The indices in data.qpos corresponding to the 6 motors
# If your URDF has exactly 6 moving joints, this is usually 0 to 5.
QPOS_INDICES = [0, 1, 2, 3, 4, 5]
# ---------------------

def read_sts_position_rad(ser, motor_id):
    """Reads the motor position and returns it directly in radians."""
    address = 56
    bytes_to_read = 2
    checksum = ~(motor_id + 4 + 0x02 + address + bytes_to_read) & 0xFF
    packet = bytes([0xFF, 0xFF, motor_id, 4, 0x02, address, bytes_to_read, checksum])

    ser.reset_input_buffer()
    ser.write(packet)

    # Fast timeout for real-time streaming
    start_time = time.time()
    while ser.in_waiting < 8:
        if time.time() - start_time > 0.02: 
            return None

    response = ser.read(8)
    if response[0] == 0xFF and response[1] == 0xFF and response[2] == motor_id:
        if response[4] == 0:
            raw_pos = response[5] + (response[6] << 8)
            # Convert ticks (0-4095) to Radians (0 - 2*Pi)
            angle_rad = (raw_pos / 4096.0) * (2 * math.pi)
            return angle_rad
    return None

def main():
    # 1. Initialize Serial
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.01)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Failed to open port: {e}")
        return

    # 2. Load MuJoCo Model
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    print("Launching MuJoCo Viewer. Close the window to exit.")
    
    # 3. Launch Passive Viewer (non-blocking)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            
            # Read all motors
            for i, mid in enumerate(MOTOR_IDS):
                angle_rad = read_sts_position_rad(ser, mid)
                if angle_rad is not None:
                    # Apply calibration: Direction * (Hardware_Angle - Zero_Offset)
                    urdf_angle = JOINT_DIRECTIONS[i] * (angle_rad - JOINT_OFFSETS[i])
                    
                    # Overwrite the simulation state directly
                    qpos_idx = QPOS_INDICES[i]
                    data.qpos[qpos_idx] = urdf_angle
            
            # Update kinematics (computes geometries and meshes based on qpos)
            mujoco.mj_forward(model, data)
            
            # Sync the viewer with the updated data
            viewer.sync()

            # Throttle the loop to ~50Hz to avoid overwhelming the serial bus
            elapsed = time.time() - step_start
            time.sleep(max(0, 0.02 - elapsed))

    ser.close()
    print("Simulation closed.")

if __name__ == '__main__':
    main()