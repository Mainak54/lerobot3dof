import serial
import time
import math
import json

# --- Configuration ---
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
CONFIG_FILE = 'calib_config.json'
# ---------------------

def set_torque(ser, motor_id, enable=False):
    address = 41
    value = 1 if enable else 0
    length = 4
    instruction = 0x03
    checksum = ~(motor_id + length + instruction + address + value) & 0xFF
    packet = bytes([0xFF, 0xFF, motor_id, length, instruction, address, value, checksum])
    ser.write(packet)
    time.sleep(0.01)

def sync_read_positions(ser, motor_ids):
    """Uses Sync Read to fetch all motor positions efficiently."""
    address = 56
    data_len = 2
    length = len(motor_ids) + 4
    checksum_sum = 254 + length + 0x82 + address + data_len + sum(motor_ids)
    checksum = (~checksum_sum) & 0xFF
    
    packet = [0xFF, 0xFF, 254, length, 0x82, address, data_len] + motor_ids + [checksum]
    ser.reset_input_buffer()
    ser.write(bytes(packet))
    
    expected_bytes = len(motor_ids) * 8
    start_time = time.time()
    while ser.in_waiting < expected_bytes:
        if time.time() - start_time > 0.05:
            return None
            
    raw_data = ser.read(expected_bytes)
    positions = {}
    
    for i in range(len(motor_ids)):
        chunk = raw_data[i*8 : (i+1)*8]
        if len(chunk) == 8 and chunk[0] == 0xFF and chunk[1] == 0xFF:
            mid = chunk[2]
            if chunk[4] == 0: # Check error byte
                raw_pos = chunk[5] + (chunk[6] << 8)
                positions[mid] = (raw_pos / 4096.0) * (2 * math.pi)
                
    return [positions.get(mid, 0.0) for mid in motor_ids]

def main():
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    
    for mid in MOTOR_IDS:
        set_torque(ser, mid, enable=False)
    print("Torque disabled. The arm moves freely.\n")
    
    print("STEP 1: Move the hardware arm to the exact ZERO pose shown in MuJoCo.")
    input("Press ENTER when ready...")
    zero_offsets = sync_read_positions(ser, MOTOR_IDS)
    print(f"Zero offsets captured: {[round(z, 3) for z in zero_offsets]}\n")
    
    min_limits = list(zero_offsets)
    max_limits = list(zero_offsets)
    
    print("STEP 2: Move the arm freely to all its extreme physical limits.")
    print("Recording MIN and MAX angles. Press Ctrl+C when finished tracking limits.\n")
    
    try:
        while True:
            current_positions = sync_read_positions(ser, MOTOR_IDS)
            if current_positions:
                for i in range(len(MOTOR_IDS)):
                    if current_positions[i] < min_limits[i]:
                        min_limits[i] = current_positions[i]
                    if current_positions[i] > max_limits[i]:
                        max_limits[i] = current_positions[i]
                
                # Live status line
                status = " | ".join([f"M{m}: {c:.2f} ({min_limits[i]:.2f} to {max_limits[i]:.2f})" 
                                     for i, (m, c) in enumerate(zip(MOTOR_IDS, current_positions))])
                print("\r" + status, end="")
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        print("\n\nRecording stopped. Saving configuration...")
        
        config = {
            "motor_ids": MOTOR_IDS,
            "joint_offsets": zero_offsets,
            "min_limits": min_limits,
            "max_limits": max_limits,
            # Directions default to 1. Edit the JSON manually if a joint is inverted in MuJoCo.
            "joint_directions": [1, 1, 1, 1, 1, 1] 
        }
        
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"Calibration saved to {CONFIG_FILE}.")
        
    finally:
        ser.close()

if __name__ == '__main__':
    main()