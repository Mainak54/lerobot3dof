import serial
import time

# --- Configuration ---
# Update this to match your serial interface (e.g., /dev/ttyUSB0, /dev/ttyACM0)
SERIAL_PORT = '/dev/ttyACM0'
BAUDRATE = 1000000
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
# ---------------------

def read_sts_position(ser, motor_id):
    # Feetech STS Protocol: Read Present Position (Address 56, 2 bytes)
    # Packet: [0xFF, 0xFF, ID, Length(4), Instruction(0x02), Address(56), Bytes(2), Checksum]
    address = 56
    bytes_to_read = 2
    checksum = ~(motor_id + 4 + 0x02 + address + bytes_to_read) & 0xFF

    packet = bytes([0xFF, 0xFF, motor_id, 4, 0x02, address, bytes_to_read, checksum])

    ser.reset_input_buffer()
    ser.write(packet)

    # Wait up to 50ms for the 8-byte response
    start_time = time.time()
    while ser.in_waiting < 8:
        if time.time() - start_time > 0.05:
            return None

    response = ser.read(8)

    # Validate header and ID
    if response[0] == 0xFF and response[1] == 0xFF and response[2] == motor_id:
        error = response[4]
        if error == 0:
            # Little Endian: Low Byte first, High Byte second
            raw_pos = response[5] + (response[6] << 8)
            # STS3215 resolution is 4096 ticks for 360 degrees
            angle_deg = (raw_pos / 4096.0) * 360.0
            return raw_pos, angle_deg
    return None

def main():
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
        print(f"Connected to {SERIAL_PORT} at {BAUDRATE} baud.")
    except Exception as e:
        print(f"Failed to open port {SERIAL_PORT}: {e}")
        return

    print("Reading STS motors (Press Ctrl+C to stop)...")
    print("-" * 75)

    try:
        while True:
            output = []
            for mid in MOTOR_IDS:
                result = read_sts_position(ser, mid)
                if result:
                    _, angle = result
                    output.append(f"M{mid}: {angle:6.2f}°")
                else:
                    output.append(f"M{mid}:  ERR  ")

            # Print dynamically on the same line
            print("\r" + " | ".join(output), end="")
            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        ser.close()

if __name__ == '__main__':
    main()