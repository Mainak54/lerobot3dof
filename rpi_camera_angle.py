#!/usr/bin/env python3
import socket
import json
import struct
import time
import threading
import cv2
import numpy as np

# --- Configuration ---
RPI_IP = '192.168.29.26'  # <-- Change to your RPi's IP
MOTOR_PORT = 5000
VIDEO_PORT = 5001
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
# ---------------------

# Global shared frame to decouple video rendering from motor loop
latest_frame = None 

def recvall(sock, count):
    """Helper to receive exactly 'count' bytes."""
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def video_receiver_thread():
    global latest_frame
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((RPI_IP, VIDEO_PORT))
        print("[Video] Connected to RPi stream.")
    except Exception as e:
        print(f"[Video] Connection failed: {e}")
        return

    while True:
        try:
            # Read the 4-byte length prefix
            length_bytes = recvall(s, 4)
            if not length_bytes: break
            msglen = struct.unpack('>L', length_bytes)[0]
            
            # Read the actual JPEG data
            frame_data = recvall(s, msglen)
            if not frame_data: break
            
            # Decode JPEG back to BGR image
            frame = cv2.imdecode(np.frombuffer(frame_data, dtype=np.uint8), cv2.IMREAD_COLOR)
            latest_frame = frame
        except Exception as e:
            print(f"[Video] Stream error: {e}")
            break
            
    s.close()

def main():
    # Start the video receiver in the background
    threading.Thread(target=video_receiver_thread, daemon=True).start()

    # Connect to Motors
    s_motor = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"[Motors] Connecting to {RPI_IP}:{MOTOR_PORT}...")
    try:
        s_motor.connect((RPI_IP, MOTOR_PORT))
    except Exception as e:
        print(f"[Motors] Connection failed: {e}")
        return

    conn_file = s_motor.makefile('rw')
    print("Streams active. Press 'Q' or Esc in the video window to quit.\n")

    try:
        while True:
            # 1. Fetch Motors
            req = json.dumps({"motor_ids": MOTOR_IDS})
            conn_file.write(req + '\n')
            conn_file.flush()

            resp_line = conn_file.readline()
            if not resp_line: break
            
            raw_positions = json.loads(resp_line).get("positions", [])
            
            # Print motor output
            output = []
            for mid, raw_pos in zip(MOTOR_IDS, raw_positions):
                if raw_pos is not None:
                    output.append(f"M{mid}: {(raw_pos / 4096.0) * 360.0:6.1f}°")
                else:
                    output.append(f"M{mid}:  ERR  ")
            print("\r" + " | ".join(output), end="")

            # 2. Display Video (if we have received a frame)
            if latest_frame is not None:
                cv2.imshow("Remote RPi Camera", latest_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):
                break
                
            time.sleep(0.05) # Loop pacing

    except KeyboardInterrupt:
        pass
    finally:
        print("\nExiting...")
        s_motor.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()