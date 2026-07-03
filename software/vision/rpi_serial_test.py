import serial
import struct
import threading
import time
import math
import sys

# Default to /dev/ttyACM0 for Raspberry Pi USB, but allow overriding via command line
SERIAL_PORT = sys.argv[1] if len(sys.argv) > 1 else "/dev/ttyACM0"
BAUD_RATE = 1000000

# --- COBS IMPLEMENTATION ---
def cobs_encode(data: bytes) -> bytes:
    encoded = bytearray()
    code_idx = 0
    encoded.append(0)
    code = 1
    for b in data:
        if b == 0:
            encoded[code_idx] = code
            code = 1
            code_idx = len(encoded)
            encoded.append(0)
        else:
            encoded.append(b)
            code += 1
            if code == 0xFF:
                encoded[code_idx] = code
                code = 1
                code_idx = len(encoded)
                encoded.append(0)
    encoded[code_idx] = code
    return bytes(encoded)

def cobs_decode(data: bytes) -> bytes:
    decoded = bytearray()
    idx = 0
    while idx < len(data):
        code = data[idx]
        idx += 1
        for i in range(1, code):
            if idx >= len(data): return bytes()
            decoded.append(data[idx])
            idx += 1
        if code != 0xFF and idx < len(data):
            decoded.append(0)
    return bytes(decoded)

# --- GLOBALS ---
telemetry_data = {}
ally_embeddings = []

def telemetry_reader_thread(ser):
    buffer = bytearray()
    while True:
        try:
            if ser.in_waiting > 0:
                b = ser.read(1)
                if not b: continue
                if b[0] == 0x00:
                    if len(buffer) > 0:
                        decoded = cobs_decode(buffer)
                        _process_packet(decoded)
                        buffer.clear()
                else:
                    buffer.append(b[0])
            else:
                time.sleep(0.001)
        except Exception as e:
            print(f"Serial Read Error: {e}")
            time.sleep(0.1)

def _process_packet(payload: bytes):
    if not payload: return
    header = payload[0:1]
    
    if header == b'T' and len(payload) == 54:
        data = struct.unpack('<12fBI', payload[1:])
        global telemetry_data
        telemetry_data = {
            "pose_x": data[0], "pose_y": data[1], "pose_th": data[2],
            "vel_x": data[3], "vel_y": data[4], "omega": data[5],
            "timestamp": data[13]
        }
    elif header == b'E' and len(payload) == 69:
        data = struct.unpack('<17f', payload[1:])
        global ally_embeddings
        ally_embeddings = list(data)

def main():
    print("=========================================")
    print(" RASPBERRY PI <-> ESP32 SERIAL TEST TOOL ")
    print("=========================================")
    print(f"Connecting to ESP32 on {SERIAL_PORT} @ {BAUD_RATE} baud...")
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    except Exception as e:
        print(f"\nFATAL: Failed to open {SERIAL_PORT}")
        print(f"Error: {e}")
        print("Make sure the ESP32 is plugged into the Pi and you have permission (sudo chmod a+rw /dev/ttyACM0).")
        return

    # Start the COBS decoding thread
    t = threading.Thread(target=telemetry_reader_thread, args=(ser,), daemon=True)
    t.start()
    
    print("\nStarting Test Loop. Press Ctrl+C to stop.")
    start_time = time.time()
    
    try:
        while True:
            # 1. Send a dummy Motor Command (Spin slowly in a circle)
            t_now = time.time() - start_time
            vx = math.cos(t_now) * 0.2 # 0.2 m/s
            vy = math.sin(t_now) * 0.2
            omega = 0.5 # 0.5 rad/s
            
            packet_m = struct.pack('<cffff', b'M', vx, vy, omega, 0.0)
            ser.write(cobs_encode(packet_m) + b'\x00')
            
            # 2. Send dummy Neural Embeddings to test ESP-NOW Broadcasting
            # We'll send an array of 17 floats counting up: 0.0, 1.0, ... 16.0
            fake_embed = [float(i) for i in range(17)]
            packet_e = struct.pack('<c17f', b'E', *fake_embed)
            ser.write(cobs_encode(packet_e) + b'\x00')
            
            # 3. Print the incoming telemetry from the ESP32
            if telemetry_data:
                print(f"ESP32 TELEMETRY -> X:{telemetry_data['pose_x']:+5.2f}  Y:{telemetry_data['pose_y']:+5.2f}  Th:{telemetry_data['pose_th']:+5.2f}  |  Ally Embeddings Received: {len(ally_embeddings)}")
            else:
                print("Waiting for ESP32 telemetry... (Is the ESP32 sending?)")
                
            # Send at 50Hz (20ms)
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        print("\nStopping Robot...")
    finally:
        # Blast a Safety Stop 'S' command before closing
        safety_packet = cobs_encode(b'S') + b'\x00'
        ser.write(safety_packet)
        ser.close()
        print("Serial port closed.")

if __name__ == "__main__":
    main()
