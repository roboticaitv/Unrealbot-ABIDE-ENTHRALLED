import serial
import struct
import threading
import time

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
    return bytes(encoded) + b'\x00'

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

class SerialBridge:
    def __init__(self, port='/dev/ttyACM0', baudrate=1000000):
        self.port = port
        self.baudrate = baudrate
        self.serial_conn = None
        self.running = False
        self.read_thread = None
        
        # Latest received telemetry
        self.telemetry = {
            "pose_x": 0.0, "pose_y": 0.0, "pose_th": 0.0,
            "vel_x": 0.0, "vel_y": 0.0, "omega": 0.0,
            "current_fl": 0.0, "current_fr": 0.0, "current_rl": 0.0,
            "accel_x": 0.0, "accel_y": 0.0, "gyro_z": 0.0,
            "sensors": 0, "timestamp": 0
        }
        
        self.ally_embeddings = [0.0] * 17
        self.telemetry_lock = threading.Lock()
        
    def __enter__(self):
        self.connect()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.disconnect()

    def connect(self):
        try:
            self.serial_conn = serial.Serial(self.port, self.baudrate, timeout=0.1)
            self.running = True
            self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
            self.read_thread.start()
            print(f"Connected to ESP32 on {self.port} at {self.baudrate} baud")
        except Exception as e:
            print(f"Failed to connect to serial port {self.port}: {e}")

    def disconnect(self):
        self.running = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
        if self.serial_conn and self.serial_conn.is_open:
            self.send_safety_stop()
            self.serial_conn.close()
            print("Disconnected from ESP32")

    def send_motor_command(self, vx, vy, omega, kick_strength):
        if not self.serial_conn or not self.serial_conn.is_open:
            return
        try:
            packet = struct.pack('<cffff', b'M', float(vx), float(vy), float(omega), float(kick_strength))
            self.serial_conn.write(cobs_encode(packet))
        except Exception as e:
            print(f"Error sending motor command: {e}")

    def send_pose_command(self, x, y, theta):
        if not self.serial_conn or not self.serial_conn.is_open:
            return
        try:
            packet = struct.pack('<cffff', b'P', float(x), float(y), float(theta), 0.0)
            self.serial_conn.write(cobs_encode(packet))
        except Exception as e:
            print(f"Error sending pose command: {e}")

    def send_embeddings(self, embeddings_list):
        if not self.serial_conn or not self.serial_conn.is_open:
            return
        if len(embeddings_list) != 17:
            return
        try:
            format_str = '<c17f'
            packet = struct.pack(format_str, b'E', *embeddings_list)
            self.serial_conn.write(cobs_encode(packet))
        except Exception as e:
            print(f"Error sending embeddings: {e}")

    def send_safety_stop(self):
        if not self.serial_conn or not self.serial_conn.is_open:
            return
        try:
            packet = b'S'
            self.serial_conn.write(cobs_encode(packet))
        except Exception as e:
            print(f"Error sending safety stop: {e}")

    def get_telemetry(self):
        with self.telemetry_lock:
            return self.telemetry.copy()
            
    def get_ally_embeddings(self):
        with self.telemetry_lock:
            return self.ally_embeddings.copy()

    def _read_loop(self):
        buffer = bytearray()
        while self.running and self.serial_conn.is_open:
            try:
                if self.serial_conn.in_waiting > 0:
                    b = self.serial_conn.read(1)
                    if not b: continue
                    
                    if b[0] == 0x00:
                        if len(buffer) > 0:
                            decoded = cobs_decode(buffer)
                            self._process_packet(decoded)
                            buffer.clear()
                    else:
                        buffer.append(b[0])
                else:
                    time.sleep(0.001)
            except Exception as e:
                print(f"Error reading from serial: {e}")
                time.sleep(0.1)

    def _process_packet(self, payload: bytes):
        if not payload: return
        header = payload[0:1]
        
        if header == b'T' and len(payload) == 54:
            data = struct.unpack('<12fBI', payload[1:])
            with self.telemetry_lock:
                self.telemetry["pose_x"] = data[0]
                self.telemetry["pose_y"] = data[1]
                self.telemetry["pose_th"] = data[2]
                self.telemetry["vel_x"] = data[3]
                self.telemetry["vel_y"] = data[4]
                self.telemetry["omega"] = data[5]
                self.telemetry["current_fl"] = data[6]
                self.telemetry["current_fr"] = data[7]
                self.telemetry["current_rl"] = data[8]
                self.telemetry["accel_x"] = data[9]
                self.telemetry["accel_y"] = data[10]
                self.telemetry["gyro_z"] = data[11]
                self.telemetry["sensors"] = data[12]
                self.telemetry["timestamp"] = data[13]
        elif header == b'E' and len(payload) == 69:
            data = struct.unpack('<17f', payload[1:])
            with self.telemetry_lock:
                self.ally_embeddings = list(data)

if __name__ == "__main__":
    # Test
    with SerialBridge(port='COM1') as bridge:
        bridge.send_motor_command(0.5, 0.0, 0.0, 0.0)
        time.sleep(1)
        print(bridge.get_telemetry())
