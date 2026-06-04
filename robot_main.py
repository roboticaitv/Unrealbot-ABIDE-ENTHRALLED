import multiprocessing
import time
import sys
import os

# Add vision folder to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))

from vision.vision_manager import VisionManager
from vision.ai_engine import AIEngine
from vision.serial_bridge import SerialBridge

def vision_process(state_dict, ready_event):
    print("[Vision Process] Starting...")
    vision = VisionManager()
    vision.start()
    
    try:
        while True:
            # Get latest vision state
            # This runs as fast as the cameras capture
            state = vision.get_world_state()
            if state:
                # Update shared dictionary
                state_dict.update(state)
                # Signal AI process
                ready_event.set()
    except KeyboardInterrupt:
        pass
    finally:
        vision.stop()
        print("[Vision Process] Terminated.")

def ai_process(state_dict, ready_event):
    print("[AI Process] Starting...")
    ai = AIEngine(models_dir=os.path.join(os.path.dirname(__file__), "ONNX_models"))
    
    # Determine the serial port (typically /dev/ttyACM0 or /dev/ttyUSB0 on Pi)
    # Using a common default for ESP32 on Pi
    port = '/dev/ttyACM0' if os.path.exists('/dev/ttyACM0') else '/dev/ttyUSB0'
    
    # Start serial bridge
    # Provide fallback if port is not available (like running on PC)
    try:
        with SerialBridge(port=port, baudrate=1000000) as bridge:
            print(f"[AI Process] Serial bridge connected on {port}.")
            _run_ai_loop(ai, bridge, state_dict, ready_event)
    except Exception as e:
        print(f"[AI Process] Serial port {port} unavailable: {e}. Running AI purely locally.")
        _run_ai_loop(ai, None, state_dict, ready_event)

def _run_ai_loop(ai, bridge, state_dict, ready_event):
    frames_processed = 0
    start_time = time.time()
    
    try:
        while True:
            # Wait for vision to provide new frame (sleeps until signaled to save CPU)
            if ready_event.wait(timeout=0.1):
                ready_event.clear()
                
                # 1. Get snapshot of state
                current_state = dict(state_dict)
                
                # 2. Merge telemetry from ESP32 into state
                if bridge:
                    telemetry = bridge.get_telemetry()
                    if telemetry:
                        MAX_SPEED = 2.0
                        MAX_ACCEL = 10.0
                        MAX_GYRO = 10.0
                        STALL_THRESHOLD = 3.0
                        
                        vx, vy = telemetry.get("vel_x", 0.0), telemetry.get("vel_y", 0.0)
                        current_state["ego_speed_norm"] = min(1.0, (vx**2 + vy**2)**0.5 / MAX_SPEED)
                        
                        ax, ay = telemetry.get("accel_x", 0.0), telemetry.get("accel_y", 0.0)
                        current_state["ego_accel_norm"] = min(1.0, (ax**2 + ay**2)**0.5 / MAX_ACCEL)
                        
                        current_state["yaw_rate_norm"] = min(1.0, abs(telemetry.get("gyro_z", 0.0)) / MAX_GYRO)
                        
                        c_fl = telemetry.get("current_fl", 0.0)
                        c_fr = telemetry.get("current_fr", 0.0)
                        c_rl = telemetry.get("current_rl", 0.0)
                        if max(c_fl, c_fr, c_rl) > STALL_THRESHOLD:
                            current_state["hardware_fault_indicator"] = 1.0
                        else:
                            current_state["hardware_fault_indicator"] = 0.0

                # 3. Infer motor commands
                action, _ = ai.infer(current_state)
                
                # 4. Send motor commands
                if bridge:
                    bridge.send_motor_command(action["vx"], action["vy"], action["omega"], action["kick"])
                
                frames_processed += 1
                if time.time() - start_time > 5.0:
                    fps = frames_processed / (time.time() - start_time)
                    print(f"[AI Process] Logic running at {fps:.2f} FPS")
                    frames_processed = 0
                    start_time = time.time()
                    
    except KeyboardInterrupt:
        pass
    finally:
        print("[AI Process] Terminated.")

if __name__ == '__main__':
    print("======================================================")
    print("   UNREALBOT ABIDE-ENTHRALLED AUTONOMY ORCHESTRATOR   ")
    print("======================================================")
    print("[Orchestrator] Booting...")
    
    manager = multiprocessing.Manager()
    shared_state = manager.dict()
    frame_ready = multiprocessing.Event()
    
    p_vision = multiprocessing.Process(target=vision_process, args=(shared_state, frame_ready))
    p_ai = multiprocessing.Process(target=ai_process, args=(shared_state, frame_ready))
    
    p_vision.start()
    p_ai.start()
    
    try:
        p_vision.join()
        p_ai.join()
    except KeyboardInterrupt:
        print("\n[Orchestrator] Shutting down cleanly...")
        p_vision.terminate()
        p_ai.terminate()
        p_vision.join()
        p_ai.join()
        print("[Orchestrator] Exit complete.")
