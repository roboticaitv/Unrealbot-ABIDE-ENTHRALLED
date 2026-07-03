import multiprocessing
import time
import sys
import os

# Add vision and models folders to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'vision'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'models'))

from vision.vision_manager import VisionManager
from vision.serial_bridge import SerialBridge
from models.abide.abide_run import AbideRun
from models.abide.inferencia import Inferencia

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
    ai = AbideRun(models_dir=os.path.join(os.path.dirname(__file__), "models"))
    inferencia = Inferencia()
    
    # Determine the serial port (typically /dev/ttyACM0 or /dev/ttyUSB0 on Pi)
    # Using a common default for ESP32 on Pi
    port = '/dev/ttyACM0' if os.path.exists('/dev/ttyACM0') else '/dev/ttyUSB0'
    
    # Start serial bridge
    # Provide fallback if port is not available (like running on PC)
    try:
        with SerialBridge(port=port, baudrate=1000000) as bridge:
            print(f"[AI Process] Serial bridge connected on {port}.")
            _run_ai_loop(ai, inferencia, bridge, state_dict, ready_event)
    except Exception as e:
        print(f"[AI Process] Serial port {port} unavailable: {e}. Running AI purely locally.")
        _run_ai_loop(ai, inferencia, None, state_dict, ready_event)

def _run_ai_loop(ai, inferencia, bridge, state_dict, ready_event):
    frames_processed = 0
    start_time = time.time()
    
    try:
        while True:
            # Wait for vision to provide new frame (sleeps until signaled to save CPU)
            if ready_event.wait(timeout=0.1):
                ready_event.clear()
                
                # 1. Get snapshot of raw vision data
                raw_vision = dict(state_dict)
                
                # 2. Get telemetry
                telemetry = bridge.get_telemetry() if bridge else None
                
                # 3. Process raw data into ABIDE schema
                current_state = inferencia.update(
                    detections=raw_vision.get("detections", {"ball": None, "blue_goal": None, "yellow_goal": None}),
                    threats=raw_vision.get("threats", []),
                    telemetry=telemetry
                )
                
                # 4. Infer motor commands
                action, _ = ai.infer(current_state)
                
                # --- SISTEMA DE FRENADO DE EMERGENCIA (LÍNEAS BLANCAS) ---
                # Las cámaras miden 624px de alto. Si una línea está muy abajo (ej. y > 500), 
                # significa que la línea está literalmente tocando el frente/atrás del robot.
                lines = raw_vision.get("lines", [])
                too_close_front = any(line["cam_id"] == 0 and line["center"][1] > 500 for line in lines)
                too_close_back  = any(line["cam_id"] == 1 and line["center"][1] > 500 for line in lines)
                
                # Si queremos avanzar (vx > 0) y hay línea al frente, o retroceder (vx < 0) y hay línea atrás:
                #if (action["vx"] > 0 and too_close_front) or (action["vx"] < 0 and too_close_back):
                #    print("[Seguridad] ¡LÍNEA DETECTADA! Motores detenidos para no salir del campo.")
                #    action["vx"] = 0.0
                #    action["vy"] = 0.0  # También detenemos el movimiento lateral por seguridad
                
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
