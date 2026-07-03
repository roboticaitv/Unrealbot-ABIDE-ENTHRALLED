import cv2
import time
import numpy as np
import os
import sys
import argparse
from color_tracking import ColorTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker
from ai_engine import AIEngine
from vision_config import config
from hud import draw_detections

# Ruta de videos de prueba
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.normpath(os.path.join(THIS_DIR, "..", "videos_prueba"))

def find_video_pairs(folder):
    """Agrupa los videos por timestamp -> [(cam0_path, cam1_path), ...]."""
    if not os.path.exists(folder):
        return []
    files = sorted(os.listdir(folder))
    cam0 = [f for f in files if f.startswith("cam0") and f.endswith(".avi")]
    cam1 = [f for f in files if f.startswith("cam1") and f.endswith(".avi")]

    pairs = []
    for c0 in cam0:
        suffix = c0[4:]
        match = "cam1" + suffix
        if match in cam1:
            pairs.append((
                os.path.join(folder, c0),
                os.path.join(folder, match),
            ))
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Dual Camera Model Tester")
    parser.add_argument("pair_idx", type=int, nargs="?", default=1, help="Index of the video pair to play (default: 1)")
    parser.add_argument("--blind-back", action="store_true", help="Ciega la camara trasera (CAM0) haciendola totalmente negra")
    args = parser.parse_args()

    pairs = find_video_pairs(VIDEO_DIR)
    if not pairs:
        print(f"Error: No se encontraron videos en {VIDEO_DIR}")
        return

    pair_idx = args.pair_idx % len(pairs)
    cam0_path, cam1_path = pairs[pair_idx]
    print(f"\n[INFO] Reproduciendo par {pair_idx}:")
    print(f"       CAM0: {os.path.basename(cam0_path)}")
    print(f"       CAM1: {os.path.basename(cam1_path)}")
    if args.blind_back:
        print("       [WARNING] CAMARA TRASERA (CAM0) CEGADA")

    cap0 = cv2.VideoCapture(cam0_path)
    cap1 = cv2.VideoCapture(cam1_path)

    tracker = ColorTracker()
    
    # State Tracker
    st0 = StateTracker()
    st1 = StateTracker()
    
    # AI Engine
    models_dir = os.path.join(os.path.dirname(THIS_DIR), "ONNX_models")
    ai_engine = AIEngine(models_dir=models_dir)

    cv2.namedWindow("CAM0 — Modelo Completo", cv2.WINDOW_NORMAL)
    cv2.namedWindow("CAM1 — Modelo Completo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("CAM0 — Modelo Completo", 820, 616)
    cv2.resizeWindow("CAM1 — Modelo Completo", 820, 616)

    paused = False
    frame_n = 0
    show_hud = True

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = not paused
            print("[INFO] Pausado" if paused else "[INFO] Reanudado")
        elif key == ord('h'):
            show_hud = not show_hud

        if paused:
            time.sleep(0.03)
            continue

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print("[INFO] Fin del video - reiniciando...")
            cap0.set(cv2.CAP_PROP_POS_FRAMES, 0)
            cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_n = 0
            continue

        frame_n += 1
        
        # Redimensionar a escala estándar de procesamiento
        frame0 = cv2.resize(frame0, (820, 616))
        frame1 = cv2.resize(frame1, (820, 616))

        if args.blind_back:
            frame0 = np.zeros_like(frame0)

        # ─── PROCESAR CAM1 (Frontal) ───
        t0_start = time.perf_counter()
        results1 = tracker.process_yuv_frame(frame1, cam_id=1)
        threats1 = filter_threats(results1["unknowns"], [], safe_radius=40)
        state1 = st1.update(results1, threats1)
        action1, debug1 = ai_engine.infer(state1)
        t1_total = (time.perf_counter() - t0_start) * 1000.0

        # Optimización de cruce: si CAM1 ya encontró balón/porterías/enemigos, podemos saltarlos en CAM0
        skip_ball = results1["ball"] is not None
        skip_blue = results1["blue_goal"] is not None
        skip_yellow = results1["yellow_goal"] is not None
        skip_enemies = len(results1["unknowns"]) > 0

        # ─── PROCESAR CAM0 (Trasera) ───
        t0_start_cam0 = time.perf_counter()
        results0 = tracker.process_yuv_frame(
            frame0,
            skip_ball=skip_ball,
            skip_blue_goal=skip_blue,
            skip_yellow_goal=skip_yellow,
            skip_enemies=skip_enemies,
            cam_id=0
        )
        
        # Limit to max 2 enemies across both cameras
        total_unknowns = [(box, 1) for box in results1.get("unknowns", [])] + [(box, 0) for box in results0.get("unknowns", [])]
        total_unknowns.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
        top_unknowns = total_unknowns[:2]
        results1["unknowns"] = [box for box, cam in top_unknowns if cam == 1]
        results0["unknowns"] = [box for box, cam in top_unknowns if cam == 0]
        
        threats0 = filter_threats(results0["unknowns"], [], safe_radius=40)
        state0 = st0.update(results0, threats0)
        action0, debug0 = ai_engine.infer(state0)
        t0_total = (time.perf_counter() - t0_start_cam0) * 1000.0

        # Imprimir telemetría del AI loop a la consola cada 30 cuadros
        if frame_n % 30 == 0:
            print(f"\n--- FRAME {frame_n} AI REPORT (CAM1) ---")
            if debug1:
                print("  Ego Speed Norm:", round(state1.get("ego_speed_norm", 0.0), 3))
                print("  Ball Angle Deg:", round(state1.get("ball_angle_deg", 0.0), 1))
                print("  Action (vx, vy, omega):", 
                      round(action1.get("vx", 0.0), 2), 
                      round(action1.get("vy", 0.0), 2), 
                      round(action1.get("omega", 0.0), 2))
                print("  NET_T Emb Prefix:", [round(x, 2) for x in debug1.get("NET_T_emb", [])[:4]])

        # ─── DIBUJAR HUD overlays ───
        vis0 = draw_detections(
            frame0.copy(), 
            results0, 
            threats0, 
            state0, 
            action=action0, 
            cam_label="CAM0 (BACK)", 
            latency_ms=t0_total, 
            show_hud=show_hud
        )
        vis1 = draw_detections(
            frame1.copy(), 
            results1, 
            threats1, 
            state1, 
            action=action1, 
            cam_label="CAM1 (FRONT)", 
            latency_ms=t1_total, 
            show_hud=show_hud
        )

        # Agregar número de frame y FPS visual
        for vis in (vis0, vis1):
            cv2.putText(vis, f"Frame: {frame_n}", (10, vis.shape[0] - 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("CAM0 — Modelo Completo", vis0)
        cv2.imshow("CAM1 — Modelo Completo", vis1)

    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()
    print("[INFO] Video tester dual finalizado.")

if __name__ == "__main__":
    main()
