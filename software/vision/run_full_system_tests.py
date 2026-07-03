import os
import sys
import time
import cv2
import numpy as np

# Add vision to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS_DIR)

from color_tracking import ColorTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker
from ai_engine import AIEngine
from vision_config import config

VIDEO_DIR = os.path.join(THIS_DIR, "..", "videos_prueba")

def find_video_pairs(folder):
    """Group the videos by timestamp -> [(cam0_path, cam1_path), ...]."""
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

def run_tests():
    print("="*60)
    print("RUNNING HEADLESS FULL SYSTEM INTEGRATION TESTS")
    print("="*60)
    
    # 1. Verify models directory and files
    models_dir = os.path.join(os.path.dirname(THIS_DIR), "ONNX_models")
    print(f"Models directory: {models_dir}")
    
    # 2. Find video pairs
    pairs = find_video_pairs(VIDEO_DIR)
    if not pairs:
        print(f"[ERROR] No video pairs found in {VIDEO_DIR}!")
        sys.exit(1)
        
    print(f"Found {len(pairs)} test video pairs:")
    for idx, (p0, p1) in enumerate(pairs):
        print(f"  [{idx}]: CAM0: {os.path.basename(p0)} | CAM1: {os.path.basename(p1)}")
        
    # 3. Initialize components
    print("\nInitializing pipeline components...")
    try:
        tracker = ColorTracker()
        print("  [OK] ColorTracker initialized.")
        
        st0 = StateTracker()
        st1 = StateTracker()
        print("  [OK] StateTrackers initialized.")
        
        ai_engine = AIEngine(models_dir=models_dir)
        if ai_engine.enabled:
            print("  [OK] AIEngine initialized with ONNX models.")
        else:
            print("  [WARNING] AIEngine initialized in dummy mode (missing models or libraries).")
            
    except Exception as e:
        print(f"  [ERROR] Component initialization failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    # 4. Run through video frames
    frames_to_test = 100
    print(f"\nProcessing {frames_to_test} frames per video pair...")
    
    all_success = True
    
    for idx, (p0, p1) in enumerate(pairs):
        print(f"\n--- Testing Video Pair {idx} ---")
        cap0 = cv2.VideoCapture(p0)
        cap1 = cv2.VideoCapture(p1)
        
        if not cap0.isOpened() or not cap1.isOpened():
            print(f"  [ERROR] Could not open video files for pair {idx}!")
            all_success = False
            continue
            
        frame_n = 0
        cam0_latencies = []
        cam1_latencies = []
        ai_latencies = []
        
        try:
            while frame_n < frames_to_test:
                ret0, frame0 = cap0.read()
                ret1, frame1 = cap1.read()
                
                if not ret0 or not ret1:
                    print(f"  [INFO] Reached end of video early at frame {frame_n}")
                    break
                    
                frame_n += 1
                
                # Standard resolution scaling
                frame0 = cv2.resize(frame0, (820, 616))
                frame1 = cv2.resize(frame1, (820, 616))
                
                # --- PROCESS CAM1 (Front) ---
                t1_start = time.perf_counter()
                results1 = tracker.process_yuv_frame(frame1, cam_id=1)
                
                threats1 = filter_threats(results1["unknowns"], [], safe_radius=40)
                state1 = st1.update(results1, threats1)
                
                # AI Inference (CAM1)
                t_ai_start = time.perf_counter()
                action1, debug1 = ai_engine.infer(state1)
                ai_latencies.append((time.perf_counter() - t_ai_start) * 1000.0)
                
                cam1_latencies.append((time.perf_counter() - t1_start) * 1000.0)
                
                # Cross-camera skip optimizations
                skip_ball = results1["ball"] is not None
                skip_blue = results1["blue_goal"] is not None
                skip_yellow = results1["yellow_goal"] is not None
                skip_enemies = len(results1["unknowns"]) > 0
                
                # --- PROCESS CAM0 (Back) ---
                t0_start = time.perf_counter()
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
                
                # AI Inference (CAM0)
                action0, debug0 = ai_engine.infer(state0)
                
                cam0_latencies.append((time.perf_counter() - t0_start) * 1000.0)
                
            print(f"  [OK] Successfully processed {frame_n} frames.")
            print(f"  CAM1 Latency (Avg): {np.mean(cam1_latencies):.2f} ms | Max: {np.max(cam1_latencies):.2f} ms")
            print(f"  CAM0 Latency (Avg): {np.mean(cam0_latencies):.2f} ms | Max: {np.max(cam0_latencies):.2f} ms")
            print(f"  AI Inference Latency (Avg): {np.mean(ai_latencies):.2f} ms")
            
        except Exception as e:
            print(f"  [ERROR] Exception raised during video pair {idx} processing: {e}")
            import traceback
            traceback.print_exc()
            all_success = False
        finally:
            cap0.release()
            cap1.release()
            
    print("\n" + "="*60)
    if all_success:
        print("RESULT: ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
        print("="*60)
        sys.exit(0)
    else:
        print("RESULT: INTEGRATION TESTS FAILED!")
        print("="*60)
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
