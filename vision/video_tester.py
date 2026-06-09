import cv2
import time
import numpy as np
import math as m
import argparse
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker
from ai_engine import AIEngine
from vision_config import config
import threading
import queue
from hud import draw_detections



class VideoThread:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.q = queue.Queue(maxsize=3)
        self.stopped = False
        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()

    def update(self):
        while not self.stopped and self.cap.isOpened():
            if not self.q.full():
                ret, frame = self.cap.read()
                if not ret:
                    self.stop()
                    return
                self.q.put(frame)
            else:
                time.sleep(0.01)

    def read(self):
        return self.q.get()
        
    def empty(self):
        return self.q.empty()

    def isOpened(self):
        return self.cap.isOpened() or not self.q.empty()

    def release(self):
        self.stop()

    def stop(self):
        self.stopped = True
        self.cap.release()

class VideoTester:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = VideoThread(video_path)
        self.color_tracker = ColorTracker()
        self.aruco_tracker = IntermittentArucoTracker(interval=5)
        self.state_tracker = StateTracker()
        import os
        self.ai_engine = AIEngine(models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "ONNX_models"))
        self.paused = False
        
    def run(self):
        if not self.cap.isOpened():
            print(f"Error opening video file: {self.video_path}")
            return

        print("Press 'q' to quit, 'space' to pause, 'h' to hide/show HUD.")
        
        show_hud = True
        
        while self.cap.isOpened():
            if self.cap.empty() and not self.cap.isOpened():
                print("End of video stream.")
                break
            frame = self.cap.read()
                
            frame = cv2.resize(frame, (820, 616))
            
            # Process frame
            start_time = time.perf_counter()
            
            color_results = self.color_tracker.process_yuv_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # The True "Diminishing" Upgrade: Physical Array Slicing for ArUco
            hy = int(color_results.get("horizon_y", 0))
            aruco_positions = self.aruco_tracker.process_frame(gray[hy:, :], y_offset=hy)
            ally_boxes = []
            for marker_id, data in aruco_positions.items():
                cx, cy, _ = data
                ally_boxes.append((cx - 40, cy - 40, 80, 80))
            threats = filter_threats(color_results["unknowns"], ally_boxes, safe_radius=40)
            
            # Compute physics state
            state = self.state_tracker.update(color_results, threats, aruco_positions)
            
            # AI Inference
            action, debug = self.ai_engine.infer(state)
            
            # Print embeddings nicely to console if not in dummy mode
            if debug:
                print("\n" + "="*40)
                print(f"FRAME TIME: {time.perf_counter():.2f}")
                print("--- RAW STATE (17 values) ---")
                for k, v in debug["raw_state"].items():
                    print(f"  {k}: {v:.2f}")
                print("--- EMBEDDINGS ---")
                print(f"  NET_A: {[round(x, 2) for x in debug['NET_A_emb']]}")
                print(f"  NET_B: {[round(x, 2) for x in debug['NET_B_emb']]}")
                print(f"  NET_C: {[round(x, 2) for x in debug['NET_C_emb']]}")
                print(f"  NET_T: {[round(x, 2) for x in debug['NET_T_emb']]}")
                print("--- FINAL ACTION ---")
                for k, v in action.items():
                    print(f"  {k}: {v:.2f}")
                print("="*40)
            
            end_time = time.perf_counter()
            process_ms = (end_time - start_time) * 1000
            
            # 3. Draw overlays with HUD (if enabled)
            display_frame = draw_detections(
                frame.copy(), 
                color_results, 
                aruco_positions, 
                threats,
                state,
                action=action,
                cam_label="VIDEO TEST",
                latency_ms=process_ms,
                show_hud=show_hud
            )
                
            # Show output
            cv2.imshow("Vision Test", display_frame)
            
            key = cv2.waitKey(max(1, 33 - int(process_ms))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                # Pause until another key is pressed
                while True:
                    pause_key = cv2.waitKey(0) & 0xFF
                    if pause_key == ord(' '):
                        break
                    elif pause_key == ord('h'):
                        show_hud = not show_hud
                        # Redraw from scratch
                        disp = draw_detections(
                            frame.copy(), 
                            color_results, 
                            aruco_positions, 
                            threats,
                            state,
                            action=action,
                            cam_label="VIDEO TEST",
                            latency_ms=process_ms,
                            show_hud=show_hud
                        )
                        cv2.imshow("Vision Test", disp)
                    elif pause_key == ord('q'):
                        self.cap.release()
                        cv2.destroyAllWindows()
                        return
            elif key == ord('h'):
                show_hud = not show_hud

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        tester = VideoTester(video_file)
        tester.run()
    else:
        print("Usage: python video_tester.py <path_to_video.mp4>")
