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

# ── Camera calibration for IMX219 + 200° fisheye lens at 1640x1232 ──
FOCAL_LENGTH_EQ = 470
IMAGE_CX = 820   # Optical center X (half of 1640)
IMAGE_CY = 616   # Optical center Y (half of 1232)

# Known physical sizes (millimeters)
BALL_DIAMETER_MM = 43      
ARUCO_SIZE_MM = 50         

def estimate_distance_mm(known_size_mm, apparent_size_px, obj_cx=IMAGE_CX, obj_cy=IMAGE_CY):
    """Fisheye equidistant model distance estimation."""
    if apparent_size_px <= 0:
        return -1
    angular_size = apparent_size_px / FOCAL_LENGTH_EQ
    half_angle = angular_size / 2.0
    if half_angle >= m.pi / 2:
        return -1
    return known_size_mm / (2.0 * m.tan(half_angle))



def draw_detections(bgr_frame, detections, aruco_positions, threats, state, action=None, cam_label="VIDEO", latency_ms=0):
    """Draw bounding boxes, labels, distances, threats, and a HUD data panel."""
    hud_lines = [cam_label, f"Latency: {latency_ms:.1f}ms"]
    
    # ── Draw Horizon ──
    if "horizon_y" in detections:
        hy = detections["horizon_y"]
        cv2.line(bgr_frame, (0, hy), (bgr_frame.shape[1], hy), (255, 0, 255), 2)
        cv2.putText(bgr_frame, "HORIZON", (10, hy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # ── Ball ──
    if detections["ball"]:
        x, y, w, h = detections["ball"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 165, 255), 2)
        hud_lines.append(f"Ball Dist: {state['ball_distance_norm']:.2f}")
        hud_lines.append(f"Ball Spd: {state['ball_speed_norm']:.2f}")
        hud_lines.append(f"Shot Opp: {state['shot_opportunity_ego']:.2f}")
    else:
        hud_lines.append("Ball: ---")
        
    # ── Goals ──
    if detections["blue_goal"]:
        x, y, w, h = detections["blue_goal"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
    if detections["yellow_goal"]:
        x, y, w, h = detections["yellow_goal"]["bbox"]
        cv2.rectangle(bgr_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
    else:
        hud_lines.append("Yellow Goal: ---")
        
    # ── ArUco ──
    if aruco_positions:
        for marker_id, data in aruco_positions.items():
            cx, cy, apparent_size = data
            cv2.circle(bgr_frame, (int(cx), int(cy)), int(apparent_size//2), (0, 255, 0), 2)
    
    # ── Enemies (Threats) ──
    if len(threats) > 0:
        hud_lines.append(f"E1 Dist: {state['enemy1_distance_norm']:.2f}")
        hud_lines.append(f"E1 Spd: {state['enemy1_velocity_norm']:.2f}")
        for i, (x, y, w, h) in enumerate(threats):
            cv2.rectangle(bgr_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(bgr_frame, f"ENEMY", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
    # ── AI Actions ──
    if action:
        hud_lines.append(f"CMD Vx: {action['vx']:.2f}")
        hud_lines.append(f"CMD Vy: {action['vy']:.2f}")
        hud_lines.append(f"CMD W:  {action['omega']:.2f}")
        hud_lines.append(f"CMD Kick: {action['kick']:.2f}")
    
    # ── Draw HUD panel ──
    line_h = 36
    panel_h = line_h * len(hud_lines) + 16
    overlay = bgr_frame.copy()
    cv2.rectangle(overlay, (0, 0), (bgr_frame.shape[1], panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, bgr_frame, 0.4, 0, bgr_frame)
    
    for i, line in enumerate(hud_lines):
        color = (0, 255, 0) if i == 0 else ((0, 255, 255) if i == 1 else (255, 255, 255))
        cv2.putText(bgr_frame, line, (10, 30 + i * line_h), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    
    return bgr_frame

class VideoTester:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
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

        print("Press 'q' to quit, 'space' to pause.")
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream.")
                break
                
            frame = cv2.resize(frame, (820, 616))
                
            # Process frame
            start_time = time.perf_counter()
            
            color_results = self.color_tracker.process_yuv_frame(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_positions = self.aruco_tracker.process_frame(gray)
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
            
            # 3. Draw overlays with HUD
            display_frame = draw_detections(
                frame, 
                color_results, 
                aruco_positions, 
                threats,
                state,
                action=action,
                cam_label="VIDEO TEST",
                latency_ms=process_ms
            )
            # Show output
            cv2.imshow("Vision Test", display_frame)
            
            key = cv2.waitKey(max(1, 33 - int(process_ms))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0)

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
