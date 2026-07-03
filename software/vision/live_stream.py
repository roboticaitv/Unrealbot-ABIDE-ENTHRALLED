import cv2
import numpy as np
import threading
import time
import json
import os
import sys
import argparse
import math as m
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from dual_camera import DualCamera
from color_tracking import ColorTracker
from hitbox_math import filter_threats

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models'))
from abide.inferencia import Inferencia
from abide.abide_run import AbideRun
from vision_config import config

# Globals
SHOW_HUD = True
cameras = DualCamera(resolution=(832, 624), framerate=83, format="YUV420")
tracker = ColorTracker()
st0 = Inferencia()
st1 = Inferencia()
ai_engine = AbideRun(models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models"))

# ── Camera calibration ──
FOCAL_LENGTH_EQ = config["camera"]["focal_length_eq"]
IMAGE_CX = config["camera"]["image_cx"]
IMAGE_CY = config["camera"]["image_cy"]

# Known physical sizes (millimeters)
BALL_DIAMETER_MM = config["physics"]["ball_diameter_mm"]

latest_jpeg = None
allied_goal_color = None
jpeg_lock = threading.Lock()
latest_data = {"state": {}, "action": {}, "debug": {}}
data_lock = threading.Lock()

from hud import draw_detections

def yuv420_to_bgr(frame):
    """Convert a YUV420 planar frame to BGR for display."""
    if len(frame.shape) == 2:
        return cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_I420)
    return frame  # Already BGR

def get_grayscale(frame):
    """Extract grayscale from either YUV420 planar or BGR frame."""
    if len(frame.shape) == 2:
        # YUV420: Y plane is the top 2/3 of the array
        h = frame.shape[0] * 2 // 3
        return frame[:h, :]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def processing_loop():
    """Grabs frames, runs detection, encodes JPEG for the stream."""
    global latest_jpeg
    cameras.start()
    time.sleep(1.0)
    
    while True:
        f1_raw, f0_raw = cameras.get_frames() # SWAPPED: f0 is now front, f1 is back
        if f0_raw is None or f1_raw is None:
            time.sleep(0.01)
            continue
        
        t_start = time.time()
        
        # Convert to BGR for display
        bgr0 = yuv420_to_bgr(f0_raw)
        bgr1 = yuv420_to_bgr(f1_raw)
        
        # Run color detections
        det0 = tracker.process_yuv_frame(f0_raw, cam_id=0)
        det1 = tracker.process_yuv_frame(f1_raw, cam_id=1)
        
        # Limit to max 2 enemies across both cameras
        total_unknowns = [(box, 1) for box in det1.get("unknowns", [])] + [(box, 0) for box in det0.get("unknowns", [])]
        total_unknowns.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
        top_unknowns = total_unknowns[:2]
        det1["unknowns"] = [box for box, cam in top_unknowns if cam == 1]
        det0["unknowns"] = [box for box, cam in top_unknowns if cam == 0]
        
        # Dynamic goal assignment from CAM 1
        global allied_goal_color
        if allied_goal_color is None:
            if det1.get("blue_goal") is not None:
                allied_goal_color = "blue"
                config["physics"]["team_color"] = "yellow"
                print("Dynamic Goal Assignment: Allied = Blue, Enemy = Yellow")
            elif det1.get("yellow_goal") is not None:
                allied_goal_color = "yellow"
                config["physics"]["team_color"] = "blue"
                print("Dynamic Goal Assignment: Allied = Yellow, Enemy = Blue")
        
        # Fast Algebraic Hitbox Math
        threats0 = filter_threats(det0["unknowns"], [], safe_radius=40)
        threats1 = filter_threats(det1["unknowns"], [], safe_radius=40)
        
        # State Merging - Independent Object Fusion
        final_det = {
            "ball": det0["ball"] if det0["ball"] else det1["ball"],
            "ball_cam": 0 if det0["ball"] else (1 if det1["ball"] else 0),
            
            "blue_goal": det0["blue_goal"] if det0["blue_goal"] else det1["blue_goal"],
            "blue_goal_cam": 0 if det0["blue_goal"] else (1 if det1["blue_goal"] else 0),
            
            "yellow_goal": det0["yellow_goal"] if det0["yellow_goal"] else det1["yellow_goal"],
            "yellow_goal_cam": 0 if det0["yellow_goal"] else (1 if det1["yellow_goal"] else 0),
        }
        
        final_threats = [{"bbox": t, "cam_id": 0} for t in threats0] + [{"bbox": t, "cam_id": 1} for t in threats1]
        
        # Calculate State Physics (Fused 360)
        master_state = st0.update(final_det, final_threats)
        
        # AI Engine Inference
        action, debug = ai_engine.infer(master_state)
        
        # Store data for the /dashboard endpoint (replaces console spam)
        with data_lock:
            latest_data["state"] = master_state
            latest_data["action"] = action
            latest_data["debug"] = debug if debug else {}
            
        # Calculate latency
        elapsed_ms = (time.time() - t_start) * 1000.0
            
        # Draw overlays with HUD (we use the master state for both to show the 360 tracking)
        bgr0 = draw_detections(bgr0, det0, threats0, master_state, action=action, cam_label="CAM 0 (Front)", latency_ms=elapsed_ms, show_hud=SHOW_HUD)
        bgr1 = draw_detections(bgr1, det1, threats1, master_state, action=None, cam_label="CAM 1 (Rear)", latency_ms=elapsed_ms, show_hud=SHOW_HUD)
        
        # Combine side-by-side for stream (Front on the right, Rear on the left if you prefer, or keep 0 then 1)
        # Let's keep it CAM 0 then CAM 1
        combined = cv2.hconcat([bgr0, bgr1])
        
        # Encode to JPEG
        _, jpeg = cv2.imencode('.jpg', combined, [cv2.IMWRITE_JPEG_QUALITY, 70])
        
        with jpeg_lock:
            latest_jpeg = jpeg.tobytes()

class MJPEGHandler(BaseHTTPRequestHandler):
    def _send_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')

    def do_OPTIONS(self):
        self.send_response(200)
        self._send_cors_headers()
        self.end_headers()

    def do_POST(self):
        if self.path == '/api/config':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            try:
                new_config = json.loads(post_data)
                
                # Update tracker bounds
                for color_name, bounds in [('orange', tracker.orange_bounds), 
                                         ('blue', tracker.blue_bounds),
                                         ('yellow', tracker.yellow_bounds),
                                         ('green', tracker.green_bounds)]:
                    if color_name in new_config:
                        for k, v in new_config[color_name].items():
                            bounds[k] = int(v)
                            
                # Update StateTracker params
                if 'camera' in new_config:
                    f = float(new_config['camera']['focal_length_eq'])
                    k = float(new_config['camera']['edge_correction'])
                    st0.FOCAL_LENGTH_EQ = f
                    st1.FOCAL_LENGTH_EQ = f
                    st0.EDGE_CORRECTION = k
                    st1.EDGE_CORRECTION = k
                    
                self.send_response(200)
                self._send_cors_headers()
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"status":"error"}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self):
        if self.path == '/api/config':
            config = {
                'orange': tracker.orange_bounds,
                'blue': tracker.blue_bounds,
                'yellow': tracker.yellow_bounds,
                'green': tracker.green_bounds,
                'camera': {
                    'focal_length_eq': st0.FOCAL_LENGTH_EQ,
                    'edge_correction': st0.EDGE_CORRECTION
                }
            }
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(config).encode())
        elif self.path == '/data':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._send_cors_headers()
            self.end_headers()
            with data_lock:
                self.wfile.write(json.dumps(latest_data).encode())
                
        elif self.path == '/dashboard':
            try:
                with open(os.path.join(os.path.dirname(__file__), 'dashboard.html'), 'rb') as f:
                    html = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                
        elif self.path == '/calibrate':
            try:
                with open(os.path.join(os.path.dirname(__file__), 'calibrate.html'), 'rb') as f:
                    html = f.read()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
                
        elif self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(b'''
            <html><head><title>Unrealbot Vision</title></head>
            <body style="background:#111; margin:0; display:flex; flex-direction:column; justify-content:center; align-items:center; height:100vh; overflow:hidden;">
                <div style="position: absolute; top: 20px; right: 20px; z-index: 1000;">
                    <a href="/dashboard" style="background: #38bdf8; color: #0f172a; padding: 12px 24px; text-decoration: none; font-family: sans-serif; font-weight: bold; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); transition: all 0.2s;">Abrir Panel de Datos &rarr;</a>
                </div>
                <img src="/stream" style="max-width:100%; max-height:100vh;">
            </body></html>
            ''')
        elif self.path == '/stream':
            self.send_response(200)
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            
            while True:
                with jpeg_lock:
                    if latest_jpeg is None:
                        time.sleep(0.01)
                        continue
                    frame_data = latest_jpeg
                
                try:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
                    self.wfile.write(frame_data)
                    self.wfile.write(b'\r\n')
                    time.sleep(0.033)  # ~30 FPS stream rate (saves bandwidth)
                except BrokenPipeError:
                    break
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP request logs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unrealbot Live Stream")
    parser.add_argument('--no-hud', action='store_true', help="Disable the HUD overlay to reduce stream bandwidth and save CPU")
    args = parser.parse_args()
    
    if args.no_hud:
        SHOW_HUD = False

    # Start the processing thread
    proc_thread = threading.Thread(target=processing_loop, daemon=True)
    proc_thread.start()
    
    # Start the HTTP server
    port = 8080
    server = ThreadingHTTPServer(('0.0.0.0', port), MJPEGHandler)
    print(f"\n  Live stream ready at: http://192.168.1.90:{port}\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        cameras.stop()
