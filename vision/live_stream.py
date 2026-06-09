import cv2
import numpy as np
import threading
import time
import json
import os
import argparse
import math as m
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from dual_camera import DualCamera
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker
from ai_engine import AIEngine
from vision_config import config

# Globals
SHOW_HUD = True
cameras = DualCamera(resolution=(832, 624), framerate=83, format="YUV420")
tracker = ColorTracker()
aruco0 = IntermittentArucoTracker(interval=10)
aruco1 = IntermittentArucoTracker(interval=10)
st0 = StateTracker()
st1 = StateTracker()
ai_engine = AIEngine(models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "ONNX_models"))

# ── Camera calibration ──
FOCAL_LENGTH_EQ = config["camera"]["focal_length_eq"]
IMAGE_CX = config["camera"]["image_cx"]
IMAGE_CY = config["camera"]["image_cy"]

# Known physical sizes (millimeters)
BALL_DIAMETER_MM = config["physics"]["ball_diameter_mm"]
ARUCO_SIZE_MM = config["physics"]["aruco_size_mm"]

latest_jpeg = None
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
        
        # Convert to BGR for display
        bgr0 = yuv420_to_bgr(f0_raw)
        bgr1 = yuv420_to_bgr(f1_raw)
        
        # Run color detections
        det0 = tracker.process_yuv_frame(f0_raw)
        det1 = tracker.process_yuv_frame(f1_raw)
        
        # Run ArUco on both cameras, physically sliced at the horizon to save ~50% CPU!
        gray0 = get_grayscale(f0_raw)
        gray1 = get_grayscale(f1_raw)
        
        hy0 = int(det0.get("horizon_y", 0))
        hy1 = int(det1.get("horizon_y", 0))
        
        aruco_det0 = aruco0.process_frame(gray0[hy0:, :], y_offset=hy0)
        aruco_det1 = aruco1.process_frame(gray1[hy1:, :], y_offset=hy1)
        
        # Run enemy detection
        ally_boxes0 = []
        for marker_id, data in aruco_det0.items():
            cx, cy, _ = data
            ally_boxes0.append((cx - 40, cy - 40, 80, 80))
            
        ally_boxes1 = []
        for marker_id, data in aruco_det1.items():
            cx, cy, _ = data
            ally_boxes1.append((cx - 40, cy - 40, 80, 80))
            
        threats0 = filter_threats(det0["unknowns"], ally_boxes0, safe_radius=40)
        threats1 = filter_threats(det1["unknowns"], ally_boxes1, safe_radius=40)
        
        # Calculate State Physics
        state0 = st0.update(det0, threats0, aruco_det0)
        state1 = st1.update(det1, threats1, aruco_det1)
        
        # AI Engine Inference
        action, debug = ai_engine.infer(state0)
        
        # Store data for the /dashboard endpoint (replaces console spam)
        with data_lock:
            latest_data["state"] = state0
            latest_data["action"] = action
            latest_data["debug"] = debug if debug else {}
            
        # Draw overlays with HUD
        bgr0 = draw_detections(bgr0, det0, aruco_det0, threats0, state0, "CAM 0", show_hud=SHOW_HUD)
        bgr1 = draw_detections(bgr1, det1, aruco_det1, threats1, state1, "CAM 1", show_hud=SHOW_HUD)
        
        # Combine side-by-side for stream
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
            <body style="background:#111; margin:0; display:flex; justify-content:center; align-items:center; height:100vh;">
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
