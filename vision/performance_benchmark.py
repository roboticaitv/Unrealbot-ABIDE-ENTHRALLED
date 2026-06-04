import time
import numpy as np
import cv2
from dual_camera import DualCamera
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker

def get_grayscale(frame):
    if len(frame.shape) == 2:
        h = frame.shape[0] * 2 // 3
        return frame[:h, :]
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

def run_benchmark(num_frames=1000):
    print(f"Starting Performance Benchmark ({num_frames} frames)...")
    print("Initializing hardware and allocating memory...")
    
    cameras = DualCamera(resolution=(820, 616), framerate=83, format="YUV420")
    tracker = ColorTracker()
    # Force ArUco to run every single frame to measure maximum load
    aruco0 = IntermittentArucoTracker(interval=1) 
    aruco1 = IntermittentArucoTracker(interval=1)
    st0 = StateTracker()
    st1 = StateTracker()
    
    cameras.start()
    time.sleep(1.0) # Warm up sensor
    
    # Pre-allocate history arrays
    io_times = np.zeros(num_frames)
    color_times = np.zeros(num_frames)
    aruco_times = np.zeros(num_frames)
    physics_times = np.zeros(num_frames)
    total_times = np.zeros(num_frames)
    
    valid_frames = 0
    
    print("Running...")
    start_benchmark = time.perf_counter()
    
    while valid_frames < num_frames:
        t_start = time.perf_counter()
        
        # 1. IO Time
        f0, f1 = cameras.get_frames()
        t_io = time.perf_counter()
        
        if f0 is None or f1 is None:
            time.sleep(0.01)
            continue
            
        # 2. Color Tracking Time (both cameras)
        det0 = tracker.process_yuv_frame(f0)
        det1 = tracker.process_yuv_frame(f1)
        t_color = time.perf_counter()
        
        # 3. ArUco Tracking Time (both cameras)
        gray0 = get_grayscale(f0)
        gray1 = get_grayscale(f1)
        aruco_det0 = aruco0.process_frame(gray0)
        aruco_det1 = aruco1.process_frame(gray1)
        t_aruco = time.perf_counter()
        
        # 4. Physics Engine Time (both cameras)
        ally_boxes0 = [(int(cx-20), int(cy-20), 40, 40) for cx, cy, _ in aruco_det0.values()]
        ally_boxes1 = [(int(cx-20), int(cy-20), 40, 40) for cx, cy, _ in aruco_det1.values()]
        
        threats0 = filter_threats(det0["unknowns"], ally_boxes0, safe_radius=40)
        threats1 = filter_threats(det1["unknowns"], ally_boxes1, safe_radius=40)
        
        state0 = st0.update(det0, threats0, aruco_det0)
        state1 = st1.update(det1, threats1, aruco_det1)
        t_physics = time.perf_counter()
        
        # Save metrics
        io_times[valid_frames] = (t_io - t_start) * 1000.0
        color_times[valid_frames] = (t_color - t_io) * 1000.0
        aruco_times[valid_frames] = (t_aruco - t_color) * 1000.0
        physics_times[valid_frames] = (t_physics - t_aruco) * 1000.0
        total_times[valid_frames] = (t_physics - t_start) * 1000.0
        
        valid_frames += 1
        
        if valid_frames % 100 == 0:
            print(f"Processed {valid_frames}/{num_frames} frames...")
            
    end_benchmark = time.perf_counter()
    cameras.stop()
    
    # Process Results
    print("\n" + "="*50)
    print(f" PERFORMANCE REPORT ({num_frames} Frames)")
    print("="*50)
    
    print(f"{'Module (Dual Camera)':<25} | {'Avg Latency':<12} | {'99th %':<12}")
    print("-" * 50)
    
    avg_io = np.mean(io_times)
    p99_io = np.percentile(io_times, 99)
    print(f"{'1. Picamera IO':<25} | {avg_io:>8.2f} ms | {p99_io:>8.2f} ms")
    
    avg_color = np.mean(color_times)
    p99_color = np.percentile(color_times, 99)
    print(f"{'2. Color Tracking (YUV)':<25} | {avg_color:>8.2f} ms | {p99_color:>8.2f} ms")
    
    avg_aruco = np.mean(aruco_times)
    p99_aruco = np.percentile(aruco_times, 99)
    print(f"{'3. ArUco Dictionary':<25} | {avg_aruco:>8.2f} ms | {p99_aruco:>8.2f} ms")
    
    avg_physics = np.mean(physics_times)
    p99_physics = np.percentile(physics_times, 99)
    print(f"{'4. Physics Engine (State)':<25} | {avg_physics:>8.2f} ms | {p99_physics:>8.2f} ms")
    
    print("-" * 50)
    avg_total = np.mean(total_times)
    p99_total = np.percentile(total_times, 99)
    print(f"{'TOTAL LATENCY PER FRAME':<25} | {avg_total:>8.2f} ms | {p99_total:>8.2f} ms")
    print("="*50)
    
    fps_theoretical = 1000.0 / avg_total
    print(f"Theoretical Max Throughput: {fps_theoretical:.1f} FPS")
    print(f"Actual Runtime: {end_benchmark - start_benchmark:.2f} seconds")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_benchmark(1000)
