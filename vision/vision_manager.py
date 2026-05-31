import time
import cv2
from dual_camera import DualCamera
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats

class VisionManager:
    def __init__(self):
        # Initialize the hardware
        self.cameras = DualCamera(resolution=(1640, 1232), framerate=83, format="YUV420")
        
        # Initialize the trackers
        self.color_tracker = ColorTracker()
        self.aruco_tracker = IntermittentArucoTracker(interval=10)
        
        self.running = False

    def start(self):
        self.cameras.start()
        self.running = True
        
        # Give cameras time to spin up
        time.sleep(1.0)
        print("Vision Pipeline Started and Ready.")

    def stop(self):
        self.running = False
        self.cameras.stop()
        print("Vision Pipeline Stopped.")

    def get_world_state(self):
        """
        Pulls the latest frames, runs the lightweight tracking algorithms, 
        and returns a semantic dictionary of the world state to be fed to ABIDE.
        """
        f0, f1 = self.cameras.get_frames()
        
        if f0 is None or f1 is None:
            return None
            
        state = {
            "ball": None,
            "blue_goal": None,
            "yellow_goal": None,
            "allies": {},
            "threats": []
        }

        # 1. High Speed Color Tracking (Using YUV directly)
        # Assuming f0 is the main forward-facing camera for the ball
        color_results = self.color_tracker.process_yuv_frame(f0)
        state["ball"] = color_results["ball"]
        state["blue_goal"] = color_results["blue_goal"]
        state["yellow_goal"] = color_results["yellow_goal"]

        # 2. Intermittent ArUco Tracking
        # Convert Y channel of YUV to grayscale for ArUco
        # If it's a BGR frame from testing, we convert to gray. 
        # For pure YUV420, Y is the first channel block.
        if len(f0.shape) == 3 and f0.shape[2] == 3:
            gray = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
        else:
            # Assuming pure Y plane if pulling raw from libcamera
            gray = f0 

        aruco_positions = self.aruco_tracker.process_frame(gray)
        
        # Map known Aruco IDs to allies (e.g., ID 1 is us, ID 2 is our teammate)
        # Store their bounding boxes conceptually for hitbox math (just making 50x50 boxes around centers for now)
        ally_boxes = []
        for marker_id, (cx, cy) in aruco_positions.items():
            state["allies"][marker_id] = (cx, cy)
            ally_boxes.append((cx - 25, cy - 25, 50, 50))

        # 3. Fast Algebraic Hitbox Math
        # Instead of doing mask subtraction, we check generic unknown moving blobs.
        # For demonstration, assume we have a list of `unknown_blobs` from a basic background subtractor
        # or non-colored contour list.
        unknown_blobs = [] # Replace with actual unknown contour boxes if needed
        
        true_threats = filter_threats(unknown_blobs, ally_boxes, safe_radius=30)
        state["threats"] = true_threats

        return state

if __name__ == "__main__":
    vision = VisionManager()
    vision.start()
    
    start_time = time.time()
    frames = 0
    try:
        # Run at max speed
        while time.time() - start_time < 5.0:
            state = vision.get_world_state()
            if state:
                frames += 1
                # print(state)
    except KeyboardInterrupt:
        pass
        
    vision.stop()
    print(f"Processed {frames} logic ticks in 5 seconds ({(frames/5.0):.2f} FPS)")
