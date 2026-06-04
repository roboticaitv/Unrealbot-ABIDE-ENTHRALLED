import time
import cv2
from dual_camera import DualCamera
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats
from state_tracker import StateTracker

class VisionManager:
    def __init__(self):
        # Initialize the hardware
        self.cameras = DualCamera(resolution=(832, 624), framerate=83, format="YUV420")
        
        # Initialize the trackers
        self.color_tracker = ColorTracker()
        self.aruco_tracker0 = IntermittentArucoTracker(interval=10)
        self.aruco_tracker1 = IntermittentArucoTracker(interval=10)
        self.state_tracker0 = StateTracker()
        self.state_tracker1 = StateTracker()
        
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
        det0 = self.color_tracker.process_yuv_frame(f0)
        det1 = self.color_tracker.process_yuv_frame(f1)

        # 2. Intermittent ArUco Tracking
        # Convert Y channel of YUV to grayscale for ArUco
        if len(f0.shape) == 3 and f0.shape[2] == 3:
            gray0 = cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)
            gray1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
        else:
            # Extract just the Y plane for grayscale
            h = f0.shape[0] * 2 // 3
            gray0 = f0[:h, :]
            gray1 = f1[:h, :]

        aruco_positions0 = self.aruco_tracker0.process_frame(gray0)
        aruco_positions1 = self.aruco_tracker1.process_frame(gray1)
        
        ally_boxes0 = [(cx - 25, cy - 25, 50, 50) for cx, cy, _ in aruco_positions0.values()]
        ally_boxes1 = [(cx - 25, cy - 25, 50, 50) for cx, cy, _ in aruco_positions1.values()]

        # 3. Fast Algebraic Hitbox Math
        threats0 = filter_threats(det0.get("unknowns", []), ally_boxes0, safe_radius=30)
        threats1 = filter_threats(det1.get("unknowns", []), ally_boxes1, safe_radius=30)

        # 4. State Tracking & Merging
        state0 = self.state_tracker0.update(det0, threats0, aruco_positions0)
        state1 = self.state_tracker1.update(det1, threats1, aruco_positions1)
        
        # Merge logic: Currently prioritizing CAM 0 (Front) if it sees the ball, else CAM 1
        final_state = state0 if det0["ball"] else state1

        return final_state

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
