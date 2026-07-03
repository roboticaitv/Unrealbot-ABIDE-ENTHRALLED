import time
import cv2
from dual_camera import DualCamera
from color_tracking import ColorTracker
from hitbox_math import filter_threats

class VisionManager:
    def __init__(self):
        # Initialize the hardware
        self.cameras = DualCamera(resolution=(832, 624), framerate=83, format="YUV420")
        
        # Initialize the trackers
        self.color_tracker = ColorTracker()
        
        self.running = False
        self.allied_goal_color = None

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

        # 2. Process Color Detections (Pipeline)
        det0 = self.color_tracker.process_yuv_frame(f0, cam_id=0) # Front runs first

        # Camera 1 (Back) skips whatever Camera 0 (Front) already found to save computational cost!
        skip_ball = det0["ball"] is not None
        skip_blue = det0["blue_goal"] is not None
        skip_yellow = det0["yellow_goal"] is not None
        
        det1 = self.color_tracker.process_yuv_frame(
            f1, 
            skip_ball=skip_ball, 
            skip_blue_goal=skip_blue, 
            skip_yellow_goal=skip_yellow,
            cam_id=1
        )
        
        # Limit to max 2 enemies across both cameras
        total_unknowns = [(box, 1) for box in det1.get("unknowns", [])] + [(box, 0) for box in det0.get("unknowns", [])]
        total_unknowns.sort(key=lambda item: item[0][2] * item[0][3], reverse=True)
        top_unknowns = total_unknowns[:2]
        det1["unknowns"] = [box for box, cam in top_unknowns if cam == 1]
        det0["unknowns"] = [box for box, cam in top_unknowns if cam == 0]
        
        # Dynamic goal assignment from CAM 1
        if self.allied_goal_color is None:
            from vision_config import config
            if det1.get("blue_goal") is not None:
                self.allied_goal_color = "blue"
                config["physics"]["team_color"] = "yellow"
                print("Dynamic Goal Assignment: Allied = Blue, Enemy = Yellow")
            elif det1.get("yellow_goal") is not None:
                self.allied_goal_color = "yellow"
                config["physics"]["team_color"] = "blue"
                print("Dynamic Goal Assignment: Allied = Yellow, Enemy = Blue")

        # 3. Fast Algebraic Hitbox Math
        threats0 = filter_threats(det0.get("unknowns", []), [], safe_radius=40)
        threats1 = filter_threats(det1.get("unknowns", []), [], safe_radius=40)
        # 4. State Merging - Independent Object Fusion
        final_det = {
            "ball": det0["ball"] if det0["ball"] else det1["ball"],
            "ball_cam": 0 if det0["ball"] else (1 if det1["ball"] else 0),
            
            "blue_goal": det0["blue_goal"] if det0["blue_goal"] else det1["blue_goal"],
            "blue_goal_cam": 0 if det0["blue_goal"] else (1 if det1["blue_goal"] else 0),
            
            "yellow_goal": det0["yellow_goal"] if det0["yellow_goal"] else det1["yellow_goal"],
            "yellow_goal_cam": 0 if det0["yellow_goal"] else (1 if det1["yellow_goal"] else 0),
        }
        
        # Keep ALL threats across both cameras since a threat behind us is still dangerous
        final_threats = [{"bbox": t, "cam_id": 0} for t in threats0] + [{"bbox": t, "cam_id": 1} for t in threats1]
        
        # Export lines for boundary avoidance
        final_lines = []
        for line in det0.get("white_lines", []):
            final_lines.append({"center": line["center"], "cam_id": 0})
        for line in det1.get("white_lines", []):
            final_lines.append({"center": line["center"], "cam_id": 1})
            
        return {
            "detections": final_det,
            "threats": final_threats,
            "lines": final_lines
        }

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
