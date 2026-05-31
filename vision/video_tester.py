import cv2
import time
from color_tracking import ColorTracker
from aruco_tracker import IntermittentArucoTracker
from hitbox_math import filter_threats

class VideoTester:
    def __init__(self, video_path):
        self.video_path = video_path
        self.color_tracker = ColorTracker()
        self.aruco_tracker = IntermittentArucoTracker(interval=10)
        self.cap = cv2.VideoCapture(video_path)
        
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
                
            # Process frame
            start_time = time.perf_counter()
            state = self._process_frame(frame)
            end_time = time.perf_counter()
            
            # Calculate inference time
            process_ms = (end_time - start_time) * 1000
            
            # Draw overlays
            self._draw_overlay(frame, state, process_ms)
            
            # Show output
            cv2.imshow("Vision Test", frame)
            
            # Delay to roughly match video speed (assuming ~30fps video, 33ms wait)
            # Will wait longer if processing took more time.
            key = cv2.waitKey(max(1, 33 - int(process_ms))) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                cv2.waitKey(0) # Wait until space is pressed again

        self.cap.release()
        cv2.destroyAllWindows()

    def _process_frame(self, frame):
        # 1. Color tracking
        # Our color_tracker accepts standard BGR frames for testing and extracts YUV internally!
        color_results = self.color_tracker.process_yuv_frame(frame) 
        
        # 2. Aruco tracking
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        aruco_positions = self.aruco_tracker.process_frame(gray)
        
        # 3. Hitbox Math
        ally_boxes = []
        for marker_id, (cx, cy) in aruco_positions.items():
            ally_boxes.append((cx - 25, cy - 25, 50, 50))
            
        # Simulate unknown blobs if we had a generic contour list (omitted here for simplicity)
        threats = filter_threats([], ally_boxes, safe_radius=30)
        
        return {
            "ball": color_results["ball"],
            "blue_goal": color_results["blue_goal"],
            "yellow_goal": color_results["yellow_goal"],
            "allies": aruco_positions,
            "threats": threats
        }
        
    def _draw_overlay(self, frame, state, process_ms):
        # Display inference time
        cv2.putText(frame, f"Latency: {process_ms:.1f} ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw Ball
        if state["ball"]:
            x, y, w, h = state["ball"]["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
            cv2.putText(frame, "Ball", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            
        # Draw Blue Goal
        if state["blue_goal"]:
            x, y, w, h = state["blue_goal"]["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, "Blue Goal", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
        # Draw Yellow Goal
        if state["yellow_goal"]:
            x, y, w, h = state["yellow_goal"]["bbox"]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.putText(frame, "Yellow Goal", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
        # Draw Allies
        for marker_id, (cx, cy) in state["allies"].items():
            cv2.circle(frame, (cx, cy), 25, (0, 255, 0), 2)
            cv2.putText(frame, f"Ally {marker_id}", (cx-20, cy-35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        video_file = sys.argv[1]
        tester = VideoTester(video_file)
        tester.run()
    else:
        print("Usage: python video_tester.py <path_to_video.mp4>")
