import cv2
import numpy as np
import json
import sys

# Colors to calibrate
COLORS_TO_CALIBRATE = [
    "green_grass",
    "black_walls",
    "white_lines",
    "orange_ball",
    "blue_goal",
    "yellow_goal",
    "red_marker"
]

clicks = []

def mouse_callback(event, x, y, flags, param):
    """Callback for mouse clicks."""
    if event == cv2.EVENT_LBUTTONDOWN:
        clicks.append((x, y))
        print(f"Clicked: ({x}, {y})")

def run_calibration(video_path):
    """Interactive GUI to calibrate YUV color bounds."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to open {video_path}")
        return

    # Resize frame if it's too large to fit on a laptop screen
    max_width = 1280
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))

    yuv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    
    cv2.namedWindow("Calibration", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    config = {"colors": {}}
    
    print("\n=== INTERACTIVE COLOR CALIBRATION ===")
    print("For each color, click multiple points to sample the color.")
    print("Try to click the darkest and brightest spots of the object.")
    print("Press 'n' to calculate bounds and move to the next color.")
    print("Press 'q' to quit at any time.\n")
    
    for color_name in COLORS_TO_CALIBRATE:
        global clicks
        clicks = []
        
        while True:
            display_frame = frame.copy()
            
            # Draw instructions
            cv2.putText(display_frame, f"Calibrating: {color_name}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            cv2.putText(display_frame, f"Clicks: {len(clicks)} | Press 'n' to Continue", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Draw collected points
            for cx, cy in clicks:
                cv2.circle(display_frame, (cx, cy), 5, (255, 0, 0), -1)
                
            cv2.imshow("Calibration", display_frame)
            key = cv2.waitKey(20) & 0xFF
            
            if key == ord('n'):
                break
            elif key == ord('q'):
                print("Calibration aborted.")
                sys.exit()
                
        if not clicks:
            print(f"No clicks for {color_name}. Using default maximum bounds (0-255).")
            bounds = {"u_min": 0, "u_max": 255, "v_min": 0, "v_max": 255, "y_min": 0, "y_max": 255}
        else:
            y_vals, u_vals, v_vals = [], [], []
            for cx, cy in clicks:
                # Sample a 3x3 patch around the click for robustness against noise
                patch = yuv_frame[max(0, cy-1):min(yuv_frame.shape[0], cy+2), 
                                  max(0, cx-1):min(yuv_frame.shape[1], cx+2)]
                for r in range(patch.shape[0]):
                    for c in range(patch.shape[1]):
                        y_vals.append(patch[r, c, 0])
                        u_vals.append(patch[r, c, 1])
                        v_vals.append(patch[r, c, 2])
            
            # Natural variance tolerances (can be adjusted)
            tol_y, tol_u, tol_v = 30, 20, 20
            
            bounds = {
                "u_min": max(0, int(np.min(u_vals)) - tol_u),
                "u_max": min(255, int(np.max(u_vals)) + tol_u),
                "v_min": max(0, int(np.min(v_vals)) - tol_v),
                "v_max": min(255, int(np.max(v_vals)) + tol_v),
                "y_min": max(0, int(np.min(y_vals)) - tol_y),
                "y_max": min(255, int(np.max(y_vals)) + tol_y)
            }
            print(f"Recorded {len(clicks)} points for {color_name}. Computed bounds: {bounds}")
            
        config["colors"][color_name] = bounds
        
    cv2.destroyAllWindows()
    cap.release()
    
    print("\n\n=== CALIBRATION COMPLETE ===")
    print("Copy and paste the following output directly into vision_config.py:\n")
    print(json.dumps(config, indent=4))

if __name__ == "__main__":
    # If no argument is provided, default to video2.mp4
    path = sys.argv[1] if len(sys.argv) > 1 else "video2.mp4"
    run_calibration(path)
