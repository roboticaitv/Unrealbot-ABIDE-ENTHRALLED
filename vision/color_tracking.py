import cv2
import numpy as np
import math

class ColorTracker:
    def __init__(self):
        # We define ranges for U and V channels (0-255).
        # In YUV420, Y is the luma (brightness), U and V are chroma.
        # These values need to be calibrated for the specific lighting, 
        # but here are some logical defaults to start.
        
        # Orange (Ball): Target U~97, V~185
        self.orange_bounds = {'u_min': 70, 'u_max': 125, 'v_min': 155, 'v_max': 255}
        
        # Blue (Goal): Target U~158, V~100
        self.blue_bounds = {'u_min': 135, 'u_max': 255, 'v_min': 60, 'v_max': 130}
        
        # Yellow (Goal): Target U~55, V~130
        self.yellow_bounds = {'u_min': 0, 'u_max': 90, 'v_min': 105, 'v_max': 160}
        
        # Green (Grass): Target U~122, V~76
        self.green_bounds = {'u_min': 105, 'u_max': 135, 'v_min': 50, 'v_max': 95}

    def process_yuv_frame(self, yuv_frame):
        """
        Takes a YUV420 image array.
        Returns a dictionary of detected objects.
        """
        # Picamera2 returns YUV420 as a planar array. 
        # For a frame of height H and width W:
        # Y is HxW. U and V are (H/2)x(W/2).
        # For simplicity in OpenCV, we can convert it to YUV 444 or just extract channels.
        # If the input is standard OpenCV BGR, we convert to YUV.
        # Let's assume standard OpenCV YUV format (YCrCb is similar).
        
        # We will assume we received a standard BGR image for testing flexibility, 
        # or a converted YUV image.
        if len(yuv_frame.shape) == 3 and yuv_frame.shape[2] == 3:
            # If it's BGR, convert to YUV
            yuv = cv2.cvtColor(yuv_frame, cv2.COLOR_BGR2YUV)
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
        else:
            # If we received planar YUV data from libcamera directly (requires custom shaping)
            # For now, we will assume the caller converted it to a 3-channel matrix for simplicity.
            raise ValueError("Expected a 3-channel image array.")

        results = {
            "ball": self._find_orange_ball(u_channel, v_channel),
            "blue_goal": self._find_blob(u_channel, v_channel, self.blue_bounds),
            "yellow_goal": self._find_blob(u_channel, v_channel, self.yellow_bounds)
        }
        return results

    def _find_orange_ball(self, u, v):
        """Finds the orange ball using area and circularity, no morphology."""
        mask = cv2.inRange(
            cv2.merge([u, v]), 
            np.array([self.orange_bounds['u_min'], self.orange_bounds['v_min']]), 
            np.array([self.orange_bounds['u_max'], self.orange_bounds['v_max']])
        )
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 50:  # Minimum area threshold
                continue
                
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue
                
            circularity = 4 * math.pi * (area / (perimeter * perimeter))
            
            # A perfect circle has circularity == 1.0. 
            # Motion blur can drastically stretch the contour, so we widen the bounds to [0.4, 1.5]
            if 0.4 < circularity <= 1.5:
                if area > max_area:
                    max_area = area
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        x, y, w, h = cv2.boundingRect(cnt)
                        best_ball = {"x": cx, "y": cy, "bbox": (x, y, w, h), "area": area}
                        
        return best_ball

    def _find_blob(self, u, v, bounds):
        """Finds a general blob (like a goal) using just area."""
        mask = cv2.inRange(
            cv2.merge([u, v]), 
            np.array([bounds['u_min'], bounds['v_min']]), 
            np.array([bounds['u_max'], bounds['v_max']])
        )
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_blob = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:  # Goals should be large
                continue
                
            if area > max_area:
                max_area = area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    x, y, w, h = cv2.boundingRect(cnt)
                    best_blob = {"x": cx, "y": cy, "bbox": (x, y, w, h), "area": area}
                    
        return best_blob

if __name__ == "__main__":
    # Test script
    tracker = ColorTracker()
    print("Color tracker initialized. Awaiting frames.")
