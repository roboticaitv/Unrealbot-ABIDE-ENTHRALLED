import cv2
import numpy as np

class IntermittentArucoTracker:
    def __init__(self, dictionary_type=cv2.aruco.DICT_4X4_50, interval=10):
        """
        Runs ArUco detection only once every `interval` frames to save CPU.
        Returns the last known position for the frames in between.
        """
        self.dictionary = cv2.aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = cv2.aruco.DetectorParameters()
        self.parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_CONTOUR
        self.detector = cv2.aruco.ArucoDetector(self.dictionary, self.parameters)
        
        self.interval = interval
        self.frame_count = 0
        
        self.last_corners = None
        self.last_ids = None
        self.last_known_positions = {}

    def process_frame(self, gray_frame, y_offset=0):
        """
        Takes a grayscale image. Detects ArUcos every N frames.
        Returns a dictionary mapping marker ID to its center (x, y).
        """
        self.frame_count += 1
        
        # Only run full detection every `interval` frames
        if self.frame_count % self.interval == 0:
            corners, ids, _ = self.detector.detectMarkers(gray_frame)
            
            if ids is not None:
                self.last_corners = corners
                self.last_ids = ids.flatten()
                
                # Calculate centers and apparent widths
                self.last_known_positions.clear()
                for i, marker_id in enumerate(self.last_ids):
                    c = corners[i][0]
                    cx = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
                    cy = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
                    
                    # Add horizon offset back to make it full-frame coordinate
                    cy += y_offset
                    
                    # Apparent width in pixels (average of top and bottom edge)
                    top_w = np.sqrt((c[1][0]-c[0][0])**2 + (c[1][1]-c[0][1])**2)
                    bot_w = np.sqrt((c[2][0]-c[3][0])**2 + (c[2][1]-c[3][1])**2)
                    apparent_px = (top_w + bot_w) / 2.0
                    self.last_known_positions[marker_id] = (cx, cy, apparent_px)
            else:
                # We lost them
                self.last_known_positions.clear()

        # Returning the last known positions (either freshly updated or from memory)
        return self.last_known_positions

if __name__ == "__main__":
    tracker = IntermittentArucoTracker(interval=10)
    print("Aruco tracker initialized. Awaiting frames.")
