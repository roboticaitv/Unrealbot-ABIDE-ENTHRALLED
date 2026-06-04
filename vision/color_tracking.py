import cv2
import numpy as np
import math

class ColorTracker:
    def __init__(self, fisheye_crop=0.88):
        # We define ranges for U and V channels (0-255).
        # In YUV420, Y is the luma (brightness), U and V are chroma.
        # These values need to be calibrated for the specific lighting, 
        # but here are some logical defaults to start.
        
        # Orange (Ball): Sampled U~94, V~144. Pure center pixels reach V~243.
        # Lowered v_min from 170->130 to catch the full ball. Shape (circularity)
        # separates ball from yellow goal which overlaps slightly in UV.
        self.orange_bounds = {'u_min': 50, 'u_max': 125, 'v_min': 130, 'v_max': 255}
        
        # Blue (Goal): Sampled U~155, V~95. Confirmed working.
        self.blue_bounds = {'u_min': 135, 'u_max': 255, 'v_min': 60, 'v_max': 130}
        
        # Yellow (Goal): Sampled U~95, V~129. Tightened v_max from 180->145
        # to reduce overlap with orange ball (ball V~144). Width ratio filter
        # provides additional separation (goals are wide, ball is round).
        self.yellow_bounds = {'u_min': 0, 'u_max': 110, 'v_min': 100, 'v_max': 145}
        
        # Green (Grass): Sampled U~120-124, V~79-107. Wall starts at V~124.
        self.green_bounds = {'u_min': 100, 'u_max': 135, 'v_min': 10, 'v_max': 115}
        
        # Black/Gray/White (Walls, Field Lines): Sampled U~129-131, V~125-134.
        self.black_bounds = {'u_min': 115, 'u_max': 141, 'v_min': 115, 'v_max': 141}
        
        # Fisheye lens mask: ignore everything outside useful circular area
        # fisheye_crop = 0.88 means we use 88% of the inscribed circle radius
        self.fisheye_crop = fisheye_crop
        
        # Vertical crop: Ignore top 10% (always useless) and bottom 30% (robot chassis/wires)
        self.crop_top = 0.10
        self.crop_bottom = 0.30
        
        self._mask_cache = {}
    
    def _get_lens_mask(self, h, w):
        """Generate a mask that blacks out the fisheye border, the ceiling, and the robot's own wiring."""
        key = (h, w)
        if key not in self._mask_cache:
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2
            
            # 1. Draw the valid fisheye circle
            radius = int(min(cx, cy) * self.fisheye_crop)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            # 2. Black out the ceiling (top)
            top_px = int(h * self.crop_top)
            cv2.rectangle(mask, (0, 0), (w, top_px), 0, -1)
            
            # 3. Black out the robot body/wires (bottom)
            bottom_px = int(h * (1.0 - self.crop_bottom))
            cv2.rectangle(mask, (0, bottom_px), (w, h), 0, -1)
            
            # 4. Black out hardware padding on all edges (ISP alignment artifacts)
            pad = 16
            cv2.rectangle(mask, (0, 0), (pad, h), 0, -1)          # left
            cv2.rectangle(mask, (w - pad, 0), (w, h), 0, -1)      # right
            cv2.rectangle(mask, (0, 0), (w, pad), 0, -1)           # top
            cv2.rectangle(mask, (0, h - pad), (w, h), 0, -1)      # bottom
            
            self._mask_cache[key] = mask
        return self._mask_cache[key]
        
    def get_dynamic_lens_mask(self, u, v, scale=1):
        """
        Builds a dynamic mask by finding the green field.
        Everything above the green field (plus a margin) is ignored (horizon detection).
        This eliminates walls, ceilings, people, and background noise.
        """
        # 1. Get the static mask (fisheye circle + bottom wire crop)
        h, w = u.shape
        static_mask = self._get_lens_mask(h, w).copy()
        
        # 2. Find the green grass
        green_mask = cv2.inRange(
            cv2.merge([u, v]), 
            np.array([self.green_bounds['u_min'], self.green_bounds['v_min']]), 
            np.array([self.green_bounds['u_max'], self.green_bounds['v_max']])
        )
        green_mask = cv2.bitwise_and(green_mask, static_mask)
        
        # 3. Morphological cleanup: merge field across white lines, remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Default horizon is the top crop if no field is found
        horizon_y = int(h * self.crop_top)
        
        # 4. Find the field: the largest green blob whose bottom edge
        #    extends past 50% of the frame (proving it's the actual field,
        #    not a reflection on a wall or ceiling)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_field_area = 2000 / (scale * scale)
        field_contour = None
        
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < min_field_area:
                break
            x, y, bw, bh = cv2.boundingRect(cnt)
            # The field must extend into the bottom half of the frame
            if (y + bh) > h * 0.5:
                field_contour = cnt
                break
        
        if field_contour is not None:
            x, y, bw, bh = cv2.boundingRect(field_contour)
            # The horizon is the top edge of the grass. We subtract 80 pixels 
            # (adjusted by scale) to reach the top of the black wall.
            margin = int(80 / scale)
            horizon_y = max(0, y - margin)
            cv2.rectangle(static_mask, (0, 0), (w, horizon_y), 0, -1)
                
        return static_mask, horizon_y

    def process_yuv_frame(self, frame):
        """
        Takes either:
        - A planar YUV420 array from Picamera2 (shape: H*3//2, W)
        - A standard BGR image from OpenCV (shape: H, W, 3)
        Returns a dictionary of detected objects.
        """
        scale = 1
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Standard BGR image (from video files or OpenCV fallback)
            yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
        elif len(frame.shape) == 2:
            # Planar YUV420 (I420) from Picamera2
            # Shape is (H * 3 // 2, W) where H is the actual image height
            total_h, w = frame.shape
            h = total_h * 2 // 3
            half_h = h // 2
            half_w = w // 2
            
            # Extract U and V at their native half resolution (no resize!)
            u_channel = frame[h : h + h // 4, :].reshape(half_h, half_w)
            v_channel = frame[h + h // 4 :, :].reshape(half_h, half_w)
            # Coordinates will be at half scale, so we multiply results by 2
            scale = 2
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Compute the dynamic lens mask and horizon ONCE per frame
        lens_mask, horizon_y = self.get_dynamic_lens_mask(u_channel, v_channel, scale)

        results = {
            "ball": self._find_orange_ball(u_channel, v_channel, lens_mask, scale),
            "blue_goal": self._find_blob(u_channel, v_channel, self.blue_bounds, lens_mask, scale),
            "yellow_goal": self._find_blob(u_channel, v_channel, self.yellow_bounds, lens_mask, scale),
            "unknowns": self._find_unknown_blobs(u_channel, v_channel, lens_mask, scale),
            "horizon_y": horizon_y * scale
        }
        return results

    def _find_unknown_blobs(self, u, v, lens_mask, scale=1):
        """Finds any large blobs that do not match known colors."""
        known_mask = np.zeros(u.shape, dtype=np.uint8)
        uv_merged = cv2.merge([u, v])
        
        for bounds in [self.orange_bounds, self.blue_bounds, self.yellow_bounds, self.green_bounds, self.black_bounds]:
            mask = cv2.inRange(
                uv_merged,
                np.array([bounds['u_min'], bounds['v_min']]),
                np.array([bounds['u_max'], bounds['v_max']])
            )
            known_mask = cv2.bitwise_or(known_mask, mask)
        
        # Invert and apply the dynamic horizon mask
        unknown_mask = cv2.bitwise_not(known_mask)
        unknown_mask = cv2.bitwise_and(unknown_mask, lens_mask)
        
        # Morphological CLOSE to fill holes and unify fragmented robot pieces
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        unknown_boxes = []
        # Sort contours by largest first
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(cnt)
            
            # Adjusted area limits (scaled for downsampled UV)
            min_area = 200 / (scale * scale)  # roughly 800px at full scale
            max_area = 15000 / (scale * scale) # roughly 60,000px at full scale
            if area < min_area or area > max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Shape Ratio Filter: Reject extremely wide lines or extremely tall wires
            # A robot is generally cubic (1:1). 
            if w > h * 3.5 or h > w * 3.5:
                continue
                
            unknown_boxes.append((x * scale, y * scale, w * scale, h * scale))
            
            # Limit to top 2 biggest enemies to save downstream processing
            if len(unknown_boxes) >= 2:
                break
            
        return unknown_boxes
        
    def _find_orange_ball(self, u, v, lens_mask, scale=1):
        """Finds the orange ball using area and circularity, no morphology."""
        mask = cv2.inRange(
            cv2.merge([u, v]), 
            np.array([self.orange_bounds['u_min'], self.orange_bounds['v_min']]), 
            np.array([self.orange_bounds['u_max'], self.orange_bounds['v_max']])
        )
        
        # Apply dynamic horizon mask to ignore everything outside the field
        mask = cv2.bitwise_and(mask, lens_mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = 100 / (scale * scale) # Roughly 25px on the downsampled UV plane
            if area < min_area:
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
                        cx = int(M["m10"] / M["m00"]) * scale
                        cy = int(M["m01"] / M["m00"]) * scale
                        x, y, w, h = cv2.boundingRect(cnt)
                        best_ball = {"x": cx, "y": cy, "bbox": (x*scale, y*scale, w*scale, h*scale), "area": area}
                        
        return best_ball

    def _find_blob(self, u, v, bounds, lens_mask, scale=1):
        """Finds a general blob (like a goal) using just area."""
        mask = cv2.inRange(
            cv2.merge([u, v]), 
            np.array([bounds['u_min'], bounds['v_min']]), 
            np.array([bounds['u_max'], bounds['v_max']])
        )
        
        # Apply dynamic horizon mask to ignore everything outside the field
        mask = cv2.bitwise_and(mask, lens_mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_blob = None
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500:  # Goals should be large
                continue
            
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Goal logic: A goal should generally be wide.
            # Reject things that are extremely tall and thin (like wires)
            if h > w * 2:
                continue
            
            if area > max_area:
                max_area = area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) * scale
                    cy = int(M["m01"] / M["m00"]) * scale
                    x, y, w, h = cv2.boundingRect(cnt)
                    best_blob = {"x": cx, "y": cy, "bbox": (x*scale, y*scale, w*scale, h*scale), "area": area}
        
        return best_blob

if __name__ == "__main__":
    # Test script
    tracker = ColorTracker()
    print("Color tracker initialized. Awaiting frames.")
