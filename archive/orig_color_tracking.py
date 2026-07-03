import cv2
import numpy as np
import math
from vision_config import config

class ColorTracker:
    def __init__(self):
        # Load Color bounds from config
        self.orange_bounds = config["colors"]["orange_ball"]
        self.blue_bounds = config["colors"]["blue_goal"]
        self.yellow_bounds = config["colors"]["yellow_goal"]
        self.green_bounds = config["colors"]["green_grass"]
        self.black_bounds = config["colors"]["black_walls"]
        self.white_bounds = config["colors"]["white_lines"]
        self.red_bounds = config["colors"].get("red_marker", {"u_min":0, "u_max":255, "v_min":160, "v_max":255})
        
        self.min_ball_area = config["physics"].get("min_ball_area_px", 30)
        self.min_enemy_area = config["physics"].get("min_enemy_area_px", 400)
        
        # Load Masking constraints from config
        self.fisheye_rx = config["masking"]["fisheye_radius_x_pct"]
        self.fisheye_ry = config["masking"]["fisheye_radius_y_pct"]
        self.crop_top = config["masking"]["crop_top"]
        self.crop_bottom = config["masking"]["crop_bottom"]
        self.wedge_w = config["masking"]["corner_wedge_width"]
        self.wedge_h = config["masking"]["corner_wedge_height"]
        
        self._mask_cache = {}
        
        from verify_net import VerifyNet
        self.verifier = VerifyNet()
    
    def _get_lens_mask(self, h, w):
        """Generate a mask that blacks out the fisheye border, the ceiling, and the robot's own wiring."""
        key = (h, w)
        if key not in self._mask_cache:
            mask = np.zeros((h, w), dtype=np.uint8)
            cx, cy = w // 2, h // 2
            
            # 1. Draw the valid fisheye ellipse (fixes horizontal screen loss)
            rx = int(cx * self.fisheye_rx)
            ry = int(cy * self.fisheye_ry)
            cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
            
            # 2. Black out the ceiling (top)
            top_px = int(h * self.crop_top)
            cv2.rectangle(mask, (0, 0), (w, top_px), 0, -1)
            
            # 3. Black out the robot chassis (bottom)
            bottom_px = int(h * (1.0 - self.crop_bottom))
            cv2.rectangle(mask, (0, bottom_px), (w, h), 0, -1)
            
            # 3b. Corner wedge exclusion zones where robot hardware sticks up
            bl_pts = np.array([[0, int(h * self.wedge_h)], [0, h], [int(w * self.wedge_w), h]], np.int32)
            cv2.fillPoly(mask, [bl_pts], 0)
            br_pts = np.array([[w, int(h * self.wedge_h)], [w, h], [int(w * (1.0 - self.wedge_w)), h]], np.int32)
            cv2.fillPoly(mask, [br_pts], 0)
            
            # 4. Black out hardware padding on all edges (ISP alignment artifacts)
            pad = 16
            cv2.rectangle(mask, (0, 0), (pad, h), 0, -1)          # left
            cv2.rectangle(mask, (w - pad, 0), (w, h), 0, -1)      # right
            cv2.rectangle(mask, (0, 0), (w, pad), 0, -1)           # top
            cv2.rectangle(mask, (0, h - pad), (w, h), 0, -1)      # bottom
            
            self._mask_cache[key] = mask
        return self._mask_cache[key]
        
    def get_dynamic_lens_mask(self, yuv_merged, scale=1):
        """
        Builds a dynamic mask by finding the green field.
        Everything above the green field (plus a margin) is ignored (horizon detection).
        This eliminates walls, ceilings, people, and background noise.
        """
        # 1. Get the static mask (fisheye circle + bottom wire crop)
        h, w = yuv_merged.shape[:2]
        static_mask = self._get_lens_mask(h, w)
        
        # 2. Find the green grass
        green_mask = cv2.inRange(
            yuv_merged[:, :, 1:], 
            np.array([self.green_bounds['u_min'], self.green_bounds['v_min']]), 
            np.array([self.green_bounds['u_max'], self.green_bounds['v_max']])
        )
        green_mask = cv2.bitwise_and(green_mask, static_mask)
        
        # 3. Morphological cleanup: merge field across white lines, remove small noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        
        # Default horizon is the top crop if no field is found
        highest_y = int(h * self.crop_top)
        horizon_poly = None
        
        # 4. Find the field: the largest green blob whose bottom edge
        #    extends past 50% of the frame (proving it's the actual field)
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_field_area = 2000 / (scale * scale)
        field_contour = None
        
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(cnt) < min_field_area:
                break
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if (by + bh) > h * 0.5:
                field_contour = cnt
                break
        
        if field_contour is not None:
            # ── 1. The Convex Anchor ──
            field_hull = cv2.convexHull(field_contour)
            safe_zone_mask = np.zeros_like(green_mask)
            cv2.drawContours(safe_zone_mask, [field_hull], -1, 255, -1)
            
            # ── 2. The Wall Mask ──
            wall_mask = np.zeros_like(green_mask)
            for bounds in [self.black_bounds, self.blue_bounds, self.yellow_bounds]:
                mask = cv2.inRange(
                    yuv_merged,
                    np.array([bounds.get('y_min', 0), bounds['u_min'], bounds['v_min']]),
                    np.array([bounds.get('y_max', 255), bounds['u_max'], bounds['v_max']])
                )
                wall_mask = cv2.bitwise_or(wall_mask, mask)
            
            wall_mask = cv2.dilate(wall_mask, np.ones((5, 1), np.uint8))
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
            wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, v_kernel)
            
            # ── 3. Upward Scan from the Convex Anchor ──
            step_x = max(1, w // 30)
            raw_points = []
            
            for x in range(0, w, step_x):
                col_grass = safe_zone_mask[:, x]
                green_ys = np.nonzero(col_grass)[0]
                if len(green_ys) == 0:
                    continue
                
                top_green_y = green_ys[0]
                
                col_wall = wall_mask[:top_green_y, x]
                rev_wall = col_wall[::-1]
                zeros = np.nonzero(rev_wall == 0)[0]
                
                if len(zeros) > 0:
                    climb_height = zeros[0] # Allow natural wall thickness
                else:
                    climb_height = len(rev_wall)
                    
                raw_points.append([x, top_green_y - climb_height])
                
            # Rightmost edge
            if len(raw_points) > 0 and raw_points[-1][0] < w - 1:
                x = w - 1
                col_grass = safe_zone_mask[:, x]
                green_ys = np.nonzero(col_grass)[0]
                if len(green_ys) > 0:
                    top_green_y = green_ys[0]
                    col_wall = wall_mask[:top_green_y, x]
                    rev_wall = col_wall[::-1]
                    zeros = np.nonzero(rev_wall == 0)[0]
                    climb_height = zeros[0] if len(zeros) > 0 else len(rev_wall)
                    raw_points.append([x, top_green_y - climb_height])
            
            # ── 4. Outlaw Elimination (Local Derivative Filter) ──
            if len(raw_points) > 0:
                # Anchor the filter in the center of the image to avoid locking to an edge Outlaw
                center_idx = len(raw_points) // 2
                filtered_points = [None] * len(raw_points)
                filtered_points[center_idx] = raw_points[center_idx]
                
                max_jump = int(15 / scale) # Strict derivative calibration
                
                # Scan Right
                for i in range(center_idx + 1, len(raw_points)):
                    curr_x, curr_y = raw_points[i]
                    prev_x, prev_y = filtered_points[i-1]
                    if abs(curr_y - prev_y) > max_jump:
                        filtered_points[i] = [curr_x, prev_y] # OUTLAW ELIMINATED! Hold steady.
                    else:
                        filtered_points[i] = [curr_x, curr_y]
                        
                # Scan Left
                for i in range(center_idx - 1, -1, -1):
                    curr_x, curr_y = raw_points[i]
                    prev_x, prev_y = filtered_points[i+1]
                    if abs(curr_y - prev_y) > max_jump:
                        filtered_points[i] = [curr_x, prev_y] # OUTLAW ELIMINATED!
                    else:
                        filtered_points[i] = [curr_x, curr_y]
                        
                # Smooth the filtered points
                ys = [p[1] for p in filtered_points]
                window = 5
                smoothed_ys = []
                for i in range(len(ys)):
                    start = max(0, i - window // 2)
                    end = min(len(ys), i + window // 2 + 1)
                    smoothed_ys.append(int(sum(ys[start:end]) / (end - start)))
                    
                polygon_points = [[filtered_points[i][0], smoothed_ys[i]] for i in range(len(filtered_points))]
                
                # Extrapolate curve to the far left edge (x=0) to fix the visual dip
                if len(polygon_points) > 1 and polygon_points[0][0] > 0:
                    x0, y0 = polygon_points[0]
                    x1, y1 = polygon_points[1]
                    slope = (y1 - y0) / (x1 - x0) if x1 != x0 else 0
                    y_extrap = int(y0 - slope * x0) 
                    y_extrap = max(0, min(h, y_extrap))
                    polygon_points.insert(0, [0, y_extrap])
                    
                # Extrapolate curve to the far right edge (x=w)
                if len(polygon_points) > 1 and polygon_points[-1][0] < w:
                    x_last, y_last = polygon_points[-1]
                    x_prev, y_prev = polygon_points[-2]
                    slope = (y_last - y_prev) / (x_last - x_prev) if x_last != x_prev else 0
                    y_extrap = int(y_last + slope * (w - x_last)) 
                    y_extrap = max(0, min(h, y_extrap))
                    polygon_points.append([w, y_extrap])
            else:
                polygon_points = []
                
            horizon_poly = [(pt[0] * scale, pt[1] * scale) for pt in polygon_points]
            
            if len(polygon_points) > 0:
                highest_y = int(np.min([p[1] for p in polygon_points]))
            else:
                highest_y = int(np.min(field_hull[:, 0, 1]))

            # ── 5. Create the Dynamic Lens Mask ──
            dynamic_lens_mask = np.zeros_like(green_mask)
            if len(polygon_points) > 0:
                fill_pts = [p[:] for p in polygon_points]
                fill_pts.append([w, fill_pts[-1][1]])
                fill_pts.append([w, h])
                fill_pts.append([0, h])
                fill_pts.insert(0, [0, fill_pts[0][1]])
                pts = np.array(fill_pts, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(dynamic_lens_mask, [pts], 255)
            
            dynamic_lens_mask = cv2.bitwise_and(static_mask, dynamic_lens_mask)   

        else:
            # ALWAYS black out everything above the flat default horizon if no field
            dynamic_lens_mask = static_mask.copy()
            cv2.rectangle(dynamic_lens_mask, (0, 0), (w, highest_y), 0, -1)
                
        return dynamic_lens_mask, highest_y, horizon_poly, field_contour, static_mask

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
            y_channel = yuv[:, :, 0]
            u_channel = yuv[:, :, 1]
            v_channel = yuv[:, :, 2]
        elif len(frame.shape) == 2:
            # Planar YUV420 (I420) from Picamera2
            # Shape is (H * 3 // 2, W) where H is the actual image height
            total_h, w = frame.shape
            h = total_h * 2 // 3
            half_h = h // 2
            half_w = w // 2
            
            # Y is full resolution, extract and downsample to match U/V
            y_channel_full = frame[:h, :]
            y_channel = y_channel_full[::2, ::2]
            
            # Extract U and V at their native half resolution (no resize!)
            u_channel = frame[h : h + h // 4, :].reshape(half_h, half_w)
            v_channel = frame[h + h // 4 :, :].reshape(half_h, half_w)
            # Coordinates will be at half scale, so we multiply results by 2
            scale = 2
        else:
            raise ValueError(f"Unexpected frame shape: {frame.shape}")

        # Compute the dynamic lens mask and horizon ONCE per frame
        yuv_merged = cv2.merge([y_channel, u_channel, v_channel])
        dynamic_lens_mask, highest_y, horizon_poly, field_contour, static_lens_mask = self.get_dynamic_lens_mask(yuv_merged, scale)

        # The True "Diminishing" Upgrade: Physical Array Slicing
        # We only slice for things that exist ON the grass (ball, enemies).
        yuv_sliced = yuv_merged[highest_y:, :, :]
        lens_mask_sliced = dynamic_lens_mask[highest_y:, :]
        
        # Scale mask to full frame resolution for visualization
        if scale > 1:
            full_mask = cv2.resize(dynamic_lens_mask, (dynamic_lens_mask.shape[1] * scale, dynamic_lens_mask.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        else:
            full_mask = dynamic_lens_mask.copy()
            
        scaled_field_contour = None
        if field_contour is not None:
            scaled_field_contour = (field_contour * scale).astype(np.int32)

        ball_result = self._find_orange_ball(yuv_sliced, lens_mask_sliced, scale, highest_y)
        
        results = {
            "ball": ball_result["best_ball"],
            "ball_candidates": ball_result["candidates"],
            "blue_goal": self._find_blob(yuv_sliced, self.blue_bounds, lens_mask_sliced, scale, highest_y),
            "yellow_goal": self._find_blob(yuv_sliced, self.yellow_bounds, lens_mask_sliced, scale, highest_y),
            "unknowns": self._find_unknown_blobs(yuv_sliced, lens_mask_sliced, scale, highest_y),
            "horizon_y": highest_y * scale,
            "horizon_poly": horizon_poly,
            "detection_mask": full_mask,
            "field_contour": scaled_field_contour
        }
        return results

    def _find_unknown_blobs(self, yuv_merged, lens_mask, scale=1, horizon_y=0):
        """Finds any large blobs that do not match known colors (Orange, Blue, Yellow, Green, White).
        By NOT including 'Black' in the known list, black robots become unknowns.
        By INCLUDING 'White' in the known list, white field lines are ignored."""
        known_mask = np.zeros(lens_mask.shape, dtype=np.uint8)
        
        # We explicitly do NOT add black_bounds to the known_mask.
        for bounds in [self.orange_bounds, self.blue_bounds, self.yellow_bounds, self.green_bounds, self.white_bounds, self.red_bounds]:
            mask = cv2.inRange(
                yuv_merged,
                np.array([bounds.get('y_min', 0), bounds['u_min'], bounds['v_min']]),
                np.array([bounds.get('y_max', 255), bounds['u_max'], bounds['v_max']])
            )
            known_mask = cv2.bitwise_or(known_mask, mask)
            
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
            min_area = self.min_enemy_area / (scale * scale)
            max_area = 15000 / (scale * scale) # roughly 60,000px at full scale
            if area < min_area or area > max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Shape Ratio Filter: Reject extremely wide lines or extremely tall wires
            # A robot is generally a cylinder/cube (1:1). 
            if w > h * 1.8 or h > w * 1.8:
                continue
                
            # Edge Density Shadow Rejection (Notebook LM algorithm)
            # 3D physical cylinders have high internal edge density (wheels, seams, glare).
            # 2D flat shadows on grass have virtually zero internal edges.
            y_patch = yuv_merged[y:y+h, x:x+w, 0]
            if y_patch.size > 0:
                edges = cv2.Canny(y_patch, 30, 100)
                edge_density = np.count_nonzero(edges) / edges.size
                if edge_density < 0.05: # Less than 5% edge pixels = Flat Shadow! Reject!
                    continue
            
            # Must-Touch-Field: A real enemy robot stands ON the green field.
            # If the blob touches the very bottom of the screen, it's our own chassis! Reject!
            foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
            if foot_y_start >= yuv_merged.shape[0] - int(5 / scale):
                continue
                
            # Check for green pixels just below the blob's bottom edge.
            check_below = max(3, int(10 / scale))
            foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
            foot_y_end = min(y + h + check_below, yuv_merged.shape[0])
            if foot_y_end > foot_y_start and w > 0:
                foot_uv = yuv_merged[foot_y_start:foot_y_end, x:x+w, 1:]
                if foot_uv.size > 0:
                    green_check = cv2.inRange(
                        foot_uv,
                        np.array([self.green_bounds['u_min'], self.green_bounds['v_min']]),
                        np.array([self.green_bounds['u_max'], self.green_bounds['v_max']])
                    )
                    green_ratio = np.count_nonzero(green_check) / green_check.size
                    if green_ratio < 0.15:  # Less than 15% green below = not on field
                        continue
                
            # Add the physical slice offset back
            y += horizon_y
            
            unknown_boxes.append((x * scale, y * scale, w * scale, h * scale))
            
            # Limit to top 2 biggest enemies to save downstream processing
            if len(unknown_boxes) >= 2:
                break
            
        return unknown_boxes
        
    def _find_orange_ball(self, yuv_merged, lens_mask, scale=1, horizon_y=0):
        """Finds the orange ball using area + morphological closing for motion blur."""
        mask = cv2.inRange(
            yuv_merged, 
            np.array([self.orange_bounds.get('y_min', 0), self.orange_bounds['u_min'], self.orange_bounds['v_min']]), 
            np.array([self.orange_bounds.get('y_max', 255), self.orange_bounds['u_max'], self.orange_bounds['v_max']])
        )
        
        # Apply dynamic horizon mask to ignore everything outside the field
        mask = cv2.bitwise_and(mask, lens_mask)
        
        # Morphological close to bridge motion-blur fragments into one blob
        ball_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ball_kernel)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_ball = None
        candidates = []
        max_area = 0
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            min_area = self.min_ball_area / (scale * scale)
            if area < min_area:
                continue
                
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                x, y, w, h = cv2.boundingRect(cnt)
                
                # Reject if the blob touches the very bottom of the screen (our own chassis!)
                foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
                if foot_y_start >= yuv_merged.shape[0] - int(5 / scale):
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                candidate = {
                    "bbox": (int(x * scale), int((y + horizon_y) * scale), int(w * scale), int(h * scale)),
                    "center": (int(cx * scale), int((cy + horizon_y) * scale)),
                    "area": area
                }
                candidates.append(candidate)
                
                if hasattr(self, 'verifier') and self.verifier.loaded:
                    patch_size = 32 // scale
                    half_patch = patch_size // 2
                    
                    px_min, px_max = max(0, cx - half_patch), min(yuv_merged.shape[1], cx + half_patch)
                    py_min, py_max = max(0, cy - half_patch), min(yuv_merged.shape[0], cy + half_patch)
                    
                    y_patch = yuv_merged[py_min:py_max, px_min:px_max, 0]
                    u_patch = yuv_merged[py_min:py_max, px_min:px_max, 1]
                    v_patch = yuv_merged[py_min:py_max, px_min:px_max, 2]
                    
                    if y_patch.shape[0] > 0 and y_patch.shape[1] > 0:
                        prob = self.verifier.predict_patch(y_patch, u_patch, v_patch)
                        if prob < 0.5:
                            continue # Neural Network strictly rejected it!
                else:
                    # Fallback to Math Filter
                    if w > h * 2.5 or h > w * 2.5:
                        continue
                    perimeter = cv2.arcLength(cnt, True)
                    if perimeter == 0:
                        continue
                    circularity = 4 * np.pi * (area / (perimeter * perimeter))
                    if circularity < 0.2:
                        continue
                        
                if area > max_area:
                    max_area = area
                    best_ball = candidate
                        
        return {"best_ball": best_ball, "candidates": candidates}

    def _find_blob(self, yuv_merged, bounds, lens_mask, scale=1, horizon_y=0):
        """Finds a general blob (like a goal) using just area."""
        mask = cv2.inRange(
            yuv_merged, 
            np.array([bounds.get('y_min', 0), bounds['u_min'], bounds['v_min']]), 
            np.array([bounds.get('y_max', 255), bounds['u_max'], bounds['v_max']])
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
            
            bx, by, bw, bh = cv2.boundingRect(cnt)
            
            # Goal logic: A goal should generally be wide.
            # Reject things that are extremely tall and thin (like wires)
            if bh > bw * 2:
                continue
            
            if area > max_area:
                max_area = area
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    abs_y = by + horizon_y
                    cx = int((bx + bw/2) * scale)
                    cy = int((abs_y + bh/2) * scale)
                    best_blob = {"x": cx, "y": cy, "bbox": (bx*scale, abs_y*scale, bw*scale, bh*scale), "area": area}
        
        return best_blob

if __name__ == "__main__":
    # Test script
    tracker = ColorTracker()
    print("Color tracker initialized. Awaiting frames.")
