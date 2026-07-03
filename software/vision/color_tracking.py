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
        self.prev_polys = {}
    
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
            cv2.rectangle(mask, (0, 0), (w, pad), 0, -1)           # top
            cv2.rectangle(mask, (0, h - pad), (w, h), 0, -1)      # bottom
            
            # 5. HARD CROP for exactly 180° FOV (Removes overlap between cameras)
            # Base width is 832px. 90° = 369px from center. Margin = 416 - 369 = 47px.
            # Scale proportionally based on 'w'
            margin = int(47 * (w / 832.0))
            cv2.rectangle(mask, (0, 0), (margin, h), 0, -1)          # left
            cv2.rectangle(mask, (w - margin, 0), (w, h), 0, -1)      # right
            
            self._mask_cache[key] = mask
        return self._mask_cache[key]
        
    def get_dynamic_lens_mask(self, yuv_merged, scale=1, cam_id=None):
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
        
        # 3. Morphological cleanup: merge field across white lines (open is redundant since we filter by massive area)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        
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
        
        polygon_points = []
        poly_coeffs = None
        
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
            max_climb = int(64 / scale)
            
            start_x = int(w * 0.1)
            end_x = int(w * 0.9)
            
            for x in range(start_x, end_x, step_x):
                col_grass = safe_zone_mask[:, x]
                green_ys = np.nonzero(col_grass)[0]
                if len(green_ys) == 0:
                    continue
                
                top_green_y = green_ys[0]
                
                # Limit scan region to max_climb above the grass to prevent climbing into background/ceiling
                scan_start = max(0, top_green_y - max_climb)
                col_wall = wall_mask[scan_start:top_green_y, x]
                rev_wall = col_wall[::-1]
                zeros = np.nonzero(rev_wall == 0)[0]
                
                climb_height = zeros[0] if len(zeros) > 0 else len(rev_wall)
                raw_points.append([x, top_green_y - climb_height])
            
            # Fit polynomial to raw_points
            if len(raw_points) > 3:
                xs = [p[0] for p in raw_points]
                ys = [p[1] for p in raw_points]
                poly_coeffs = np.polyfit(xs, ys, 2)
                
                if cam_id is not None:
                    if cam_id in self.prev_polys:
                        poly_coeffs = 0.15 * poly_coeffs + 0.85 * self.prev_polys[cam_id]
                    self.prev_polys[cam_id] = poly_coeffs
                    
        elif cam_id is not None and cam_id in self.prev_polys:
            poly_coeffs = self.prev_polys[cam_id]
            
        # Reconstruct polygon_points from poly_coeffs if available
        if poly_coeffs is not None:
            start_x = int(w * 0.1)
            end_x = int(w * 0.9)
            step_x = max(1, w // 30)
            for x in range(0, w + 1, step_x):
                eval_x = min(x, w)
                clamped_x = max(start_x, min(end_x, eval_x))
                y_val = int(np.polyval(poly_coeffs, clamped_x))
                y_val = max(0, min(h, y_val))
                polygon_points.append([eval_x, y_val])
                
        if len(polygon_points) > 0:
            horizon_poly = [(pt[0] * scale, pt[1] * scale) for pt in polygon_points]
            highest_y = int(np.min([p[1] for p in polygon_points]))
            
            # Create the Dynamic Lens Mask
            dynamic_lens_mask = np.zeros_like(green_mask)
            fill_pts = [p[:] for p in polygon_points]
            fill_pts.append([w, fill_pts[-1][1]])
            fill_pts.append([w, h])
            fill_pts.append([0, h])
            fill_pts.insert(0, [0, fill_pts[0][1]])
            pts = np.array(fill_pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(dynamic_lens_mask, [pts], 255)
            dynamic_lens_mask = cv2.bitwise_and(static_mask, dynamic_lens_mask)
        else:
            # Fallback when no polynomial is available
            horizon_poly = None
            if field_contour is not None:
                # Field Hull was found, but not enough points to fit a curve
                field_hull = cv2.convexHull(field_contour)
                highest_y = int(np.min(field_hull[:, 0, 1]))
            else:
                highest_y = int(h * self.crop_top)
            dynamic_lens_mask = static_mask.copy()
            cv2.rectangle(dynamic_lens_mask, (0, 0), (w, highest_y), 0, -1)
                
        return dynamic_lens_mask, highest_y, horizon_poly, field_contour, static_mask

    def process_yuv_frame(self, frame, skip_ball=False, skip_blue_goal=False, skip_yellow_goal=False, skip_enemies=False, cam_id=None):
        """
        Takes either:
        - A planar YUV420 array from Picamera2 (shape: H*3//2, W)
        - A standard BGR image from OpenCV (shape: H, W, 3)
        Returns a dictionary of detected objects.
        """
        import time
        t0 = time.perf_counter()
        
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

        t_convert = (time.perf_counter() - t0) * 1000.0

        # Compute the dynamic lens mask and horizon ONCE per frame
        t_mask_start = time.perf_counter()
        yuv_merged = cv2.merge([y_channel, u_channel, v_channel])
        dynamic_lens_mask, highest_y, horizon_poly, field_contour, static_lens_mask = self.get_dynamic_lens_mask(yuv_merged, scale, cam_id=cam_id)
        t_mask = (time.perf_counter() - t_mask_start) * 1000.0

        # Guard: highest_y must be a valid int (can be None on frames with no field)
        if highest_y is None:
            highest_y = int(yuv_merged.shape[0] * self.crop_top)
        highest_y = int(highest_y)

        # Calculate wall margin and extended masks (used for goals and enemies)
        wall_margin = int(50 / scale)
        ext_y = max(0, highest_y - wall_margin)

        extended_mask = dynamic_lens_mask.copy()
        # We NO LONGER bypass the dynamic mask for the top 50 pixels
        # This prevents background people (like blue shirts) from being detected as goals
        
        yuv_enemy  = yuv_merged[ext_y:, :, :]
        mask_enemy = dynamic_lens_mask[ext_y:, :]

        # Slicing strictly for ball (which exists strictly below the horizon)
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

        t_ball_start = time.perf_counter()
        ball_result = {"best_ball": None, "candidates": []}
        if not skip_ball:
            ball_result   = self._find_orange_ball(yuv_sliced, lens_mask_sliced, scale, highest_y)
        t_ball = (time.perf_counter() - t_ball_start) * 1000.0

        t_goals_start = time.perf_counter()
        blue_goal = None
        if not skip_blue_goal:
            blue_goal     = self._find_blob(yuv_enemy, self.blue_bounds,  mask_enemy, scale, ext_y)
        yellow_goal = None
        if not skip_yellow_goal:
            yellow_goal   = self._find_blob(yuv_enemy, self.yellow_bounds, mask_enemy, scale, ext_y)
        t_goals = (time.perf_counter() - t_goals_start) * 1000.0

        # Build exclusion list relative to ext_y
        exclude_rects = []
        if ball_result["best_ball"]:
            bx, by, bw, bh = ball_result["best_ball"]["bbox"]
            m = int(10 / scale)
            exclude_rects.append((
                max(0, bx // scale - m),
                max(0, (by // scale) - ext_y - m),
                bw // scale + m * 2,
                bh // scale + m * 2
            ))
        for goal in (blue_goal, yellow_goal):
            if goal:
                gx, gy, gw, gh = [int(v) for v in goal["bbox"]]
                m = int(15 / scale)
                exclude_rects.append((
                    max(0, gx // scale - m),
                    max(0, (gy // scale) - ext_y - m),
                    gw // scale + m * 2,
                    gh // scale + m * 2
                ))

        t_enemies_start = time.perf_counter()
        unknowns = []
        white_lines = []
        if not skip_enemies:
            unknowns, white_lines = self._find_unknown_blobs(
                                       yuv_enemy, dynamic_lens_mask[ext_y:, :], scale, ext_y,
                                       exclude_rects=exclude_rects,
                                       grass_mask=dynamic_lens_mask[ext_y:, :])
        t_enemies = (time.perf_counter() - t_enemies_start) * 1000.0

        t_total = (time.perf_counter() - t0) * 1000.0

        timing_dict = {
            "Conversion": t_convert,
            "Mask": t_mask,
            "Ball": t_ball,
            "Goals": t_goals,
            "Enemies": t_enemies,
            "Total": t_total
        }

        results = {
            "ball":            ball_result["best_ball"],
            "ball_candidates": ball_result["candidates"],
            "blue_goal":       blue_goal,
            "yellow_goal":     yellow_goal,
            "unknowns":        unknowns,
            "white_lines":     white_lines,
            "horizon_y":       highest_y * scale,
            "horizon_poly":    horizon_poly,
            "detection_mask":  full_mask,
            "field_contour":   scaled_field_contour,
            "timing":          timing_dict
        }
        return results

    def _find_unknown_blobs(self, yuv_merged, lens_mask, scale=1, horizon_y=0,
                            exclude_rects=None, grass_mask=None):
        """
        Detects enemy robots using the user's core strategy:
            "Everything standing on the grass, except the ball and goals, is an enemy."

        Instead of excluding pixels by color (which breaks for cyan/teal/any-colored robots),
        we only mask out:
          1. The green grass itself (background)
          2. Orange ball color (to avoid ball fragments triggering enemy detection)
          3. White field lines (thin flat lines, not robots)
        Then we SUBTRACT the specific bounding boxes of detected goals/ball (exclude_rects)
        so their exact regions don't produce false enemies.

        """
        # Identify known non-enemy colors to exclude them from the enemy mask
        known_mask = np.zeros_like(lens_mask)
        for bounds in [self.green_bounds, self.orange_bounds]:
            mask = cv2.inRange(
                yuv_merged,
                np.array([bounds.get('y_min', 0), bounds['u_min'], bounds['v_min']]),
                np.array([bounds.get('y_max', 255), bounds['u_max'], bounds['v_max']])
            )
            known_mask = cv2.bitwise_or(known_mask, mask)

        # Process white mask separately to only exclude thin lines, keeping thick white blobs (robots) as enemies
        white_mask = cv2.inRange(
            yuv_merged,
            np.array([self.white_bounds.get('y_min', 0), self.white_bounds['u_min'], self.white_bounds['v_min']]),
            np.array([self.white_bounds.get('y_max', 255), self.white_bounds['u_max'], self.white_bounds['v_max']])
        )

        # 1. Remove tiny noise dots completely
        kernel_noise = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel_noise, iterations=1)

        # 2. Extract very thick blobs (e.g. white robots). A 9x9 kernel will erase lines up to ~8px thick.
        kernel_thick = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        thick_blobs = cv2.morphologyEx(white_clean, cv2.MORPH_OPEN, kernel_thick, iterations=1)
        
        # 3. The actual field lines are what's left after removing the thick blobs
        thin_lines = cv2.subtract(white_clean, thick_blobs)
        
        # Add the identified lines to known_mask so they are ignored as enemies
        known_mask = cv2.bitwise_or(known_mask, thin_lines)

        # APPLY HORIZON MASK SO WE DON'T DETECT LINES OUTSIDE THE FIELD
        thin_lines_masked = cv2.bitwise_and(thin_lines, lens_mask)

        # Extract bounding boxes for thin white lines to display them on the HUD
        white_lines_boxes = []
        line_contours, _ = cv2.findContours(thin_lines_masked, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in line_contours:
            # Increase min area to ignore small remaining noise patches
            if cv2.contourArea(cnt) > 40 / (scale * scale):
                scaled_cnt = cnt.copy()
                scaled_cnt[:, :, 0] = scaled_cnt[:, :, 0] * scale
                scaled_cnt[:, :, 1] = (scaled_cnt[:, :, 1] + horizon_y) * scale
                M = cv2.moments(scaled_cnt)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    lx, ly, lw, lh = cv2.boundingRect(scaled_cnt)
                    cx, cy = lx + lw//2, ly + lh//2
                white_lines_boxes.append({"contour": scaled_cnt, "center": (cx, cy)})

        unknown_mask = cv2.bitwise_not(known_mask)
        unknown_mask = cv2.bitwise_and(unknown_mask, lens_mask)

        # Crop the far left and right edges (12% of width) where extreme fisheye lens distortion,
        # chromatic aberration, and background objects (door frames, walls, curtains) cause false positives.
        edge_pad = int(unknown_mask.shape[1] * 0.12)
        if edge_pad > 0:
            unknown_mask[:, :edge_pad] = 0
            unknown_mask[:, -edge_pad:] = 0

        # Suppress the exact regions of detected goals and ball so they don't
        # produce false enemy detections.
        if exclude_rects:
            for (rx, ry, rw, rh) in exclude_rects:
                x1 = max(0, rx)
                y1 = max(0, ry)
                x2 = min(unknown_mask.shape[1], rx + rw)
                y2 = min(unknown_mask.shape[0], ry + rh)
                if x2 > x1 and y2 > y1:
                    unknown_mask[y1:y2, x1:x2] = 0

        # Morphological CLOSE to fill holes and unify fragmented robot pieces
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        unknown_mask = cv2.morphologyEx(unknown_mask, cv2.MORPH_CLOSE, kernel)

        # Get contours
        contours, _ = cv2.findContours(unknown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        unknown_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Adjusted area limits (scaled for downsampled UV)
            min_area = self.min_enemy_area / (scale * scale)
            max_area = 50000 / (scale * scale) # roughly 200,000px at full scale
            if area < min_area or area > max_area:
                continue
                
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Shape Ratio Filter: Reject extremely wide lines or extremely tall wires.
            # Relaxed to 4.0 to allow wide robots/boxes.
            if w > h * 4.0 or h > w * 4.0:
                continue
                
            # Edge Density Shadow Rejection (Notebook LM algorithm)
            # 3D physical cylinders have high internal edge density (wheels, seams, glare).
            # 2D flat shadows and black walls have virtually zero internal edges.
            y_patch = yuv_merged[y:y+h, x:x+w, 0]
            mask_patch = unknown_mask[y:y+h, x:x+w]
            if y_patch.size > 0 and np.count_nonzero(mask_patch) > 0:
                # Only analyze the pixels that actually belong to the blob
                valid_pixels = np.count_nonzero(mask_patch)
                mean_luma = np.sum(cv2.bitwise_and(y_patch, y_patch, mask=mask_patch)) / valid_pixels
                edges = cv2.Canny(y_patch, 30, 100)
                edges = cv2.bitwise_and(edges, edges, mask=mask_patch)
                edge_density = np.count_nonzero(edges) / valid_pixels
                
                # Extent (Fill Ratio) filter: reject C-shaped blobs that wrap around excluded regions
                extent = area / (w * h)
                
                # Shadows and walls are dark and uniform. Also reject very sparse blobs.
                if (edge_density < 0.1 and mean_luma < 130) or extent < 0.2:
                    continue
            
            # Must-Touch-Field: A real enemy robot stands ON the green field.
            # If the blob touches the very bottom of the screen, it's our own chassis! Reject!
            foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
            if foot_y_start >= yuv_merged.shape[0] - int(5 / scale):
                continue

            # If the blob is cut off by the static mask boundary (bottom crop or corner wedges),
            # it is our own chassis. We check if the lens_mask is 0 just below the foot of the blob.
            check_y = min(y + h + int(3 / scale), lens_mask.shape[0] - 1)
            sample_xs = [x + int(w * pct) for pct in [0.25, 0.5, 0.75]]
            mask_samples = [lens_mask[check_y, sx] for sx in sample_xs if 0 <= sx < lens_mask.shape[1]]
            if len(mask_samples) > 0 and np.mean(mask_samples) < 127:
                continue
                
            # Check for green NEAR the blob (below AND sides).
            # Enemies on the black wall boundary need 40px scan to reach the grass
            # through the ~20-30px thick black rubber border.
            check_below = max(5, int(40 / scale))
            foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
            foot_y_end   = min(y + h + check_below, yuv_merged.shape[0])

            found_green = True

            # 1. Look directly below the blob foot
            if foot_y_end > foot_y_start and w > 0:
                # Green in YUV: U around 128, V between 0-100 (negative V = green)
                # Let's use the actual self.green_bounds
                green_below = cv2.inRange(
                    yuv_merged[foot_y_start:foot_y_end, x:x+w],
                    np.array([self.green_bounds.get('y_min', 0), self.green_bounds['u_min'], self.green_bounds['v_min']]),
                    np.array([self.green_bounds.get('y_max', 255), self.green_bounds['u_max'], self.green_bounds['v_max']])
                )
                if cv2.countNonZero(green_below) > (w * 2): # At least 2 lines of green
                    found_green = True

            # 2. Look at flanks (left and right of the blob's bottom half)
            if not found_green and h > 10:
                y_mid = y + h // 2
                flank_y_end = min(y + h, yuv_merged.shape[0])
                
                # Left flank
                lx_start = max(0, x - int(10 / scale))
                if x > lx_start:
                    green_left = cv2.inRange(
                        yuv_merged[y_mid:flank_y_end, lx_start:x],
                        np.array([self.green_bounds.get('y_min', 0), self.green_bounds['u_min'], self.green_bounds['v_min']]),
                        np.array([self.green_bounds.get('y_max', 255), self.green_bounds['u_max'], self.green_bounds['v_max']])
                    )
                    if cv2.countNonZero(green_left) > 10:
                        found_green = True
                        
                # Right flank
                rx_end = min(yuv_merged.shape[1], x + w + int(10 / scale))
                if not found_green and rx_end > x + w:
                    green_right = cv2.inRange(
                        yuv_merged[y_mid:flank_y_end, x+w:rx_end],
                        np.array([self.green_bounds.get('y_min', 0), self.green_bounds['u_min'], self.green_bounds['v_min']]),
                        np.array([self.green_bounds.get('y_max', 255), self.green_bounds['u_max'], self.green_bounds['v_max']])
                    )
                    if cv2.countNonZero(green_right) > 10:
                        found_green = True

            if not found_green:
                continue  # Not near the field -- reject
                
            # Add the physical slice offset back
            y += horizon_y
            
            unknown_boxes.append((x * scale, y * scale, w * scale, h * scale))
            
            # Limit to top 2 biggest enemies to save downstream processing
            if len(unknown_boxes) >= 2:
                break
            
        return unknown_boxes, white_lines_boxes
        
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
            x, y, w, h = cv2.boundingRect(cnt)
            effective_area = max(area, w * h)
            min_area = self.min_ball_area / (scale * scale)
            if effective_area < min_area:
                continue
                
            # Reject if the blob touches the very bottom of the screen (our own chassis!)
            foot_y_start = min(y + h, yuv_merged.shape[0] - 1)
            if foot_y_start >= yuv_merged.shape[0] - int(5 / scale):
                continue
                
            # Use moments for center of larger blobs, fallback to bbox center for tiny blobs
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2
                
            candidate = {
                "bbox": (int(x * scale), int((y + horizon_y) * scale), int(w * scale), int(h * scale)),
                "center": (int(cx * scale), int((cy + horizon_y) * scale)),
                "area": effective_area
            }
            
            # Always apply geometry filters (aspect ratio and tiered circularity) to non-tiny blobs
            if effective_area > 5:
                if w > h * 1.6 or h > w * 1.6:
                    continue
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0:
                    continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Determine minimum required circularity based on blob size
                if effective_area > 50:
                    min_circ = 0.50
                elif effective_area > 20:
                    min_circ = 0.40
                elif effective_area > 10:
                    min_circ = 0.30
                else:
                    min_circ = 0.15
                    
                if circularity < min_circ:
                    continue
            
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
            
            candidates.append(candidate)
            if effective_area > max_area:
                max_area = effective_area
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
            if area < 500 / (scale * scale):  # Goals should be large (scaled dynamically)
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
