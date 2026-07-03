import time
import math as m
import cv2
import numpy as np
from vision_config import config

class CVKalmanFilter:
    def __init__(self, timeout=1.5):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        
        self.last_update_time = 0
        self.timeout = timeout
        self.initialized = False
        self.last_size = 0  # Store apparent size for distance calculation when coasting
        
    def update(self, x, y, size):
        current_time = time.perf_counter()
        dt = current_time - self.last_update_time if self.initialized else 0.033
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        self.last_size = size
        
        meas = np.array([[x], [y]], dtype=np.float32)
        
        if not self.initialized:
            self.kf.statePre = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.kf.statePost = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
            self.initialized = True
            self.last_update_time = current_time
            return x, y, 0.0, 0.0, size
            
        self.kf.correct(meas)
        pred = self.kf.predict()
        self.last_update_time = current_time
        return float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]), size
        
    def predict(self):
        current_time = time.perf_counter()
        if not self.initialized or (current_time - self.last_update_time > self.timeout):
            self.initialized = False
            return None, None, 0.0, 0.0, 0
            
        dt = current_time - self.last_update_time
        self.kf.transitionMatrix[0, 2] = dt
        self.kf.transitionMatrix[1, 3] = dt
        pred = self.kf.predict()
        self.last_update_time = current_time
        return float(pred[0]), float(pred[1]), float(pred[2]), float(pred[3]), self.last_size

class Inferencia:
    def __init__(self, fps=30):
        # Load Configuration
        self.FOCAL_LENGTH_EQ = config["camera"]["focal_length_eq"]
        self.EDGE_CORRECTION = config["camera"]["edge_correction"]
        self.FRAME_WIDTH = config["camera"]["image_cx"] * 2
        self.FRAME_HEIGHT = config["camera"]["image_cy"] * 2
        
        self.BALL_SIZE_MM = config["physics"]["ball_diameter_mm"]
        self.ROBOT_SIZE_MM = config["physics"]["robot_size_mm"]
        self.GOAL_WIDTH_MM = config["physics"]["goal_width_mm"]
        
        self.MAX_DIST = config["physics"]["max_distance_mm"]
        self.MAX_VEL = config["physics"]["max_velocity_px_s"]
        self.FIELD_LENGTH_MM = config.get("physics", {}).get("field_length_mm", 2400.0)
        self.TEAM_COLOR = config.get("physics", {}).get("team_color", "blue")
        TIMEOUT = config.get("physics", {}).get("kalman_timeout_sec", 1.5)
        
        self.ball_kf_0 = CVKalmanFilter(timeout=TIMEOUT)
        self.ball_kf_1 = CVKalmanFilter(timeout=TIMEOUT)
        self.enemy_kf_0 = CVKalmanFilter(timeout=TIMEOUT)
        self.enemy_kf_1 = CVKalmanFilter(timeout=TIMEOUT)
        
        self.last_time = time.perf_counter()
        
        # Variables to track physical ball speed in m/s
        self.last_ball_x_m = None
        self.last_ball_y_m = None
        self.last_ball_time = time.perf_counter()
        self.filtered_ball_speed = 0.0
        
    def _get_distance(self, apparent_size_px, known_size_mm, center_x, center_y):
        """
        Calculate distance using the Equidistant Fisheye Model.
        Includes a radial correction factor for extreme edges.
        """
        if apparent_size_px <= 0:
            return -1
            
        # Standard equidistant formula
        angular_size = apparent_size_px / self.FOCAL_LENGTH_EQ
        half_angle = angular_size / 2.0
        if half_angle >= m.pi / 2:
            return -1
            
        base_distance = known_size_mm / (2.0 * m.tan(half_angle))
        
        # Calculate radial distance from center of lens
        dx = center_x - (self.FRAME_WIDTH / 2.0)
        dy = center_y - (self.FRAME_HEIGHT / 2.0)
        r = m.sqrt(dx*dx + dy*dy)
        max_r = m.sqrt((self.FRAME_WIDTH/2)**2 + (self.FRAME_HEIGHT/2)**2)
        
        # Apply radial edge correction
        correction = 1.0 + (self.EDGE_CORRECTION * (r / max_r)**2)
        
        return base_distance * correction

    def _calculate_velocity(self, vx, vy):
        """Returns raw velocity magnitude from Kalman vector."""
        vel_px_s = m.sqrt(vx*vx + vy*vy)
        return vel_px_s

    def _localize_ego(self, state):
        """Triangulate absolute X coordinate using distance to both goals."""
        d_blue = state.get("blue_goal_distance_m", -1)
        if d_blue > 0: d_blue *= 1000.0
        d_yellow = state.get("yellow_goal_distance_m", -1)
        if d_yellow > 0: d_yellow *= 1000.0
        
        if d_blue > 0 and d_yellow > 0:
            L = self.FIELD_LENGTH_MM
            # Geometric intersection of two circles on the X axis:
            # (x - L)^2 + y^2 = d_blue^2
            # x^2 + y^2 = d_yellow^2
            # x = (L^2 - d_blue^2 + d_yellow^2) / (2L)
            x = (L*L - d_blue*d_blue + d_yellow*d_yellow) / (2 * L)
            state["ego_x"] = x
            
            y_sq = d_yellow*d_yellow - x*x
            state["ego_y_abs"] = m.sqrt(y_sq) if y_sq > 0 else 0.0
            state["ego_pose_confidence"] = 1.0
            
            # HEURISTICA VISUAL: Riesgo de salir del campo (Boundary Risk)
            # El campo mide L. Si estamos a menos de 300mm (30cm) de un borde, el riesgo sube.
            margin_mm = 300.0
            risk_x = 0.0
            if x < margin_mm:
                risk_x = 1.0 - (max(0, x) / margin_mm)
            elif (L - x) < margin_mm:
                risk_x = 1.0 - (max(0, L - x) / margin_mm)
                
            state["near_boundary_risk"] = min(1.0, risk_x)
            
        elif d_blue > 0 or d_yellow > 0:
            state["ego_pose_confidence"] = 0.5
        else:
            state["ego_pose_confidence"] = 0.0

    def _line_intersects_rect(self, p1, p2, rect):
        """Checks if line segment p1->p2 intersects a bounding box."""
        x, y, w, h = rect
        rect_pts = [
            (x, y), (x+w, y),
            (x+w, y+h), (x, y+h)
        ]
        
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
            
        def intersect(A, B, C, D):
            return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)
            
        # Check all 4 edges of the rectangle
        for i in range(4):
            if intersect(p1, p2, rect_pts[i], rect_pts[(i+1)%4]):
                return True
        return False

    def update(self, detections, threats, telemetry=None):
        """
        Takes raw vision outputs and computes physics variables for the NN.
        Returns a dictionary of normalized values exactly matching the ABIDE schema.
        """
        self.TEAM_COLOR = config.get("physics", {}).get("team_color", "blue")
        
        state = {
            "ball_possession_ego": 0.0,
            "ball_possession_ally": 0.0,
            "ball_distance_norm": 0.0,
            "ball_speed_norm": 0.0,
            "shot_opportunity_ego": 0.0,
            "ball_direction_alignment": 0.5,
            "ego_speed_norm": 0.0,
            "ego_accel_norm": 0.0,
            "slip_indicator": 0.0,
            "ego_pose_confidence": 0.0,
            "ego_x": 0.0,
            "ego_y_abs": 0.0,
            "yaw_rate_norm": 0.0,
            "near_boundary_risk": 0.0,
            "ally_distance_norm": 0.0,
            "hardware_fault_indicator": 0.0,
            "enemy1_distance_norm": 0.0,
            "enemy1_velocity_norm": 0.0,
            "enemy_blocking_lane": 0.0,
            "enemy_pressure_level": 0.0,
            "enemy_observation_conf": 0.0
        }
        
        if telemetry:
            STALL_THRESHOLD = 3.0
            vx, vy = telemetry.get("vel_x", 0.0), telemetry.get("vel_y", 0.0)
            state["ego_speed_norm"] = m.sqrt(vx*vx + vy*vy)
            ax, ay = telemetry.get("accel_x", 0.0), telemetry.get("accel_y", 0.0)
            state["ego_accel_norm"] = m.sqrt(ax*ax + ay*ay)
            state["yaw_rate_norm"] = abs(telemetry.get("gyro_z", 0.0))
            if any([telemetry.get(k, 0.0) > STALL_THRESHOLD for k in ["current_fl", "current_fr", "current_rl"]]):
                state["hardware_fault_indicator"] = 1.0
        
        # ── Ego Pose Confidence ──
        if detections.get("blue_goal") or detections.get("yellow_goal"):
            state["ego_pose_confidence"] = 1.0
            
        # ── Ally Distance ──
        # (Removed as there are no allies)
        
        # ── 1. Ball Physics ──
        ball_data = detections.get("ball")
        ball_cam = detections.get("ball_cam", 0)
        ball_kf = self.ball_kf_1 if ball_cam == 1 else self.ball_kf_0
        
        if ball_data:
            bx, by, bw, bh = ball_data["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            size = max(bw, bh)
            kx, ky, kvx, kvy, _ = ball_kf.update(cx, cy, size)
        else:
            kx, ky, kvx, kvy, size = ball_kf.predict()

        ball_geom = None
        if kx is not None:
            dist_mm = self._get_distance(size, self.BALL_SIZE_MM, kx, ky)
            state["ball_distance_norm"] = dist_mm / 1000.0
            dx = kx - (self.FRAME_WIDTH / 2.0)
            theta_rad = dx / self.FOCAL_LENGTH_EQ
            theta_deg = m.degrees(theta_rad)
            
            if ball_cam == 1:
                theta_deg = -180.0 + theta_deg
                if theta_deg < -180.0: theta_deg += 360.0
            
            state["ball_angle_deg"] = theta_deg
            state["ball_direction_alignment"] = (theta_deg + 180.0) / 360.0
            
            # Calculate physical speed in meters per second
            dist_m = state["ball_distance_norm"]
            angle_rad = m.radians(theta_deg)
            current_x_m = dist_m * m.cos(angle_rad)
            current_y_m = dist_m * m.sin(angle_rad)
            
            current_time = time.perf_counter()
            dt = current_time - self.last_ball_time
            
            if self.last_ball_x_m is not None and dt > 0.01:
                dx_m = current_x_m - self.last_ball_x_m
                dy_m = current_y_m - self.last_ball_y_m
                speed_m_s = m.sqrt(dx_m*dx_m + dy_m*dy_m) / dt
                # Simple Low-Pass Filter to smooth out jitter
                self.filtered_ball_speed = 0.8 * self.filtered_ball_speed + 0.2 * speed_m_s
                state["ball_speed_norm"] = self.filtered_ball_speed
            else:
                state["ball_speed_norm"] = self.filtered_ball_speed
                
            self.last_ball_x_m = current_x_m
            self.last_ball_y_m = current_y_m
            self.last_ball_time = current_time
            
            ball_geom = (kx, ky)
            
            if state["ball_distance_norm"] < 0.24 and 0.35 < state["ball_direction_alignment"] < 0.65:
                state["ball_possession_ego"] = 1.0

        # ── 2. Enemy Physics ──
        threats.sort(key=lambda t: t["bbox"][2]*t["bbox"][3], reverse=True)
        
        enemy_cam_id = 0
        if len(threats) > 0:
            ex, ey, ew, eh = threats[0]["bbox"]
            enemy_cam_id = threats[0]["cam_id"]
            cx, cy = ex+ew/2, ey+eh/2
            enemy_kf = self.enemy_kf_1 if enemy_cam_id == 1 else self.enemy_kf_0
            ekx, eky, ekvx, ekvy, _ = enemy_kf.update(cx, cy, ew)
        else:
            ekx, eky, ekvx, ekvy, ew = self.enemy_kf_0.predict()
            
        if ekx is not None:
            dist_mm = self._get_distance(ew, self.ROBOT_SIZE_MM, ekx, eky)
            state["enemy1_distance_norm"] = dist_mm / 1000.0
            state["enemy1_velocity_norm"] = self._calculate_velocity(ekvx, ekvy)
            state["enemy_observation_conf"] = 1.0 
            
            # Simple pressure level heuristic
            threshold_m = 0.3 * (self.MAX_DIST / 1000.0)
            if state["enemy1_distance_norm"] < threshold_m:
                state["enemy_pressure_level"] = 1.0 - (state["enemy1_distance_norm"] / threshold_m)
        else:
            state["enemy_observation_conf"] = 0.0

        # ── 3. Goals & Shot Opportunity ──
        # Find which goal is our target based on config
        target_key = "blue_goal" if self.TEAM_COLOR == "blue" else "yellow_goal"
        target_goal_pt = None
        
        if detections.get("blue_goal"):
            blue_cam = detections.get("blue_goal_cam", 0)
            bx, by, bw, bh = detections["blue_goal"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            if target_key == "blue_goal":
                target_goal_pt = (cx, by + bh) 
            
            dist_mm = self._get_distance(bw, self.GOAL_WIDTH_MM, cx, cy)
            state["blue_goal_distance_m"] = dist_mm / 1000.0
            dx = cx - (self.FRAME_WIDTH / 2.0)
            ang = m.degrees(dx / self.FOCAL_LENGTH_EQ)
            if blue_cam == 1:
                ang = -180.0 + ang
                if ang < -180.0: ang += 360.0
            state["blue_goal_angle_deg"] = ang
            
        if detections.get("yellow_goal"):
            yellow_cam = detections.get("yellow_goal_cam", 0)
            bx, by, bw, bh = detections["yellow_goal"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            if target_key == "yellow_goal":
                target_goal_pt = (cx, by + bh)
            
            dist_mm = self._get_distance(bw, self.GOAL_WIDTH_MM, cx, cy)
            state["yellow_goal_distance_m"] = dist_mm / 1000.0
            dx = cx - (self.FRAME_WIDTH / 2.0)
            ang = m.degrees(dx / self.FOCAL_LENGTH_EQ)
            if yellow_cam == 1:
                ang = -180.0 + ang
                if ang < -180.0: ang += 360.0
            state["yellow_goal_angle_deg"] = ang
            
        self._localize_ego(state)
            
        if target_goal_pt and ball_geom:
            shot_blocked = False
            for threat in threats:
                if self._line_intersects_rect(ball_geom, target_goal_pt, threat["bbox"]):
                    shot_blocked = True
                    break
            
            state["shot_opportunity_ego"] = 0.0 if shot_blocked else 1.0
            state["enemy_blocking_lane"] = 1.0 if shot_blocked else 0.0

        # HEURISTICA VISUAL: Posesión del Aliado (Ball Possession Ally)
        # (Removed as there are no allies)

        # ── 4. Heading / Orientation (Reemplaza IMU accel) ──
        current_heading = None
        if detections.get("yellow_goal"):
            y_ang = state.get("yellow_goal_angle_deg", 0.0)
            current_heading = -y_ang
        elif detections.get("blue_goal"):
            b_ang = state.get("blue_goal_angle_deg", 0.0)
            current_heading = 180.0 - b_ang
            
        if current_heading is not None:
            while current_heading > 180.0: current_heading -= 360.0
            while current_heading <= -180.0: current_heading += 360.0
            
            if not hasattr(self, "filtered_heading"):
                self.filtered_heading = current_heading
            else:
                diff = current_heading - self.filtered_heading
                while diff > 180.0: diff -= 360.0
                while diff <= -180.0: diff += 360.0
                self.filtered_heading += 0.2 * diff
                while self.filtered_heading > 180.0: self.filtered_heading -= 360.0
                while self.filtered_heading <= -180.0: self.filtered_heading += 360.0
        elif not hasattr(self, "filtered_heading"):
            self.filtered_heading = 0.0
            
        # Map [-180, 180] to [0, 1]
        state["ego_accel_norm"] = (self.filtered_heading + 180.0) / 360.0

        return state
