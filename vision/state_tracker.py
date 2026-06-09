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

class StateTracker:
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
        
        self.ball_kf = CVKalmanFilter(timeout=TIMEOUT)
        self.enemy_kf = CVKalmanFilter(timeout=TIMEOUT)
        
        self.last_time = time.perf_counter()
        
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
        """Returns normalized velocity magnitude from Kalman vector."""
        vel_px_s = m.sqrt(vx*vx + vy*vy)
        return min(1.0, vel_px_s / self.MAX_VEL)

    def _localize_ego(self, state):
        """Triangulate absolute X coordinate using distance to both goals."""
        d_blue = state.get("blue_goal_distance_mm", -1)
        d_yellow = state.get("yellow_goal_distance_mm", -1)
        
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

    def update(self, detections, threats, aruco_positions=None, telemetry=None):
        """
        Takes raw vision outputs and computes physics variables for the NN.
        Returns a dictionary of normalized values exactly matching the ABIDE schema.
        """
        current_time = time.perf_counter()
        
        # Initialize the exact flat dictionary expected by the Neural Network
        state = {
            # -- NET A: The Ball --
            "ball_possession_ego": 0.0,    # Default (Requires ESP32 IR Sensor)
            "ball_possession_ally": 0.0,   # Default (Requires ESP-NOW)
            "ball_distance_norm": 0.0,     
            "ball_speed_norm": 0.0,        
            "shot_opportunity_ego": 0.0,   
            "ball_direction_alignment": 0.5, 
            
            # -- NET B: Self State --
            "ego_speed_norm": 0.0,         # Default (Requires ESP32 Encoders)
            "ego_accel_norm": 0.0,         # Default (Requires ESP32 IMU)
            "slip_indicator": 0.0,         # Default (Requires IMU/Encoders)
            "ego_pose_confidence": 0.0,    
            "yaw_rate_norm": 0.0,          # Default (Requires IMU)
            "near_boundary_risk": 0.0,     # Default (Requires Line Sensors)
            "ally_distance_norm": 0.0,     
            "hardware_fault_indicator": 0.0, # Default (Requires Battery Sense)
            
            # -- NET C: The Enemy --
            "enemy1_distance_norm": 0.0,   
            "enemy1_velocity_norm": 0.0,   
            "enemy_blocking_lane": 0.0,    
            "enemy_pressure_level": 0.0,   # High if we have ball and enemy is close
            "enemy_observation_conf": 0.0  
        }
        
        if telemetry:
            MAX_SPEED = 2.0
            MAX_ACCEL = 10.0
            MAX_GYRO = 10.0
            STALL_THRESHOLD = 3.0
            
            vx, vy = telemetry.get("vel_x", 0.0), telemetry.get("vel_y", 0.0)
            state["ego_speed_norm"] = min(1.0, m.sqrt(vx*vx + vy*vy) / MAX_SPEED)
            
            ax, ay = telemetry.get("accel_x", 0.0), telemetry.get("accel_y", 0.0)
            state["ego_accel_norm"] = min(1.0, m.sqrt(ax*ax + ay*ay) / MAX_ACCEL)
            
            state["yaw_rate_norm"] = min(1.0, abs(telemetry.get("gyro_z", 0.0)) / MAX_GYRO)
            
            c_fl = telemetry.get("current_fl", 0.0)
            c_fr = telemetry.get("current_fr", 0.0)
            c_rl = telemetry.get("current_rl", 0.0)
            if c_fl > STALL_THRESHOLD or c_fr > STALL_THRESHOLD or c_rl > STALL_THRESHOLD:
                state["hardware_fault_indicator"] = 1.0
            else:
                state["hardware_fault_indicator"] = 0.0
        
        ego_pos = (self.FRAME_WIDTH // 2, self.FRAME_HEIGHT) # Bottom center
        
        # ── Ego Pose Confidence ──
        # 1.0 if we can see any goals or our ally, meaning we aren't completely blind.
        if detections["blue_goal"] or detections["yellow_goal"] or (aruco_positions and len(aruco_positions) > 0):
            state["ego_pose_confidence"] = 1.0
            
        # ── Ally Distance ──
        if aruco_positions and len(aruco_positions) > 0:
            # Get the first ally
            for marker_id, data in aruco_positions.items():
                cx, cy, apparent_size = data
                dist_mm = self._get_distance(apparent_size, 50, cx, cy) # 50mm ArUco
                state["ally_distance_norm"] = min(1.0, dist_mm / self.MAX_DIST)
                break
        
        # ── 1. Ball Physics ──
        if detections["ball"]:
            bx, by, bw, bh = detections["ball"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            size = max(bw, bh)
            kx, ky, kvx, kvy, _ = self.ball_kf.update(cx, cy, size)
        else:
            kx, ky, kvx, kvy, size = self.ball_kf.predict()

        if kx is not None:
            dist_mm = self._get_distance(size, self.BALL_SIZE_MM, kx, ky)
            state["ball_distance_norm"] = min(1.0, dist_mm / self.MAX_DIST) 
            state["ball_direction_alignment"] = kx / self.FRAME_WIDTH
            
            # Calculate precise angle using fisheye equidistant projection (r = f * theta)
            dx = kx - (self.FRAME_WIDTH / 2.0)
            theta_rad = dx / self.FOCAL_LENGTH_EQ
            state["ball_angle_deg"] = m.degrees(theta_rad)
            
            state["ball_speed_norm"] = self._calculate_velocity(kvx, kvy)
            # Store smoothed coords for geometry later
            ball_geom = (kx, ky)
        else:
            ball_geom = None

        # ── 2. Enemy Physics ──
        threats.sort(key=lambda t: t[2]*t[3], reverse=True)
        
        if len(threats) > 0:
            ex, ey, ew, eh = threats[0]
            cx, cy = ex+ew/2, ey+eh/2
            kx, ky, kvx, kvy, _ = self.enemy_kf.update(cx, cy, ew)
        else:
            kx, ky, kvx, kvy, ew = self.enemy_kf.predict()
            
        if kx is not None:
            dist_mm = self._get_distance(ew, self.ROBOT_SIZE_MM, kx, ky)
            state["enemy1_distance_norm"] = min(1.0, dist_mm / self.MAX_DIST)
            state["enemy1_velocity_norm"] = self._calculate_velocity(kvx, kvy)
            state["enemy_observation_conf"] = 1.0 
            
            # Simple pressure level heuristic
            if state["enemy1_distance_norm"] < 0.3:
                state["enemy_pressure_level"] = 1.0 - (state["enemy1_distance_norm"] / 0.3)
        else:
            state["enemy_observation_conf"] = 0.0

        # ── 3. Goals & Shot Opportunity ──
        # Find which goal is our target based on config
        target_key = "blue_goal" if self.TEAM_COLOR == "blue" else "yellow_goal"
        target_goal_pt = None
        
        if detections.get("blue_goal"):
            bx, by, bw, bh = detections["blue_goal"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            if target_key == "blue_goal":
                target_goal_pt = (cx, by + bh) 
            
            dist_mm = self._get_distance(bw, self.GOAL_WIDTH_MM, cx, cy)
            state["blue_goal_distance_mm"] = dist_mm
            dx = cx - (self.FRAME_WIDTH / 2.0)
            state["blue_goal_angle_deg"] = m.degrees(dx / self.FOCAL_LENGTH_EQ)
            
        if detections.get("yellow_goal"):
            bx, by, bw, bh = detections["yellow_goal"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            if target_key == "yellow_goal":
                target_goal_pt = (cx, by + bh)
            
            dist_mm = self._get_distance(bw, self.GOAL_WIDTH_MM, cx, cy)
            state["yellow_goal_distance_mm"] = dist_mm
            dx = cx - (self.FRAME_WIDTH / 2.0)
            state["yellow_goal_angle_deg"] = m.degrees(dx / self.FOCAL_LENGTH_EQ)
            
        self._localize_ego(state)
            
        if target_goal_pt and ball_geom:
            shot_blocked = False
            for threat in threats:
                if self._line_intersects_rect(ball_geom, target_goal_pt, threat):
                    shot_blocked = True
                    break
            
            state["shot_opportunity_ego"] = 0.0 if shot_blocked else 1.0
            state["enemy_blocking_lane"] = 1.0 if shot_blocked else 0.0

        return state
