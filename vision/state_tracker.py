import time
import math as m
import numpy as np
from collections import deque

class StateTracker:
    def __init__(self, fps=30):
        # Configuration
        self.FOCAL_LENGTH_EQ = 470
        self.EDGE_CORRECTION = 0.0
        self.BALL_SIZE_MM = 43
        self.ROBOT_SIZE_MM = 180
        self.FRAME_WIDTH = 820
        self.FRAME_HEIGHT = 616
        
        self.history_len = 5
        self.ball_history = deque(maxlen=self.history_len)
        self.enemy_history = deque(maxlen=self.history_len)
        
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

    def _calculate_velocity(self, history):
        """Calculates pixel velocity over the recorded history window."""
        if len(history) < 2:
            return 0.0
            
        oldest = history[0]
        newest = history[-1]
        
        dt = newest['time'] - oldest['time']
        if dt <= 0:
            return 0.0
            
        dx = newest['x'] - oldest['x']
        dy = newest['y'] - oldest['y']
        dist_px = m.sqrt(dx*dx + dy*dy)
        
        speed_px_per_sec = dist_px / dt
        
        # Normalize: assuming max reasonable speed on screen is ~1000px/s
        norm_speed = min(1.0, speed_px_per_sec / 1000.0)
        return norm_speed

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
                state["ally_distance_norm"] = min(1.0, dist_mm / 3000.0)
                break
        
        # ── 1. Ball Physics ──
        if detections["ball"]:
            bx, by, bw, bh = detections["ball"]["bbox"]
            cx = bx + bw/2
            cy = by + bh/2
            
            dist_mm = self._get_distance(max(bw, bh), self.BALL_SIZE_MM, cx, cy)
            state["ball_distance_norm"] = min(1.0, dist_mm / 3000.0) 
            
            state["ball_direction_alignment"] = cx / self.FRAME_WIDTH
            
            self.ball_history.append({'x': cx, 'y': cy, 'time': current_time})
            state["ball_speed_norm"] = self._calculate_velocity(self.ball_history)
        else:
            self.ball_history.clear()

        # ── 2. Enemy Physics ──
        threats.sort(key=lambda t: t[2]*t[3], reverse=True)
        
        if len(threats) > 0:
            ex, ey, ew, eh = threats[0]
            dist_mm = self._get_distance(ew, self.ROBOT_SIZE_MM, ex+ew/2, ey+eh/2)
            state["enemy1_distance_norm"] = min(1.0, dist_mm / 3000.0)
            
            self.enemy_history.append({'x': ex+ew/2, 'y': ey+eh/2, 'time': current_time})
            state["enemy1_velocity_norm"] = self._calculate_velocity(self.enemy_history)
            
            # If the enemy is visible and stable
            state["enemy_observation_conf"] = 1.0 
            
            # Simple pressure level heuristic: If enemy is very close (<0.3 norm dist), pressure is high
            if state["enemy1_distance_norm"] < 0.3:
                state["enemy_pressure_level"] = 1.0 - (state["enemy1_distance_norm"] / 0.3)
        else:
            self.enemy_history.clear()

        # ── 3. Shot Opportunity (Geometric Intersection) ──
        target_goal = None
        if detections.get("blue_goal"):
            bx, by, bw, bh = detections["blue_goal"]["bbox"]
            target_goal = (bx + bw/2, by + bh) 
            
        if target_goal and detections["ball"]:
            ball_cx = detections["ball"]["bbox"][0] + detections["ball"]["bbox"][2]/2
            ball_cy = detections["ball"]["bbox"][1] + detections["ball"]["bbox"][3]/2
            
            shot_blocked = False
            for threat in threats:
                if self._line_intersects_rect((ball_cx, ball_cy), target_goal, threat):
                    shot_blocked = True
                    break
            
            state["shot_opportunity_ego"] = 0.0 if shot_blocked else 1.0
            state["enemy_blocking_lane"] = 1.0 if shot_blocked else 0.0

        return state
