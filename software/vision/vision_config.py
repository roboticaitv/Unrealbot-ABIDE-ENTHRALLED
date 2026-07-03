"""
Centralized Configuration for the Unrealbot Vision Pipeline.
Adjust these parameters to tune color detection, camera calibration, and masking.
"""

config = {
    # ── Camera Calibrations ──
    "camera": {
        "focal_length_eq": 235,     # Equidistant focal length for IMX219 200° fisheye at 832x624 resolution (half of 470)
        "image_cx": 416,            # Optical center X (half of 832)
        "image_cy": 312,            # Optical center Y (half of 624)
        "edge_correction": 0.0      # Radial edge correction multiplier
    },
    
    # ── Real World Physical Constants ──
    "physics": {
        "field_length_mm": 2400.0,  # Field X axis (goal to goal)
        "field_width_mm": 1800.0,   # Field Y axis
        "team_color": "blue",       # Target goal to attack ("blue" or "yellow")
        "kalman_timeout_sec": 0.0,  # Max seconds to coast invisible objects
        
        "ball_diameter_mm": 43,     
        "robot_size_mm": 180,
        "goal_width_mm": 600,
        
        # Minimum pixel areas to be considered a valid detection (at full scale)
        "min_ball_area_px": 1,     # MINIMUM — catches the ball at maximum field distance
        "min_enemy_area_px": 150,  # Calibrate this: increase if random line fragments trigger false enemies
        
        # Neural Network Normalization Constants
        "max_distance_mm": 3000.0,
        "max_velocity_px_s": 1000.0,
    },
    
    # ── Masking & Cropping ──
    "masking": {
        # Instead of a single circle radius, we use an ellipse to match the actual sensor.
        # 0.95 * cx = 779px wide radius. 0.95 * cy = 585px tall radius.
        "fisheye_radius_x_pct": 0.98,
        "fisheye_radius_y_pct": 0.98,
        
        "crop_top": 0.10,           # Ignore top 10% (ceiling)
        "crop_bottom": 0.04,        # Ignore bottom 18% flat (hide the robot chassis and wires!)
        
        # Corner exclusion wedges (robot body/wires)
        # Format: (width_pct, height_pct) for the inner corner of the triangle
        "corner_wedge_width": 0.35, 
        "corner_wedge_height": 0.30 
    },
    
    # ── Color Tracking Bounds (YUV420) ──
    "colors": {
        "orange_ball": {
            "u_min": 50,  "u_max": 125,
            "v_min": 155, "v_max": 255,   # Increased from 145 to reject white line chromatic aberration
            "y_min": 20,  "y_max": 255
        },
        "red_marker": {
            "u_min": 0,   "u_max": 255,
            "v_min": 160, "v_max": 255,
            "y_min": 0,   "y_max": 255
        },
        "blue_goal": {
            "u_min": 150, "u_max": 255,
            "v_min": 60,  "v_max": 130,
            "y_min": 0,   "y_max": 255
        },
        "yellow_goal": {
            "u_min": 0,   "u_max": 110,
            "v_min": 100, "v_max": 145,
            "y_min": 0,   "y_max": 255
        },
        "green_grass": {
            "u_min": 100, "u_max": 135,
            "v_min": 10,  "v_max": 110,
            "y_min": 0,   "y_max": 255
        },
        "black_walls": {
            "u_min": 110, "u_max": 146,
            "v_min": 110, "v_max": 146,
            "y_min": 0,   "y_max": 80   # Strict low luma for deep blacks
        },
        "white_lines": {
            "u_min": 110, "u_max": 146,
            "v_min": 110, "v_max": 146,
            "y_min": 81,  "y_max": 255  # Catch shadowed lines (gray) to prevent false enemies!
        }
    }
}
