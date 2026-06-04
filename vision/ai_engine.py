import os
import numpy as np
import onnxruntime as ort
from collections import deque
import time

class AIEngine:
    def __init__(self, models_dir="../ONNX_models"):
        # Configure ONNX runtime
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        
        # Load ONNX sessions
        # Provide fallback if files don't exist yet for testing without them crashing initialization
        self.enabled = True
        self.models = {}
        
        required_models = {
            "net_a": "NET_A_BALL_ENCODER.onnx",
            "net_b": "NET_B_SELF_ENCODER.onnx",
            "net_c": "NET_C_ENEMY_ENCODER.onnx",
            "net_t": "NET_T_CONTEXT_ENCODER.onnx",
            "enthralled": "AADFBS_ENTHRALLED.onnx"
        }
        
        for name, filename in required_models.items():
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                self.models[name] = ort.InferenceSession(path, options)
            else:
                print(f"[WARN] Missing model: {path}. AI Engine will run in dummy mode.")
                self.enabled = False
        
        # History for NET_T (requires 20 frames)
        self.history = deque(maxlen=20)
        self.embed_dim = 12

    def infer(self, state) -> dict:
        if not self.enabled:
            return {"vx": 0.0, "vy": 0.0, "omega": 0.0, "kick": 0.0, "urgency": 0.0, "aggression": 0.0, "defense": 0.0, "pass_pref": 0.0, "emergency": 0.0}, {}
            
        # 1. Map state dictionary to exact arrays for A, B, C
        # Calculate derived threat
        e1_threat = 0.0
        if state["enemy_observation_conf"] > 0:
            e1_threat = max(0.0, 1.0 - abs(state["enemy1_distance_norm"] - state["ball_distance_norm"]))
        ball_free = max(0.0, 1.0 - max(state["ball_possession_ego"], state["ball_possession_ally"], e1_threat))

        net_a_in = np.array([[
            state["ball_possession_ego"],
            state["ball_possession_ally"],
            e1_threat,
            0.0, # enemy2 threat
            state["ball_distance_norm"],
            state["ball_speed_norm"],
            state["shot_opportunity_ego"],
            0.0, # pass opportunity
            state["ball_direction_alignment"],
            ball_free
        ]], dtype=np.float32)

        net_b_in = np.array([[
            state["ego_speed_norm"],
            state["ego_accel_norm"],
            1.0 - state["slip_indicator"], # velocity stability
            state["ego_pose_confidence"],
            state["yaw_rate_norm"],
            1.0 - state["slip_indicator"], # angular stability
            state["slip_indicator"],
            1.0, # field zone conf
            state["near_boundary_risk"],
            state["ally_distance_norm"],
            0.5, # ally bearing
            1.0 if state["ally_distance_norm"] > 0 else 0.0, # ally conf
            max(0.0, 1.0 - state["enemy1_distance_norm"]), # free space ahead
            1.0 - state["ego_pose_confidence"] # occlusion
        ]], dtype=np.float32)

        net_c_in = np.array([[
            state["enemy1_distance_norm"],
            1.0, # enemy2 dist
            state["enemy1_velocity_norm"],
            0.0, # enemy2 vel
            e1_threat, # enemy1 ball alignment
            0.0, # enemy2 ball alignment
            state["enemy_blocking_lane"],
            0.0, # enemy2 block
            0.0, # enemy1 goal
            0.0, # enemy2 goal
            state["enemy_pressure_level"],
            state["enemy_observation_conf"]
        ]], dtype=np.float32)

        # 2. Run Encoders
        a_in_name = self.models["net_a"].get_inputs()[0].name
        b_in_name = self.models["net_b"].get_inputs()[0].name
        c_in_name = self.models["net_c"].get_inputs()[0].name
        
        a_emb = self.models["net_a"].run(None, {a_in_name: net_a_in})[0][0] # 4 dims
        b_emb = self.models["net_b"].run(None, {b_in_name: net_b_in})[0][0] # 4 dims
        c_emb = self.models["net_c"].run(None, {c_in_name: net_c_in})[0][0] # 4 dims

        # 3. Form spatial embedding & update history
        spatial_emb = np.concatenate([a_emb, b_emb, c_emb]) # 12 dims
        self.history.append(spatial_emb)

        # Fill history if booting up
        while len(self.history) < 20:
            self.history.append(spatial_emb)

        net_t_in = np.array([list(self.history)], dtype=np.float32) # (1, 20, 12)

        # 4. Run Temporal Context
        t_in_name = self.models["net_t"].get_inputs()[0].name
        t_emb = self.models["net_t"].run(None, {t_in_name: net_t_in})[0][0] # 5 dims

        # 5. Form ENTHRALLED input
        enthralled_in = np.concatenate([a_emb, b_emb, c_emb, t_emb]).astype(np.float32).reshape(1, -1) # 17 dims

        # 6. Run Decision Engine
        e_in_name = self.models["enthralled"].get_inputs()[0].name
        action = self.models["enthralled"].run(None, {e_in_name: enthralled_in})[0][0] # 9 dims

        action_dict = {
            "vx": float(action[0]),
            "vy": float(action[1]),
            "omega": float(action[2]),
            "kick": float(action[3]),
            "urgency": float(action[4]),
            "aggression": float(action[5]),
            "defense": float(action[6]),
            "pass_pref": float(action[7]),
            "emergency": float(action[8])
        }
        
        debug_info = {
            "NET_A_emb": a_emb.tolist(),
            "NET_B_emb": b_emb.tolist(),
            "NET_C_emb": c_emb.tolist(),
            "NET_T_emb": t_emb.tolist(),
            "raw_state": state
        }
        
        return action_dict, debug_info
