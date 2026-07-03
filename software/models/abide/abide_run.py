import os
import sys
import numpy as np
from collections import deque
import time

# Python 3.12+ hack to fix TensorFlow's broken flatbuffers dependency
if sys.version_info >= (3, 12) and 'imp' not in sys.modules:
    import types
    sys.modules['imp'] = types.ModuleType('imp')

# Try edge runtimes first
try:
    import ai_edge_litert.interpreter as tflite
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
    except ImportError:
        # Fallback to full TensorFlow
        try:
            import tensorflow.lite as tflite
        except Exception as e:
            tflite = None
            print(f"Warning: TFLite not found or crashed ({e}).")

class AbideRun:
    def __init__(self, models_dir="../"):
        self.enabled = True
        self.models = {}
        
        if tflite is None:
            print("Warning: AI Engine disabled because tflite library failed to load.")
            self.enabled = False
            
        required_models = {
            "net_a": "NET_A_BALL_ENCODER.tflite",
            "net_b": "NET_B_SELF_ENCODER.tflite",
            "net_c": "NET_C_ENEMY_ENCODER.tflite",
            "net_t": "AADFBS_NET_T.tflite",
            "enthralled": "AADFBS_ENTHRALLED.tflite"
        }
        
        for name, filename in required_models.items():
            if not self.enabled:
                break
            path = os.path.join(models_dir, filename)
            if os.path.exists(path):
                try:
                    interpreter = tflite.Interpreter(model_path=path)
                    interpreter.allocate_tensors()
                    
                    input_details = interpreter.get_input_details()
                    output_details = interpreter.get_output_details()
                    
                    self.models[name] = {
                        "interpreter": interpreter,
                        "input_index": input_details[0]['index'],
                        "output_index": output_details[0]['index']
                    }
                except Exception as e:
                    print(f"[WARN] Failed to load model {path}: {e}")
                    self.enabled = False
            else:
                print(f"[WARN] Missing model: {path}. AI Engine will run in dummy mode.")
                self.enabled = False
        
        # History for NET_T (requires 20 frames)
        self.history = deque(maxlen=20)
        self.embed_dim = 19

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
            e1_threat,
            0.0, # enemy2 threat
            state["ball_distance_norm"],
            state["ball_speed_norm"],
            state["shot_opportunity_ego"],
            state["ball_direction_alignment"],
            ball_free
        ]], dtype=np.float32)

        net_b_in = np.array([[
            state["ego_speed_norm"],
            state["ego_accel_norm"],
            1.0 - state["slip_indicator"], # ego_velocity_stability
            state["ego_pose_confidence"],
            state["slip_indicator"],
            1.0, # field zone conf
            state["near_boundary_risk"],
            max(0.0, 1.0 - state["enemy1_distance_norm"]), # free space ahead
            1.0 - state["ego_pose_confidence"] # visual_occlusion_level
        ]], dtype=np.float32)

        if state.get("enemy_observation_conf", 0.0) == 0.0:
            net_c_in = np.zeros((1, 12), dtype=np.float32)
        else:
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

        # Helper function
        def run_model(model_info, input_data):
            model_info["interpreter"].set_tensor(model_info["input_index"], input_data)
            model_info["interpreter"].invoke()
            return model_info["interpreter"].get_tensor(model_info["output_index"])

        # 2. Run Encoders
        a_emb = run_model(self.models["net_a"], net_a_in)[0] # 6 dims
        b_emb = run_model(self.models["net_b"], net_b_in)[0] # 7 dims
        c_emb = run_model(self.models["net_c"], net_c_in)[0] # 6 dims

        # 3. Form spatial embedding & update history
        spatial_emb = np.concatenate([a_emb, b_emb, c_emb]) # 19 dims
        self.history.append(spatial_emb)

        # Fill history if booting up
        while len(self.history) < 20:
            self.history.append(spatial_emb)

        net_t_in = np.array([list(self.history)], dtype=np.float32) # (1, 20, 19)

        # 4. Run Temporal Context
        t_emb = run_model(self.models["net_t"], net_t_in)[0] # 12 dims

        # 5. Form ENTHRALLED input
        enthralled_in = np.concatenate([a_emb, b_emb, c_emb, t_emb]).astype(np.float32).reshape(1, -1) # 31 dims

        # 6. Run Decision Engine
        action = run_model(self.models["enthralled"], enthralled_in)[0] # 7 dims

        action_dict = {
            "vx": float(action[0]),
            "vy": float(action[1]),
            "omega": float(action[2]),
            "kick": float(action[3]),
            "urgency": float(action[4]),
            "aggression": float(action[5]),
            "emergency": float(action[6]),
            "defense": 0.0,
            "pass_pref": 0.0
        }
        
        debug_info = {
            "NET_A_emb": a_emb.tolist(),
            "NET_B_emb": b_emb.tolist(),
            "NET_C_emb": c_emb.tolist(),
            "NET_T_emb": t_emb.tolist(),
            "raw_state": state
        }
        
        return action_dict, debug_info
