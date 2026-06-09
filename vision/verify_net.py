import numpy as np
import cv2
import sys

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

class VerifyNet:
    def __init__(self, model_path="micro_verify_net_int8.tflite"):
        self.loaded = False
        if tflite is None:
            print("Warning: MicroVerifyNet disabled because tflite library failed to load.")
            return
            
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.loaded = True
        except Exception as e:
            print(f"Warning: Could not load MicroVerifyNet ({e}). Running without Stage 2 Verification.")
            
    def predict_patch(self, y_patch, u_patch, v_patch):
        if not self.loaded:
            return 1.0 # If model failed to load, pass everything through
            
        # Ensure the patch is strictly 32x32. 
        # If it's smaller (edge of screen or scale=2), resize it.
        # If it's larger (shouldn't happen), resize it.
        if y_patch.shape != (32, 32):
            y_patch = cv2.resize(y_patch, (32, 32), interpolation=cv2.INTER_LINEAR)
            u_patch = cv2.resize(u_patch, (32, 32), interpolation=cv2.INTER_LINEAR)
            v_patch = cv2.resize(v_patch, (32, 32), interpolation=cv2.INTER_LINEAR)
            
        # Stack into (32, 32, 3)
        img_yuv = np.dstack([y_patch, u_patch, v_patch])
        
        # Normalize to 0.0-1.0
        arr = img_yuv.astype(np.float32) / 255.0
        
        # Add batch dimension
        arr = np.expand_dims(arr, axis=0)
        
        self.interpreter.set_tensor(self.input_details[0]['index'], arr)
        self.interpreter.invoke()
        
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return output_data[0][0] # Probability of being a ball (0 to 1)
