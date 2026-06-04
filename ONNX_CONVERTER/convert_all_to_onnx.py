import os
import tensorflow as tf
import tf2onnx

# Make sure we're in the right directory
repo_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(repo_dir)

# Ensure the ONNX_models directory exists
if not os.path.exists("ONNX_models"):
    os.makedirs("ONNX_models")

models_to_convert = [
    ("AADFBS_NET_A.py", "NET_A_BALL_ENCODER.h5", "ONNX_models/NET_A_BALL_ENCODER.onnx"),
    ("AADFBS_NET_B.py", "NET_B_SELF_ENCODER.h5", "ONNX_models/NET_B_SELF_ENCODER.onnx"),
    ("AADFBS_NET_C.PY", "NET_C_ENEMY_ENCODER.h5", "ONNX_models/NET_C_ENEMY_ENCODER.onnx"),
    ("AADFBS_NET_T.py", "AADFBS_NET_T.h5", "ONNX_models/NET_T_CONTEXT_ENCODER.onnx"),
    ("AADFBS_ENTHRALLED.py", "AADFBS_ENTHRALLED.h5", "ONNX_models/AADFBS_ENTHRALLED.onnx")
]

print("Starting Unified ONNX Conversion...\n")

for script, h5_model, onnx_model in models_to_convert:
    print(f"[{h5_model}]")
    if not os.path.exists(h5_model):
        print(f"  -> WARNING: {h5_model} not found! You must run {script} to train it first.")
        continue
        
    try:
        model = tf.keras.models.load_model(h5_model, compile=False)
        spec = (tf.TensorSpec((None, *model.input_shape[1:]), tf.float32, name="input"),)
        output_path = os.path.join(repo_dir, onnx_model)
        
        # Convert from tf/keras to ONNX
        model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13, output_path=output_path)
        print(f"  -> Successfully converted to {onnx_model}")
        
    except Exception as e:
        print(f"  -> ERROR during conversion: {e}")

print("\nConversion finished!")
