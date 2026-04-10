import tensorflow as tf
import tf2onnx
import onnx

# =========================
# CONFIG
# =========================
MODEL_PATH = "h5_models/NET_C_ENEMY_ENCODER.h5"
ONNX_PATH = "NET_C_ENEMY_ENCODER.onnx"

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)

print("Modelo cargado correctamente")

# =========================
# INPUT SIGNATURE
# =========================
input_signature = (
    tf.TensorSpec([None, 12], tf.float32, name="input"),
)

# =========================
# CONVERSIÓN
# =========================
onnx_model, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=input_signature,
    opset=13 
)

# =========================
# GUARDAR
# =========================
onnx.save(onnx_model, ONNX_PATH)

print("Modelo convertido a ONNX correctamente:", ONNX_PATH)