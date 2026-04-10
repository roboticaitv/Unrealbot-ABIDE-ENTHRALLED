# =========================================
# CONVERT H5 → ONNX (ENTHRALLED)
# =========================================

import tensorflow as tf
import tf2onnx
import onnx
import numpy as np

# =========================
# CONFIG
# =========================
MODEL_PATH = "h5_models/AADFBS_ENTHRALLED.h5"
ONNX_PATH = "AADFBS_ENTHRALLED.onnx"

INPUT_DIM = 32
OPSET = 13

# =========================
# CUSTOM LOSS (IMPORTANTE)
# =========================
def custom_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    conflict = tf.maximum(0.0, y_pred[:,4] - tf.abs(y_pred[:,0]))
    wrong_kick = tf.maximum(0.0, y_pred[:,2] - y_true[:,2])
    miss_kick = tf.maximum(0.0, y_true[:,2] - y_pred[:,2])

    return (
        mse
        + 0.8 * tf.reduce_mean(wrong_kick)
        + 0.6 * tf.reduce_mean(miss_kick)
        + 0.3 * tf.reduce_mean(conflict)
    )

# =========================
# LOAD MODEL
# =========================
print("[INFO] Cargando modelo...")

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"custom_loss": custom_loss},
    compile=False  # 🔥 importante (evita problemas en conversión)
)

model.summary()

# =========================
# INPUT SIGNATURE (CRÍTICO)
# =========================
spec = (
    tf.TensorSpec(
        shape=(None, INPUT_DIM),
        dtype=tf.float32,
        name="input"
    ),
)

# =========================
# CONVERSION
# =========================
print("[INFO] Convirtiendo a ONNX...")

model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=spec,
    opset=OPSET,
    output_path=ONNX_PATH
)

print("[SUCCESS] Modelo convertido a ONNX:", ONNX_PATH)

# =========================
# VALIDACIÓN ONNX
# =========================
print("[INFO] Validando modelo ONNX...")

onnx_model = onnx.load(ONNX_PATH)
onnx.checker.check_model(onnx_model)

print("[SUCCESS] ONNX válido ✔")

# =========================
# TEST DE INFERENCIA (🔥 MUY IMPORTANTE)
# =========================
print("[INFO] Probando inferencia ONNX...")

import onnxruntime as ort

session = ort.InferenceSession(ONNX_PATH)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# input dummy
x_test = np.random.rand(1, INPUT_DIM).astype(np.float32)

output = session.run([output_name], {input_name: x_test})

print("[SUCCESS] Inferencia OK")
print("Output shape:", np.array(output).shape)
print("Sample output:", output[0][0])