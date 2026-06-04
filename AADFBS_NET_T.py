# NET_T - Long-term Strategic Transformer
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED) 
# Version: 1.1.3
# Tensorflow version: 2.15.0

# ----------------------------------------------------------
# INPUT SEQUENCE DEFINITION
# ----------------------------------------------------------
# Per timestep embedding:
# Index | Source | Meaning
# ----------------------------------------------------------
# 0-5   | NET_A  | Ball semantic embedding
# 6-11  | NET_B  | Self state semantic embedding
# 12-17 | NET_C  | Enemy state semantic embedding
# ----------------------------------------------------------
# EMBED_DIM = 18
# ----------------------------------------------------------

# ----------------------------------------------------------
# OUTPUT
# ----------------------------------------------------------
# Index | Meaning | Type
# ----------------------------------------------------------
# 0 | Strategic pressure trend      | Continuous
# 1 | Offensive momentum            | Continuous
# 2 | Defensive overload             | Continuous
# 3 | Counterattack readiness       | Continuous
# 4 | Risk accumulation             | Continuous
# 5 | Tempo acceleration            | Continuous
# 6 | Positional stability           | Continuous
# 7 | Tactical chaos level           | Continuous
# 8 | Aggression window              | Binary-like
# 9 | Regroup recommended            | Binary-like
# 10| Long-play opportunity          | Binary-like
# 11| Emergency defense              | Binary-like
# ----------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Dense, Input, LayerNormalization,
    MultiHeadAttention, Dropout, Add
)
import matplotlib.pyplot as plt

# Parameters
np.random.seed(1448)
tf.random.set_seed(145)

SEQ_LEN = 20 # Max legnth plays considered
EMBED_DIM = 20 # Embedding per play
HIST_EMBED = 12

NUM_HEADS = 12 # Multiple attention
FF_DIM = 48 # Feed-forward dimension inside of transformer
DROPOUT = 0.1

# Input: Embebed history plays
inputs = Input(shape=(SEQ_LEN, EMBED_DIM), name="Semantic_Sequence")

# Transformer architecture block
attn = MultiHeadAttention(
    num_heads=NUM_HEADS,
    key_dim=EMBED_DIM
)(inputs,inputs)

attn = Dropout(DROPOUT)(attn)
x = Add()([inputs, attn])
x = LayerNormalization()(x)

ff = Dense(FF_DIM, activation="relu")(x)
ff = Dense(EMBED_DIM)(ff)
ff = Dropout(DROPOUT)(ff)

x = Add()([x, ff])
x = LayerNormalization()(x)

# Temporal aggregations
x = tf.keras.layers.GlobalAveragePooling1D()(x)

# Projections to the historical embedding
outputs = Dense(HIST_EMBED, activation="linear")(x)

# Model summary
model = Model(inputs=inputs, outputs=outputs, name="NET_T_CONTEXT_ENCODER")
model.summary()

# Sequence generator
def generate_sequence():
    base = np.random.rand(EMBED_DIM)
    seq = []

    for t in range(SEQ_LEN):
        noise = np.random.normal(0, 0.05, EMBED_DIM)
        seq.append(base + noise + 0.01 * t)

    seq = np.array(seq, dtype=np.float32)

    # Take into account the future context
    future_context = np.array([
        np.mean(seq[:, 0:6]), # Pressure trend
        np.mean(seq[:, 6]), # Momentum proxy
        np.mean(seq[:, 14]), # defensive overlod
        np.mean(seq[:, 7]), # counter readiness
        np.std(seq[:, 14:20]), # risk
        np.mean(np.diff(seq[:, 0])), # Tempo
        1.0 - np.std(seq[:, 6:14]),
        np.std(seq),
        1.0 if np.mean(seq[:, 0]) > 0.7 else 0.0,
        1.0 if np.mean(seq[:, 14]) > 0.7 else 0.0,
        1.0 if np.mean(seq[:, 6]) > 0.6 else 0.0,
        1.0 if np.mean(seq[:, 14]) > 0.8 else 0.0
    ], dtype=np.float32)

    return seq, future_context

# Defining our dataset
def build_dataset(samples=5000):
    X, Y = [], []

    for _ in range(samples):
        seq, target = generate_sequence()
        X.append(seq)
        Y.append(target)

    return np.array(X), np.array(Y)

# Compile the model
X, Y = build_dataset()

model.compile(
    optimizer="adam",
    loss="mse"
)

# Training config
history = model.fit(
    X, Y,
    epochs=75,
    batch_size=256,
    validation_split=0.2,
    shuffle=True
)

# Manually Verfication
test_seq, expected = generate_sequence()
pred = model.predict(test_seq[np.newaxis])

print("\n===== NET_T MANUAL TEST =====")
for i, v in enumerate(pred[0]):
    print(f"Context {i}: {v:.2f}")

# Trainin visual representation
plt.figure(figsize=(8,5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("NET_T - Historical Context Transformer Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
# # plt.show()

# model saveq
model.save("AADFBS_NET_T.h5")