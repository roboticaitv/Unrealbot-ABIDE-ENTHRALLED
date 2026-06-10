# NET_B -  SELF STATE NETWORK
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)   - self state network
# Version: 1.1.6
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, softmax

# ==================================================
# SELF STATE VECTOR DEFINITION (N_SELF_STATES = 14)
# Index | Meaning | Range
# --------------------------------------------------
# 0  | ego_speed_norm            | [0,1]
# 1  | ego_accel_norm            | [0,1]
# 2  | ego_velocity_stability    | [0,1]
# 3  | ego_pose_confidence       | [0,1]
# 4  | yaw_rate_norm             | [0,1]
# 5  | angular_stability         | [0,1]
# 6  | slip_indicator            | [0,1]
# 7  | field_zone_confidence     | [0,1]
# 8  | near_boundary_risk        | [0,1]
# 9  | ally_distance_norm        | [0,1]
# 10 | ally_bearing_alignment    | [0,1]
# 11 | ally_pose_confidence      | [0,1]
# 12 | free_space_ahead          | [0,1]
# 13 | visual_occlusion_level    | [0,1]
# ==================================================

# ==================================================
# SELF SEMANTIC EMBEDDING (SELF_EMBED = 8)
# Index | Meaning | Type
# --------------------------------------------------
# 0 | Mobility readiness        | Continuous
# 1 | Localization confidence   | Continuous
# 2 | Dynamic stability         | Continuous
# 3 | Field safety              | Continuous
# 4 | Ally coordination         | Continuous
# 5 | Exploration capability    | Continuous
# 6 | Emergency state           | Binary-like
# 7 | Control reliability       | Continuous
# ==================================================

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# General values
np.random.seed(429)
tf.random.set_seed(429)

N_SELF_STATES = 14
SELF_EMBED = 8

# Type of model
model=Sequential()

# First layer
model.add(Dense(units=32, input_dim=14))
model.add(Activation("relu"))

# Layers
# Layer 2
model.add(Dense(units=16))
model.add(Activation("relu"))

# Layer 3 (Actions Vector Output)
model.add(Dense(units=8))
model.add(Activation("linear"))

# Loss weights
EMBED_LOSS_WEIGHTS = tf.constant([
    1.0, # mobility readiness ("future develop")
    1.0, # Localization confidence
    1.0, # Dynamic stability
    1.2, # Safty field
    1.0, # Ally coordination
    1.0, # Capability for exploration
    2.0, # emergency state
    1.5 # control reliability
], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    return tf.reduce_mean(error * EMBED_LOSS_WEIGHTS)

def semantic_penalty(y_pred):
    emergency = y_pred[:, 6]
    control = y_pred[:, 7]
    stability = y_pred[:, 2]

    # The old penalty assumed that emergency and stability were mutually exclusive.
    # But a robot can be perfectly stable while driving straight into the boundary line!
    # So we remove the penalty between emergency and stability.
    
    # We only penalize if control is very high during an emergency
    p2 = tf.maximum(0.0, emergency + control - 1.5)

    return tf.reduce_mean(p2)

def total_loss(y_true, y_pred):
    return weighted_mse(y_true, y_pred) + 0.3 * semantic_penalty(y_pred)

# Compile the model
model.compile(
    optimizer="adam",
    loss=total_loss
)

# target generator
# self_semantic_target
def self_semantic_target(s):
    mobility = 0.5 * (s[0] + s[1])
    localization_quality = s[3]
    stability = np.clip(1.0 - max(s[4], s[6]), 0.0, 1.0)
    field_safety = 1.0 - s[8]
    coordination = 0.5 * (1.0 - s[9] + s[10])
    exploration = s[12] * (1.0 - s[13])
    emergency = float((s[6] > 0.7) or (s[8] > 0.7))
    control = 0.5 * (stability + localization_quality)

    return np.array([
        mobility,
        localization_quality,
        stability,
        field_safety,
        coordination,
        exploration,
        emergency,
        control
    ], dtype=np.float32)

# Dataset generator - This is a random generator for different states
def generate_random_self_state():
    s = np.random.rand(N_SELF_STATES)

    # Coherence Adjustments
    if s[6]  > 0.7: # slip
        s[0] *= 0.3
        s[1] *= 0.3

    
    if s[8] > 0.7:  # boundary
        s[12] *= 0.2
        
    
    return s.astype(np.float32)

def build_dataset(samples=20000):
    X = []
    Y = []
    for _ in range(samples):
        state = generate_random_self_state()
        target = self_semantic_target(state)

        X.append(state)
        Y.append(target)
    
    X = np.array(X)
    Y = np.array(Y, dtype=np.float32)


    print("Dataset Shape X:", X.shape)
    print("Dataset Shape Y:", Y.shape)

    return X, Y

# Training adjustment
X, Y = build_dataset(15000)

history = model.fit(
    X,
    Y,
    epochs=50,
    batch_size=64,
    validation_split=0.3,
    shuffle=True
)

# Plot code block
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.title("NET_B - SELF STATE EMBEDDING LOSS")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
# # plt.show()

# Final test
test_states = np.random.rand(5, N_SELF_STATES).astype(np.float32)
embeddings = model.predict(test_states)

print("\n===== NET_B TEST OUTPUT =====")
for i, e in enumerate(embeddings):
    print(f"\nTest case {i+1}")
    print(f"Mobility readiness:      {e[0]:.2f}")
    print(f"Localization confidence: {e[1]:.2f}")
    print(f"Dynamic stability:       {e[2]:.2f}")
    print(f"Field safety:            {e[3]:.2f}")
    print(f"Ally coordination:       {e[4]:.2f}")
    print(f"Exploration capability:  {e[5]:.2f}")
    print(f"Emergency state:         {e[6]:.2f}")
    print(f"Control reliability:     {e[7]:.2f}")

# Save the model
model.save("NET_B_SELF_ENCODER.h5")