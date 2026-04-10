# NET_A - BALL STATE NETWORK
# Artificial Belief-Integrated Decision Engine
# (ABIDE-ENTHRALLED)
# Version: 1.1.1
# Tensorflow: 2.15.0
# Activation functions used: relu, relu, linear

# ===============================================
# BELIEF VECTOR DEFINITION (N_BELIEFS = 10)
# Index | Meaning | Range
# ---------------------------------------
# 0 | P_ball_possession_ego      | [0,1]
# 1 | P_ball_possession_ally     | [0,1]
# 2 | P_enemy1_ball_threat       | [0,1]
# 3 | P_enemy2_ball_threat       | [0,1]
# 4 | ball_distance_norm         | [0,1]
# 5 | ball_speed_norm            | [0,1]
# 6 | P_shot_opportunity_ego     | [0,1]
# 7 | P_pass_opportunity         | [0,1]
# 8 | ball_direction_alignment   | [0,1]
# 9 | P_ball_free                | [0,1]
# ===============================================

# ===============================================
# BALL SEMANTIC EMBEDDING (BALL_EMBED = 6)
# Index | Meaning | Type
# ------------------------------------------------------
# 0 | Offensive opportunity     | Continuous
# 1 | Enemy threat level        | Continuous
# 2 | Ball free likelihood      | Continuous
# 3 | Shoot window              | Binary-like
# 4 | Defensive urgency         | Binary-like
# 5 | Chase ball condition      | Binary-like
# ===============================================

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow as tf

# General values
np.random.seed(1585)
tf.random.set_seed(167)

N_BELIEFS = 10
BALL_EMBED = 6

# Type of model
model=Sequential()

# First layer
model.add(Dense(units=32, input_dim=10))
model.add(Activation("relu"))

# Layers
# Layer 2
model.add(Dense(units=16))
model.add(Activation("relu"))

# Layer 3 (Ball semantic embedding)
model.add(Dense(units=6))
model.add(Activation("linear"))

# Loss designation (Importance of each semantic output)
EMBED_LOSS_WEIGHTS = tf.constant([
    1.0, # offencive opportunity
    1.0, # enemy threat
    1.0, # ball free likelihood
    2.0, # shoot window
    2.0, # defensive urgency
    2.0  # chase condition
], dtype=tf.float32)

def weighted_mse(y_true, y_pred):
    error = tf.square(y_true - y_pred)
    weighted_error = error * EMBED_LOSS_WEIGHTS
    return tf.reduce_mean(weighted_error)

# Semantic penalty
def semantic_penalty(y_pred):
    shoot = y_pred[:, 3]
    defend = y_pred[:, 4]
    chase = y_pred[:, 5]
    free = y_pred[:, 2]

    # Invalid semantic combinations
    p1 = tf.maximum(0.0, shoot + defend - 1.0)
    p2 = tf.maximum(0.0, chase - free)

    return tf.reduce_mean(p1 + p2)

# Total loss function
def total_loss(y_true, y_pred):
    return weighted_mse(y_true, y_pred) + 0.3 * semantic_penalty(y_pred)

# Compile the model
model.compile(
    optimizer="adam",
    loss=total_loss
)

#Targets generation
def hard_flag(x, threshold):
    return 1.0 if x > threshold else 0.0

# Ball semantic targets
def ball_semantic_target(b):
    shot_ego = b[6]
    enemy_threat = max(b[2], b[3])
    loose_ball = b[9]

    return np.array([
        shot_ego, # Offensive opportunity
        enemy_threat, # Enemy threat
        loose_ball, # Ball free likelihood
        1.0 if shot_ego > 0.8 else 0.0, # Shoot window
        1.0 if enemy_threat > 0.7 else 0.0, #Defensive urgency
        1.0 if loose_ball > 0.6 else 0.0 # Chase condition
    ])

# Dataset generator - This is a random generator for different states
def generate_random_belief():
    b = np.random.rand(N_BELIEFS)

    # Semantic coherence constraints
    if b[0]  > 0.7: # ego possesion
        b[9] *= 0.2
    
    if b[9] > 0.7: # Free ball
        b[0] *= 0.2
        b[1] *= 0.2
    
    return b.astype(np.float32)

def build_dataset(samples=200000):
    X = []
    Y = []
    for _ in range(samples):
        belief = generate_random_belief()
        target = ball_semantic_target(belief)

        X.append(belief)
        Y.append(target)
    
    X = np.array(X)
    Y = np.array(Y)

    print("Dataset Shape X:", X.shape)
    print("Dataset Shape Y:", Y.shape)

    return X, Y

# Training adjustment
X, Y = build_dataset(150000)

history = model.fit(
    X,
    Y,
    epochs=150,
    batch_size=64,
    validation_split=0.3,
    shuffle=True
)

# Plot code block
plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("NET_A - BALL EMBEDDING LOSS")
plt.legend()
plt.grid(True)
plt.show()

# Semantic test beliefs (FOR ONLY MANUAL VALIDATION)
test_beliefs = np.array([
    # Case 1: Ego has the ball and clear shoot
    [0.9, 0.1, 0.2, 0.1, 0.2, 0.3, 0.9, 0.4, 0.8, 0.1],

    # Case 2: Free ball with enemies near
    [0.1, 0.1, 0.8, 0.7, 0.4, 0.6, 0.1, 0.2, 0.3, 0.9],

    # Case 3: Enemy domain
    [0.05, 0.05, 0.9, 0.85, 0.6, 0.4, 0.0, 0.1, 0.2, 0.05],

    # Case 4: Viable pass without shoot
    [0.6, 0.7, 0.2, 0.2, 0.3, 0.4, 0.3, 0.8, 0.7, 0.1],

    # Case 5: Disputed balls
    [0.4, 0.4, 0.6, 0.6, 0.5, 0.5, 0.4, 0.4, 0.5, 0.3]
], dtype=np.float32)

semantic_outputs = model.predict(test_beliefs)

print("\n===== SEMANTIC TEST RESULTS =====")
for i, out in enumerate(semantic_outputs):
    print(f"\nTest case {i+1}")
    print(f"Offensive opportunity: {out[0]:.2f}")
    print(f"Enemy threat level:    {out[1]:.2f}")
    print(f"Ball free likelihood:  {out[2]:.2f}")
    print(f"Shoot window:          {out[3]:.2f}")
    print(f"Defensive urgency:     {out[4]:.2f}")
    print(f"Chase condition:       {out[5]:.2f}")

# Input vector (true belief - for debugging tests)
beliefs = np.random.rand(5, N_BELIEFS).astype(np.float32)
embeddings = model.predict(beliefs)

print("Beliefs:")
print(beliefs)
print("Semantic embeddings:")
print(embeddings)

# Saving model
model.save("NET_A_BALL_ENCODER.h5")