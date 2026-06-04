# Base
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)   - ball state network
# Version: 0.7.4
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, softmax

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

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# General values
np.random.seed(165)
N_BELIEFS = 10
N_ACTIONS = 8

ACTIONS = {
    0: "GO_TO_BALL",
    1: "PASS",
    2: "SHOOT",
    3: "DEFEND",
    4: "PRESS",
    5: "REPOSITION",
    6: "WAIT",
    7: "COVER"
}

# Type of model
model=Sequential()

# First layer
model.add(Dense(units=32, input_dim=10))
model.add(Activation("relu"))

# Layers
# Layer 2
model.add(Dense(units=16))
model.add(Activation("relu"))

# Layer 3 (Actions Vector Output)
model.add(Dense(units=8))
model.add(Activation("softmax"))


# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Virtual Policies
def virtual_policies(b):
    possession_ego = b[0]
    possession_ally = b[1]
    enemy_threat = max(b[2], b[3])
    pass_ego = b[4]
    shot_ego = b[6]
    loose_ball = b[9]

    if shot_ego > 0.8 and enemy_threat < 0.3:
        return 2
    
    if pass_ego > 0.7:
        return 1
    
    if loose_ball > 0.6:
        return 0
    
    if enemy_threat > 0.7:
        return 3
    
    return 6

# Dataset generator - This is a random generator for different states
def generate_random_belief():
    b = np.random.rand(N_BELIEFS)

    # Coherence Adjustments
    if b[0]  > 0.7:
        b[9] *= 0.2
    
    if b[9] > 0.7:
        b[0] *= 0.2
        b[1] *= 0.2
    
    return b

def build_dataset(samples=100000):
    X = []
    Y = []
    for _ in range(samples):
        belief = generate_random_belief()
        action = virtual_policies(belief)

        X.append(belief)
        Y.append(action)
    
    X = np.array(X)
    Y = to_categorical(Y, num_classes=N_ACTIONS)

    print("Dataset Shape X:", X.shape)
    print("Dataset Shape Y:", Y.shape)

    return X, Y

# Training adjustment
X, Y = build_dataset(15000)

history = model.fit(
    X,
    Y,
    epochs=100,
    batch_size=64,
    validation_split=0.2,
    shuffle=True
)

# Plot code block
plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("AADBS - Accuracy")
plt.legend()
plt.grid(True)
# # plt.show()

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("AADBS - Loss")
plt.legend()
plt.grid(True)
# # plt.show()

# Input vector (belief)
belief = np.array([0.92, 0.01, 0.3, 0.2, 0.8, 0.1, 0.9, 0.0, 0.1, 0.05])
belief = belief.reshape(1,10)

# Output vector (Output_actions)
output_actions = model.predict(belief, verbose=0)

action = int(np.argmax(output_actions))
confidence = float(output_actions[0, action])

# Print of results
print("Probability Actions:", output_actions)
print("Action Choise: ", action)
print("Action Name:", ACTIONS[action])
print("Confidence: ", confidence)

# Saving model
model.save("aadbs_v1.keras")

expert_action = virtual_policies(belief[0])

print("Expert Action:", ACTIONS[expert_action])
print("Network Action:", ACTIONS[action])