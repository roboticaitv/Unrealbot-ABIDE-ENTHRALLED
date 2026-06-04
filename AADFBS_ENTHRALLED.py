# NET_ENTHRALLED -  DETERMINISTIC DECISION NETWORK
# Artificial Belief-Integrated Decision Engine: Enhanced Through Abstract Latent Long-term Reasoning (ABIDE-ENTHRALLED)   - decision network
# Version: 1.0.7
# Tensorflow version: 2.15.0
# Activation functions used: relu, relu, tanh

# INPUTS FROM ALL SUB-NETWORKS
# ----------------------------------------------------------
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

# ---------------------------------------
# Index | Meaning | Type
# ---------------------------------------
# 0 | Overall enemy threat        | Continuous
# 1 | Immediate pressure          | Continuous
# 2 | Defensive blocking          | Continuous
# 3 | Interception risk           | Binary-like
# 4 | Evasion recommended         | Binary-like
# 5 | Aggressive play viable      | Binary-like
# ---------------------------------------

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

# OUTPUT ACTION VECTOR
# ----------------------------------------------------------
# ACTION_EMBED (8)
# Index | Meaning | Type
# ----------------------------------------------------------
# 0 | Linear velocity command        | Continuous [-1,1]
# 1 | Angular velocity command       | Continuous [-1,1]
# 2 | Kick / Actuation intensity     | Continuous [0,1]
# 3 | Action urgency                 | Continuous [0,1]
# 4 | Aggressiveness level           | Continuous [0,1]
# 5 | Defensive bias                 | Continuous [0,1]
# 6 | Pass preference                | Continuous [0,1]
# 7 | Emergency override             | Binary-like
# ----------------------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
tf.random.set_seed(54)
np.random.seed(13)

INPUT_DIM = 32 # 6 (Net A) + 8 (Net B) + 6 (Net C) + 12 (Net T)
OUTPUT_DIM = 9

SAMPLES = 80000
EPOCHS = 80
BATCH_SIZE = 256
LR = 1e-4

# =========================
# INTELIGENT SAMPLE GENERATION
# =========================
def generate_sample():

    # =========================
    # INPUTS BASE
    # =========================
    net_a = np.clip(np.random.normal(0.5, 0.25, 6), 0, 1)
    net_b = np.clip(np.random.normal(0.5, 0.25, 8), 0, 1)
    net_c = np.clip(np.random.normal(0.5, 0.3, 6), 0, 1)
    net_t = np.clip(np.random.normal(0.5, 0.25, 12), 0, 1)

    # =========================
    # CONTEXT VARIABLES
    # =========================
    ball_distance = np.random.beta(2, 5)
    alignment = np.random.uniform(0, 1)
    enemy_distance = np.random.beta(3, 2)
    blocking = np.random.uniform(0, 1)

    # =========================
    # VARIABLES
    # =========================
    ball_free = net_a[1]
    shoot_window = net_a[2]

    mobility = net_b[0]
    stability = net_b[2]

    enemy_threat = net_c[0]
    interception = net_c[3]

    pressure = net_t[0]
    defensive_overload = net_t[2]
    emergency = net_t[4]

    # =========================
    # PHYSICS ADJUSTMENTS
    # =========================
    enemy_threat *= (1 - enemy_distance)
    shoot_window *= alignment * (1 - blocking)

    # =========================
    # GOOD CONDITIONS
    # =========================
    good_shot = (
        ball_distance < 0.2 and
        alignment > 0.8 and
        blocking < 0.3 and
        enemy_threat < 0.4
    )

    bad_shot = (
        ball_distance > 0.3 or
        alignment < 0.5 or
        blocking > 0.5
    )

    danger = (
        enemy_threat > 0.6 or
        interception > 0.5 or
        emergency > 0.7
    )

    # =========================
    # OUTPUTS
    # =========================

    # VELOCITY
    if danger:
        v = -0.6
    else:
        v = 0.8 * (1 - ball_distance) * mobility

    vy = 0.0 # Strafing
    w = np.clip((alignment - 0.5) * 2, -1, 1)

    if good_shot:
        kick = 1.0
    elif bad_shot:
        kick = 0.0
    else:
        kick = 0.2 * shoot_window

    urgency = np.clip(pressure + (1 - ball_distance), 0, 1)

    aggr = np.clip((1 - enemy_threat) * alignment, 0, 1)

    defense = np.clip(enemy_threat + defensive_overload, 0, 1)

    pass_pref = np.clip(blocking * (1 - alignment), 0, 1)

    emergency_flag = 1.0 if danger else 0.0

    x = np.concatenate([net_a, net_b, net_c, net_t])

    y = np.array([
        v, vy, w, kick, urgency, aggr, defense, pass_pref, emergency_flag
    ], dtype=np.float32)

    return x, y


# =========================
# DATASET
# =========================
def build_dataset(n):
    X, Y = [], []
    for _ in range(n):
        x, y = generate_sample()
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


# =========================
# LOSS PRO
# =========================
def custom_loss(y_true, y_pred):

    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    wrong_kick = tf.maximum(0.0, y_pred[:,3] - y_true[:,3])

    miss_kick = tf.maximum(0.0, y_true[:,3] - y_pred[:,3])

    conflict = tf.maximum(0.0, y_pred[:,5] - tf.abs(y_pred[:,0]))

    return (
        mse
        + 0.8 * tf.reduce_mean(wrong_kick)
        + 0.6 * tf.reduce_mean(miss_kick)
        + 0.3 * tf.reduce_mean(conflict)
    )


# =========================
# MODEL
# =========================
def build_model():
    model = models.Sequential([
        layers.Input(shape=(INPUT_DIM,)),

        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(64, activation="relu"),
        layers.BatchNormalization(),

        layers.Dense(32, activation="relu"),

        layers.Dense(OUTPUT_DIM, activation="tanh")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
        loss=custom_loss
    )

    return model


# =========================
# TRAIN
# =========================
print("[INFO] Generando dataset...")
X, Y = build_dataset(SAMPLES)

split = int(0.85 * SAMPLES)

X_train, X_val = X[:split], X[split:]
Y_train, Y_val = Y[:split], Y[split:]

print("[INFO] Building model...")
model = build_model()

print("[INFO] Training...")
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
)

# =========================
# GRAPH
# =========================
plt.figure()
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.legend()
plt.grid()
plt.title("ENTHRALLED TRAINING")
# # plt.show()

# =========================
# SAVE
# =========================
model.save("AADFBS_ENTHRALLED.h5")