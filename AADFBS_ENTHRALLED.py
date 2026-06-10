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
def generate_sample(scenario=0):

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
    # FORCE SCENARIOS (DATASET BALANCING)
    # =========================
    if scenario == 1: # FORCED SHOT (Aggressive)
        ball_distance = np.random.uniform(0, 0.25)
        alignment = np.random.uniform(0.6, 1.0)
        blocking = np.random.uniform(0, 0.4)
        net_a[1] = np.random.uniform(0, 0.3) # Low enemy threat
        net_a[3] = np.random.uniform(0.7, 1.0) # High shoot window
    elif scenario == 2: # FORCED PASS
        ball_distance = np.random.uniform(0, 0.2)
        alignment = np.random.uniform(0, 0.4) # Bad alignment for shot
        blocking = np.random.uniform(0.6, 1.0) # Blocked by enemy
        net_b[4] = np.random.uniform(0.7, 1.0) # Ally coordination high
    elif scenario == 3: # FORCED DANGER / DEFENSE
        enemy_distance = np.random.uniform(0, 0.3)
        net_c[0] = np.random.uniform(0.7, 1.0) # High enemy threat
        net_t[4] = np.random.uniform(0.7, 1.0) # Emergency

    # =========================
    # VARIABLES
    # =========================
    ball_free = net_a[2]
    shoot_window = net_a[3]

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
    # AGGRESSIVE CONDITIONS
    # =========================
    good_shot = (
        ball_distance < 0.35 and
        alignment > 0.5 and
        blocking < 0.5
    )

    bad_shot = (
        ball_distance > 0.5 or
        alignment < 0.3 or
        blocking > 0.7
    )

    # El estado de emergencia propio de NET_B (incluye el riesgo de salir de la linea)
    self_emergency = net_b[6]

    danger = (
        enemy_threat > 0.7 or
        interception > 0.7 or
        emergency > 0.8 or
        (self_emergency > 0.6 and not good_shot) # Huir de la linea SOLO si no estamos atacando porteria
    )

    # =========================
    # AGGRESSIVE OUTPUTS
    # =========================

    # VELOCITY: Always go maximum speed if not in danger
    if danger:
        v = -1.0 # Max reverse
    else:
        v = 1.0 * mobility # Max forward speed

    vy = 0.0 # Strafing
    w = np.clip((alignment - 0.5) * 2.5, -1, 1) # Fast turning

    # KICKING: Maximize kick when possible
    if good_shot:
        kick = 1.0
    elif bad_shot:
        kick = 0.0
    else:
        kick = 0.5 * shoot_window

    urgency = np.clip(pressure + (1 - ball_distance), 0, 1)
    
    aggr = 1.0 if not danger else 0.0

    defense = np.clip(enemy_threat + defensive_overload, 0, 1)

    # Passing: If blocked and ally available, pass hard
    if blocking > 0.5 and alignment < 0.5 and net_b[4] > 0.5:
        pass_pref = 1.0
        kick = 0.8 # Force a kick to pass
    else:
        pass_pref = 0.0

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
    for i in range(n):
        scenario = i % 4 # Balance the 4 scenarios equally
        x, y = generate_sample(scenario)
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