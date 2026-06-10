import os
import numpy as np
import tensorflow as tf
import keras

# Suppress TF warnings for clean output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Fix for Keras version mismatch (unrecognized keyword 'optional')
original_from_config = keras.layers.InputLayer.from_config
def patched_from_config(config):
    if 'optional' in config:
        del config['optional']
    return original_from_config(config)
keras.layers.InputLayer.from_config = patched_from_config

original_mha_from_config = keras.layers.MultiHeadAttention.from_config
def patched_mha_from_config(config):
    if 'softmax_robust_masking' in config:
        del config['softmax_robust_masking']
    return original_mha_from_config(config)
keras.layers.MultiHeadAttention.from_config = patched_mha_from_config

print("Cargando Modelos del Sistema ABIDE...")
try:
    net_a = tf.keras.models.load_model("NET_A_BALL_ENCODER.h5", compile=False)
    net_b = tf.keras.models.load_model("NET_B_SELF_ENCODER.h5", compile=False)
    net_c = tf.keras.models.load_model("NET_C_ENEMY_ENCODER.h5", compile=False)
    net_t = tf.keras.models.load_model("AADFBS_NET_T.h5", compile=False)
    net_enthralled = tf.keras.models.load_model("AADFBS_ENTHRALLED.h5", compile=False)
    print("Modelos cargados exitosamente.\n")
except Exception as e:
    print(f"Error cargando modelos: {e}")
    exit()

def run_scenario(name, state_a, state_b, state_c):
    print(f"==================================================")
    print(f" ESCENARIO: {name}")
    print(f"==================================================")
    
    # 1. Convert to numpy arrays
    a_in = np.array([state_a], dtype=np.float32)
    b_in = np.array([state_b], dtype=np.float32)
    c_in = np.array([state_c], dtype=np.float32)
    
    # 2. Get instantaneous semantic embeddings
    embed_a = net_a.predict(a_in, verbose=0)[0]
    embed_b = net_b.predict(b_in, verbose=0)[0]
    embed_c = net_c.predict(c_in, verbose=0)[0]
    
    print("--- 1. Percepción Semántica (Lo que entiende el robot) ---")
    print(f"  [Pelota]  Oportunidad Ofensiva: {embed_a[0]:.2f} | Amenaza Enemiga: {embed_a[1]:.2f} | Ventana de Tiro: {embed_a[3]:.2f}")
    print(f"  [Propio]  Movilidad: {embed_b[0]:.2f} | Seguridad del campo: {embed_b[3]:.2f} | Estado Emergencia: {embed_b[6]:.2f}")
    print(f"  [Enemigo] Amenaza Global: {embed_c[0]:.2f} | Riesgo Intercepción: {embed_c[3]:.2f} | Juego Agresivo Viable: {embed_c[5]:.2f}")
    
    # 3. Simulate Temporal Context (NET_T needs 20 time steps). 
    # For a static test, we will assume this state has been constant for 20 frames.
    combined_embed = np.concatenate([embed_a, embed_b, embed_c])
    seq = np.tile(combined_embed, (20, 1))
    t_in = np.expand_dims(seq, axis=0) # Shape: (1, 20, 20)
    
    embed_t = net_t.predict(t_in, verbose=0)[0]
    
    # 4. Final Decision (ENTHRALLED)
    # Inputs: net_a(6) + net_b(8) + net_c(6) + net_t(12) = 32 inputs
    final_input = np.concatenate([embed_a, embed_b, embed_c, embed_t])
    final_in = np.expand_dims(final_input, axis=0)
    
    action = net_enthralled.predict(final_in, verbose=0)[0]
    
    # ACTION_EMBED:
    # 0 | Linear velocity        [-1,1]
    # 1 | Angular velocity       [-1,1]
    # 2 | Rotational/Lateral     [-1,1] -> Using as w
    # 3 | Kick intensity         [0,1]
    # 4 | Action urgency         [0,1]
    # 5 | Aggressiveness         [0,1]
    # 6 | Defensive bias         [0,1]
    # 7 | Pass preference        [0,1]
    # 8 | Emergency override     [0,1]
    
    print("\n--- 2. Ejecución (Decisión Final de los Motores) ---")
    print(f"  Acelerador (Adelante/Atrás): {action[0]:.2f}  (>0 Adelante, <0 Atrás)")
    print(f"  Giro (Izquierda/Derecha):    {action[2]:.2f}  (>0 Der, <0 Izq)")
    print(f"  Fuerza de Pateo:             {action[3]:.2f}  (1.0 = Disparo Máximo)")
    print(f"  Nivel de Urgencia:           {action[4]:.2f}")
    print(f"  Sesgo Defensivo:             {action[6]:.2f}")
    print(f"  Preferencia de Pase:         {action[7]:.2f}")
    if action[8] > 0.5:
        print("  >> ! MODO EMERGENCIA ACTIVADO ! <<")
    print("\n")


# ==========================================
# ESCENARIO 1: OPORTUNIDAD CLARA DE GOL
# Tenemos la pelota, estamos cerca de la portería enemiga, no hay enemigos cerca.
# ==========================================
# Net A: ego_poss(1), ally_poss(0), e1_threat(0), e2_threat(0), dist(0), speed(0), shot_opp(0.9), pass(0), align(0.9), free(0)
s1_a = [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.9, 0.0]
# Net B: speed(0.5), accel(0), vel_stab(1), pose_conf(1), yaw(0), ang_stab(1), slip(0), field(1), bound(0), ally_dist(1), ally_align(0), ally_conf(0), free(1), occl(0)
s1_b = [0.5, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0]
# Net C: e1_dist(1), e2_dist(1), e1_vel(0), e2_vel(0), e1_align(0), e2_align(0), e1_block(0), e2_block(0), e1_goal(0), e2_goal(0), press(0), obs_conf(1)
s1_c = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

run_scenario("OPORTUNIDAD CLARA DE GOL", s1_a, s1_b, s1_c)


# ==========================================
# ESCENARIO 2: RIVAL ATACANDO
# El enemigo tiene la pelota y se acerca rápido. Nosotros estamos lejos.
# ==========================================
# Net A: ego(0), ally(0), e1_threat(0.9), e2_threat(0), dist(0.8), speed(0.8), shot(0), pass(0), align(0.1), free(0)
s2_a = [0.0, 0.0, 0.9, 0.0, 0.8, 0.8, 0.0, 0.0, 0.1, 0.0]
# Net B: Normal stability, but maybe moving backwards
s2_b = [0.8, 0.5, 0.8, 1.0, 0.0, 0.8, 0.0, 0.5, 0.2, 1.0, 0.0, 0.0, 0.2, 0.0]
# Net C: Enemy is close (dist 0.2), moving fast (0.8), highly aligned to our goal
s2_c = [0.2, 1.0, 0.8, 0.0, 0.9, 0.0, 0.0, 0.0, 0.9, 0.0, 0.9, 1.0]

run_scenario("RIVAL PELIGROSO ATACANDO", s2_a, s2_b, s2_c)


# ==========================================
# ESCENARIO 3: PELOTA DISPUTADA (FREE BALL)
# Pelota suelta en medio, nosotros y el enemigo a la misma distancia.
# ==========================================
s3_a = [0.0, 0.0, 0.5, 0.0, 0.5, 0.1, 0.3, 0.0, 0.5, 0.9]
s3_b = [0.2, 0.8, 0.9, 1.0, 0.0, 0.9, 0.0, 0.5, 0.0, 1.0, 0.0, 0.0, 0.5, 0.0]
s3_c = [0.5, 1.0, 0.5, 0.0, 0.5, 0.0, 0.2, 0.0, 0.2, 0.0, 0.5, 1.0]

run_scenario("PELOTA DIVIDIDA (50/50)", s3_a, s3_b, s3_c)

# ==========================================
# ESCENARIO 4: BLOQUEADO (NECESITA PASAR)
# Tenemos el balón, pero un rival gigante nos bloquea el tiro al arco. Aliado libre a la derecha.
# ==========================================
s4_a = [0.9, 0.0, 0.8, 0.0, 0.1, 0.0, 0.1, 0.8, 0.2, 0.0] # High pass opportunity
s4_b = [0.1, 0.0, 0.9, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.3, 0.8, 0.9, 0.1, 0.0] # Ally close and aligned
s4_c = [0.1, 1.0, 0.2, 0.0, 0.9, 0.0, 0.9, 0.0, 0.1, 0.0, 0.8, 1.0] # Enemy very close and blocking lane (0.9)

run_scenario("BLOQUEADO POR RIVAL (DEBERIA PASAR)", s4_a, s4_b, s4_c)
