import os
import numpy as np
import tensorflow as tf
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
net_a = tf.keras.models.load_model("NET_A_BALL_ENCODER.h5", compile=False)
net_b = tf.keras.models.load_model("NET_B_SELF_ENCODER.h5", compile=False)
net_c = tf.keras.models.load_model("NET_C_ENEMY_ENCODER.h5", compile=False)
net_t = tf.keras.models.load_model("AADFBS_NET_T.h5", compile=False)
net_enthralled = tf.keras.models.load_model("AADFBS_ENTHRALLED.h5", compile=False)

def run_scenario(name, state_a, state_b, state_c):
    a_in = np.array([state_a], dtype=np.float32)
    b_in = np.array([state_b], dtype=np.float32)
    c_in = np.array([state_c], dtype=np.float32)
    
    embed_a = net_a.predict(a_in, verbose=0)[0]
    embed_b = net_b.predict(b_in, verbose=0)[0]
    embed_c = net_c.predict(c_in, verbose=0)[0]
    
    combined_embed = np.concatenate([embed_a, embed_b, embed_c])
    seq = np.tile(combined_embed, (20, 1))
    t_in = np.expand_dims(seq, axis=0)
    
    embed_t = net_t.predict(t_in, verbose=0)[0]
    
    final_input = np.concatenate([embed_a, embed_b, embed_c, embed_t])
    final_in = np.expand_dims(final_input, axis=0)
    
    action = net_enthralled.predict(final_in, verbose=0)[0]
    
    print(f"\n==================================================")
    print(f" ESCENARIO: {name}")
    print(f"==================================================")
    print(f"  V_Acel: {action[0]:.2f} | Giro: {action[2]:.2f} | Pateo: {action[3]:.2f} | Pase: {action[7]:.2f}")
    if action[8] > 0.5:
        print("  >> ! MODO EMERGENCIA (REVERSA / EVASION) ACTIVADO ! <<")

# ESCENARIO 1: Peligro de Linea Blanca (No Porteria)
s1_a = [0.0, 0.0, 0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0]
s1_b = [0.8, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.1, 0.9, 1.0, 0.0, 0.0, 1.0, 0.0] # near_boundary_risk = 0.9
s1_c = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
run_scenario("Peligro Linea Blanca (Debe Dar Reversa)", s1_a, s1_b, s1_c)

# ESCENARIO 2: Ataque en Linea Blanca (Porteria)
s2_a = [1.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.9, 0.0, 0.9, 0.0] # good shot
s2_b = [0.8, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.1, 0.9, 1.0, 0.0, 0.0, 1.0, 0.0] # near_boundary_risk = 0.9
s2_c = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
run_scenario("Atacando Linea Porteria (DEBE PATEAR Y AVANZAR)", s2_a, s2_b, s2_c)

# ESCENARIO 3: Encerrona 2vs1
s3_a = [1.0, 0.0, 0.8, 0.8, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
s3_b = [0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
s3_c = [0.1, 0.1, 0.8, 0.8, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.9, 1.0] # Two enemies extremely close and blocking
run_scenario("Encerrona 2vs1 (Reversa)", s3_a, s3_b, s3_c)

# ESCENARIO 4: Escape por Pase (1 Enemigo bloqueando, 1 Aliado libre)
s4_a = [0.9, 0.0, 0.8, 0.0, 0.1, 0.0, 0.1, 0.9, 0.2, 0.0]
s4_b = [0.1, 0.0, 0.9, 1.0, 0.0, 1.0, 0.0, 0.5, 0.0, 0.2, 0.9, 0.9, 0.1, 0.0] # ally_dist 0.2, ally_align 0.9
s4_c = [0.1, 1.0, 0.2, 0.0, 0.9, 0.0, 0.9, 0.0, 0.1, 0.0, 0.8, 1.0] # E1 block
run_scenario("Escape por Pase (Bloqueado pero Aliado Libre)", s4_a, s4_b, s4_c)

print("\n Pruebas Finalizadas.")
