# ABIDE-ENTHRALLED - Modelos TFLite y Entradas (Inputs)

Este directorio contiene las versiones exportadas (.tflite) de las redes neuronales del sistema ABIDE-ENTHRALLED. 

A continuación se documenta exactamente qué espera recibir cada red neuronal en su capa de entrada a partir de la lógica de programación establecida. **Todos los valores numéricos entregados deben estar normalizados en un rango entre `[0.0, 1.0]`**.

---

## 1. NET_A_BALL_ENCODER.tflite
**Propósito:** Entender el contexto y estado actual del balón.
**Entrada:** Vector 1D de **8 valores**:

| Índice | Variable | Descripción | Rango |
|--------|----------|-------------|-------|
| 0 | `possession_ego` | Posesión del balón por nuestro robot (0=No, 1=Sí) | `[0,1]` |
| 1 | `enemy1_ball_threat` | Amenaza táctica del Enemigo 1 sobre el balón | `[0,1]` |
| 2 | `enemy2_ball_threat` | Amenaza táctica del Enemigo 2 sobre el balón | `[0,1]` |
| 3 | `ball_distance_norm` | Distancia normalizada desde el robot hacia el balón | `[0,1]` |
| 4 | `ball_speed_norm` | Velocidad normalizada a la que se desplaza el balón | `[0,1]` |
| 5 | `shot_opportunity` | Claridad de oportunidad para realizar un tiro a portería | `[0,1]` |
| 6 | `ball_direction_alignment` | Qué tan alineado está nuestro robot mirando/yendo al balón | `[0,1]` |
| 7 | `ball_free` | Probabilidad de que el balón esté "suelto" (disputado/libre) | `[0,1]` |

**Salida:** Vector semántico del balón de tamaño 6.

---

## 2. NET_B_SELF_ENCODER.tflite
**Propósito:** Entender el estado físico y capacidades de nuestro robot (Ego).
**Aviso sobre IMU:** Debido a la remoción del sensor inercial (IMU), este vector se redujo de 11 a 9 valores. Ya no incluye `yaw_rate_norm`, y la aceleración (`ego_accel_norm`) ahora representa la orientación o *heading* visual.
**Entrada:** Vector 1D de **9 valores**:

| Índice | Variable | Descripción | Rango |
|--------|----------|-------------|-------|
| 0 | `ego_speed_norm` | Velocidad lineal normalizada de nuestro robot | `[0,1]` |
| 1 | `ego_accel_norm` | Aceleración normalizada (o heading en implementaciones) | `[0,1]` |
| 2 | `ego_velocity_stability` | Estabilidad del vector de movimiento actual | `[0,1]` |
| 3 | `ego_pose_confidence` | Nivel de confianza en la localización o estado de batería | `[0,1]` |
| 4 | `slip_indicator` | Indicador de si las ruedas están derrapando | `[0,1]` |
| 5 | `field_zone_confidence` | Confianza en qué zona de la cancha estamos ubicados | `[0,1]` |
| 6 | `near_boundary_risk` | Riesgo de salirse de la cancha o colisionar con una pared | `[0,1]` |
| 7 | `free_space_ahead` | Cuánto espacio libre tenemos para avanzar frontalmente | `[0,1]` |
| 8 | `visual_occlusion_level` | Nivel de "ceguera" u oclusión en el sensor de visión | `[0,1]` |

**Salida:** Vector semántico del robot de tamaño 7.

---

## 3. NET_C_ENEMY_ENCODER.tflite
**Propósito:** Entender la amenaza y posicionamiento de los adversarios.
**Entrada:** Vector 1D de **12 valores**:

> **Aviso Importante sobre Aliados/Sin Enemigos:** 
> El sistema asume un paradigma de 1 vs Enemigos (no existen aliados).
> Si **no hay enemigos presentes**, simplemente debes enviar un vector completamente lleno de ceros (`0.0`). La red fue entrenada para interpretar un vector nulo como un "entorno seguro" y automáticamente habilitará comportamientos agresivos/libres.

| Índice | Variable | Descripción | Rango |
|--------|----------|-------------|-------|
| 0 | `enemy1_distance` | Distancia normalizada al Enemigo 1 | `[0,1]` |
| 1 | `enemy2_distance` | Distancia normalizada al Enemigo 2 | `[0,1]` |
| 2 | `enemy1_velocity` | Velocidad normalizada del Enemigo 1 | `[0,1]` |
| 3 | `enemy2_velocity` | Velocidad normalizada del Enemigo 2 | `[0,1]` |
| 4 | `enemy1_ball_alignment` | Qué tan alineado va el Enemigo 1 hacia el balón | `[0,1]` |
| 5 | `enemy2_ball_alignment` | Qué tan alineado va el Enemigo 2 hacia el balón | `[0,1]` |
| 6 | `enemy1_blocking_lane` | Nivel en que el Enemigo 1 bloquea nuestro tiro o avance | `[0,1]` |
| 7 | `enemy2_blocking_lane` | Nivel en que el Enemigo 2 bloquea nuestro tiro o avance | `[0,1]` |
| 8 | `enemy1_goal_alignment` | Alineación táctica del Enemigo 1 hacia la portería | `[0,1]` |
| 9 | `enemy2_goal_alignment` | Alineación táctica del Enemigo 2 hacia la portería | `[0,1]` |
| 10| `enemy_pressure_level` | Presión táctica global y combinada de los enemigos | `[0,1]` |
| 11| `enemy_observation_conf`| Confianza en la detección visual de los enemigos | `[0,1]` |

**Salida:** Vector semántico del enemigo de tamaño 6.

---

## 4. Redes de Contexto y Decisión (Uso Automático Interno)
A las siguientes redes **no se les debe pasar información directa** calculada por tu código. El sistema de inferencia (ej. `system.py` o `inference.py`) se encarga de alimentarlas automáticamente concatenando los resultados semánticos (salidas) de las Redes A, B y C.

*   **AADFBS_NET_T.tflite (Transformer Estratégico):** Recibe una secuencia temporal (ventana con memoria de los últimos 20 instantes) de tamaño `(20, 19)`, donde 19 equivale a las salidas unidas de A(6) + B(7) + C(6). Devuelve un "vector de contexto histórico" de tamaño 12.
*   **AADFBS_ENTHRALLED.tflite (Red Determinista de Decisión):** Es la red final. Recibe un vector largo concatenado de tamaño `31`, producto del estado inmediato actual de las redes A(6) + B(7) + C(6) sumado al contexto a largo plazo que sacó la red T(12). Devuelve el **Vector de Acción de tamaño 7**, el cual contiene directamente las instrucciones para los motores (velocidad lineal, angular, pateo, agresividad, etc.).
