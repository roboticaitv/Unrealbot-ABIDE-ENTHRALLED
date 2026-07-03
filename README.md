# Unrealbot & ABIDE-ENTHRALLED 

Bienvenidos al repositorio central de integración del ecosistema **Unrealbot** y **ABIDE-ENTHRALLED**. Este monorepositorio sirve como la base de código unificada y definitiva para el control de hardware, telemetría, visión por computadora e Inteligencia Artificial (IA) del robot autónomo.

---

## Arquitectura del Proyecto

El repositorio está organizado estructuralmente para separar el control de bajo nivel (Microcontroladores) del pensamiento de alto nivel (Raspberry Pi / IA):

*   **`/firmware/`**: Código C++/Arduino para microcontroladores.
    *   `Station_ESPNOW/`: Código para la estación base ESP32 (puente inalámbrico ESP-NOW para comunicación de telemetría de ultra baja latencia).
    *   `Unrealbot_Embedded/`: Firmware central del robot (ESP-IDF, control de motores omnidireccionales DRV8251, lazos PID, cinemática inversa y control del pateador magnético).
*   **`/software/`**: Código en Python que ejecuta el "cerebro" del robot (Raspberry Pi 5 / PC).
    *   `robot_main.py`: Script principal orquestador que une visión, IA y puerto serial.
    *   `/models/abide/`: Contiene los modelos `.tflite` de redes neuronales y el orquestador maestro `abide_run.py`.
    *   `/vision/`: Módulos de visión por computadora (`color_tracking.py`, `hud.py`, `inferencia.py`) y el servidor web del Dashboard (`live_stream.py`).
*   **`/docs/`**: Documentación técnica detallada (literatura CV, análisis de IA, esquemas electrónicos).
*   **`/scripts/`**: Utilidades y herramientas de desarrollo (ej. sincronización por SSH `sync_to_pi.ps1`).

---

## Guía de Inicio Rápido (Getting Started)

### Entorno de Python
Este sistema requiere un entorno virtual de Python sumamente específico que contenga las librerías `picamera2` (y los bindings nativos de `libcamera` de Raspberry Pi), `ai-edge-litert` (TensorFlow Lite), `opencv-python` y `pyserial`.

En la estación base o la Raspberry Pi, el entorno virtual designado es:
```bash
/home/pi/Desktop/abide_env_313
```

Para activar este entorno, ejecuta:
```bash
source /home/pi/Desktop/abide_env_313/bin/activate
```

### Ejecución del Sistema

El sistema tiene dos modos principales de ejecución dependiendo de lo que necesites.

> [!WARNING]
> **Bloqueo de Hardware (Cámaras):** ¡No puedes ejecutar ambos modos al mismo tiempo! Las cámaras de la Raspberry Pi solo permiten que un proceso las controle a la vez. Si el Dashboard Web está corriendo, `robot_main.py` fallará y viceversa. Para cambiar de modo, primero debes detener el otro proceso (por ejemplo, con `pkill -f live_stream.py`).

#### 1. Modo de Autonomía Completa (Orquestador Principal)
Para que el robot cobre vida, encienda las cámaras, procese la IA y empiece a enviar comandos a la ESP32 para moverse, ejecuta:
```bash
python software/robot_main.py
```

#### 2. Modo de Transmisión y Calibración (Dashboard Web)
Si necesitas ver lo que el robot está viendo en tiempo real, calibrar colores o revisar la telemetría (sin activar el movimiento autónomo), inicia el servidor web:
```bash
python software/vision/live_stream.py
```
*Opcional: Puedes agregar `--no-hud` para desactivar los dibujos sobre el video y ahorrar CPU.*

El panel de control (Dashboard) y el video en vivo estarán disponibles en cualquier dispositivo de tu red local ingresando a:
`http://<IP_DE_LA_RASPBERRY>:8080`

*(Nota: El servidor está configurado por defecto para escuchar en `0.0.0.0:8080` para aceptar conexiones de toda la red. Si necesitas cambiar el host o el puerto, edita las líneas finales del archivo `software/vision/live_stream.py`).*

---

## Visión por Computadora (CV) y Pipeline

El Unrealbot opera utilizando visión a bordo de 360 grados (dos cámaras gran angular de 200° "Fisheye" apuntando hacia el frente y hacia atrás) para percibir el estado del juego.

### Flujo de Procesamiento (Pipeline)
1. **Captura de Fotogramas (Frame Capture):** Hilos en segundo plano extrayendo fotogramas crudos YUV420 directo del hardware de la Pi.
2. **Rastreo de Color (Color Tracking):** Filtros clásicos de umbrales YUV (`cv2.inRange`) optimizados para aislar la pelota (naranja) y las porterías (azul y amarilla). Incorpora filtros inteligentes de luminancia para rechazar sombras falsas.
3. **Rastreo Fiducial (Fiducial Tracking):** Detección de códigos de barras "ArUco" pegados en los chasis para identificar positivamente a los robots "Aliados".
4. **Deducción Espacial (Spatial Deduction):** Cualquier obstáculo o mancha en el campo que no tenga un ArUco de aliado, se clasifica inmediatamente por descarte como un "Enemigo" (Threat).
5. **Estimación de Distancias:** Utiliza la fórmula de proyección equidistante de la lente ojo de pez para transformar el tamaño aparente de los píxeles en distancias físicas reales (en metros).

### Retos Críticos Superados
*   **Distorsión Fisheye:** Las lentes de 200° causan una distorsión radial masiva no lineal, compensada con calibración matricial.
*   **Amnesia de Oclusión:** Para evitar que la IA "olvide" a los enemigos si parpadean un milisegundo, se utiliza memoria a corto plazo (coasting) basada en filtros de Kalman (configurable vía `kalman_timeout_sec`).
*   **Falsos Positivos por Sombras:** Las sombras proyectadas en el pasto se filtran comprobando la densidad de los bordes internos (filtros Canny) y el umbral de luminancia.

---

## Variables Físicas y Normalización (State Geometry)

El motor de inferencia (`software/models/abide/inferencia.py`) calcula un "Estado Físico" hiper-detallado en tiempo real. Estas variables son la única forma en la que la IA percibe el mundo exterior.

### Variables Clave de Telemetría y Entorno
1. **`enemy_pressure_level` (Nivel de Presión del Enemigo):**
   * **Rango:** `0.0` (Lejos/Seguro) a `1.0` (Pánico / Contacto Inminente).
   * **Lógica:** Se activa cuando un enemigo cruza la barrera de los 90cm de cercanía. Actúa como un "Gatillo de Estrés" para forzar a la IA a pasar el balón o realizar evasiones de emergencia.
2. **`enemy_blocking_lane` (Carril Bloqueado):**
   * **Rango:** Booleano `0.0` (Libre) o `1.0` (Bloqueado).
   * **Lógica:** Marca `1.0` si un enemigo está *más cerca* que el balón, y además se encuentra físicamente atravesado en la misma línea de visión directa hacia la pelota (cono visual de 25 grados).
3. **`near_boundary_risk` (Riesgo de Límite de Cancha):**
   * **Rango:** `0.0` (Seguro) a `1.0` (Fuera de la cancha).
   * **Lógica:** Utiliza la posición XY del robot para medir su distancia al centro. Si el robot sale del área segura del 80% central, el riesgo sube hacia 1.0, enseñando a la red neuronal a mantenerse lejos de las bardas y no salir de las líneas blancas.
4. **Distancias Físicas (`ally_distance_norm`, `enemy1_distance_norm`, `blue/yellow_goal_distance_m`):**
   * **Unidad de Medida:** **Metros (m)** estrictos (estandarizado en toda la pipeline).
5. **Triangulación y Localización Absoluta (`ego_x`, `ego_y_abs`):**
   * Al no tener GPS, el robot deduce su coordenada cartesiana exacta aplicando el *Teorema de los Cosenos* a las distancias observadas hacia ambas porterías (cuya distancia total de separación es fija en 2.4 metros).

### Estrategia de Normalización para la Red Neuronal (NN)
Para evitar que números gigantes colapsen matemáticamente los pesos de la IA, **TODAS las variables entran a la red comprimidas en un rango estricto de `0.0` a `1.0` o `-1.0` a `1.0`:**
*   **Conversión de Ángulos (Grados):** Los ángulos crudos (de `-180°` a `180°`) se normalizan fraccionalmente con la fórmula: `(Angulo + 180) / 360`. El frente es `0.5`.
*   **Conversión de Distancias (Metros):** La distancia máxima posible en la cancha es `3.0 metros`. Se dividen los metros crudos entre `3.0`. En ocasiones (para `abide_run.py`), esta matemática se invierte haciendo `(1.0 - Distancia_Normalizada)` para que el número más alto (`1.0`) signifique "Peligro Inminente / Cercanía Extrema".

---

## Arquitectura de IA y el Enjambre (Ensemble Neural Network)

La pipeline de Machine Learning (ML) fusiona el Estado Físico en **3 Codificadores Semánticos**, los cuales alimentan un **Codificador Temporal**, que finalmente alimenta el **Motor de Acción (Policy)**.

### Dimensiones y Modelos (TensorFlow Lite)
*   **`NET_A` (Contexto de Objetivos / Balón):** 10 entradas -> 4 dimensiones latentes.
*   **`NET_B` (Contexto del Ego / Propio):** 14 entradas -> 4 dimensiones latentes.
*   **`NET_C` (Contexto de Enemigos / Amenazas):** 12 entradas -> 4 dimensiones latentes.
*   **`NET_T` (Contexto Temporal):** Agrupa el historial de los últimos 20 fotogramas de A+B+C y escupe memoria de corto plazo -> 5 dimensiones latentes.
*   **`ENTHRALLED` (Motor de Acción final):** Toma las dimensiones condensadas de A, B, C y T (17 dimensiones en total) y toma la decisión final, arrojando **9 comandos de control**.

### Los Comandos de Salida (Lo que piensa la IA)
El orquestador produce las siguientes salidas para controlar físicamente al robot:
1. **`vx` (Forward Speed):** Aceleración frontal hacia adelante (positiva) o reversa para escapar (negativa).
2. **`vy` (Strafe Speed):** Desplazamiento lateral (izquierda/derecha) usando llantas omnidireccionales.
3. **`omega` (Turn Rate):** Giro sobre su propio eje central para apuntar hacia otros lugares.
4. **`kick` (Fuerza de Pateo):** Comando de carga y disparo del solenoide magnético (0.0 a 1.0).
5. **Indicadores de Personalidad / Táctica:**
   * **`urgency`:** Mide qué tan apremiante es la situación (ej. poco tiempo).
   * **`emergency`:** Se dispara a 1.0 cuando hay colisiones inminentes; anula otras acciones y fuerza maniobras evasivas.
   * **`aggression` vs `defense`:** Define si el robot debe atacar al balón o retroceder a cubrir su propia portería.
   * **`pass_pref`:** Si es alto, el robot optará por girar para buscar un Aliado en lugar de disparar a puerta.

---
*Desarrollado para la superioridad robótica autónoma.* 
