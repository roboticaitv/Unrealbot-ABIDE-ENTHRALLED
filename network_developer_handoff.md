# Neural Network Developer Handoff Notes

This document compiles all recent architectural findings, design decisions, and necessary improvements for the ABIDE-ENTHRALLED neural network ensemble. It is intended directly for the ML engineer designing and training the networks.

---

## 1. Dimensionality & The Architecture Pipeline
The physical outputs from the Vision script are currently fused into 3 intermediate Semantic Encoders, which feed a Temporal Encoder, which feeds the final Action Policy. The exact dimensionality is strictly enforced by the ONNX orchestrator:

*   **`NET_A` (Ball/Objective Context):** Outputs **6** dimensions.
*   **`NET_B` (Ego/Self Context):** Outputs **8** dimensions.
*   **`NET_C` (Enemy/Threat Context):** Outputs **6** dimensions.
*   **`NET_T` (Temporal Context):** Takes the combined 20 dims (A+B+C) and outputs **12** dimensions.
*   **`ENTHRALLED` (Action Engine):** Takes all **32** dimensions (A+B+C+T) and outputs the final motor commands (`vx, vy, omega, kick, urgency, aggression, defense`).

**Crucial Check:** Ensure all `OUTPUT_DIM` constants in the synthetic generators exactly match these lengths. Any mismatch crashes the ONNX runtime concatenation.

---

## 2. Deployment Constraints & ONNX Conversion
Because the robot runs on a Raspberry Pi, we **cannot run native TensorFlow/Keras** in production (it consumes too much RAM and is too slow). The pipeline uses `onnxruntime`.

*   **The Keras 3 vs Legacy Bug:** If you save a `.h5` model using modern Keras 3, it uses a metadata tag called `batch_shape`. The `tf2onnx` converter strictly looks for `batch_input_shape` and will crash. 
*   **The Fix:** You must ensure your environment is set to use legacy Keras (`os.environ["TF_USE_LEGACY_KERAS"] = "1"`), or test your models through the `convert_all_to_onnx.py` script locally before committing them.
*   **Future Optimization:** If the Pi struggles to run all 5 ONNX models at high FPS, we will need to switch the deployment target from `ONNX` to **INT8 Quantized `.tflite`** files. Keep this in mind regarding activation functions (ReLU/Sigmoid quantize well).

---

## 3. Synthetic Data Generation: The "Noise" Problem
Currently, the synthetic generators (`AADFBS_NET_A.py`, etc.) produce perfect, mathematically flawless "oracle" data. For example, if a robot approaches at 1 m/s, the velocity input is a perfectly smooth 1.0.

**The Reality:** The Computer Vision pipeline calculates velocity based on bounding box centroids from a 200° fisheye lens. Frame-to-frame pixel noise causes these bounding boxes to jitter, which creates massive, instantaneous velocity spikes in the raw physics inputs. 

**Required Action:** 
*   You must implement Data Augmentation / Domain Randomization in the Python generators. 
*   Inject standard Gaussian noise into the inputs before feeding them to the network: `noisy_value = clean_value + np.random.normal(0, 0.15)`. 
*   This forces the Encoders to stop relying on exact, pristine scalar values and learn to smooth out the CV jitter naturally.

---

## 4. Upgrading `NET_T` to a Sequence Model (LSTM)
Currently, `NET_T` (the *Temporal* encoder) is just a standard Dense feed-forward network. Its input shape is exactly 1 instantaneous frame `(1, 20)`. 

**The Reality:** The camera frequently loses sight of the ball or enemy due to motion blur or occlusion for a few frames at a time. The physics engine resets the distance to `0.0`. With a feed-forward network, the robot experiences "teleportation" and instantly forgets the object existed.

**Required Action:** 
*   `NET_T` needs to be rebuilt as a recurrent network (e.g., `tf.keras.layers.LSTM` or `Conv1D`).
*   Instead of a single row, the input shape should be a rolling window of the last $N$ frames (e.g., `[batch_size, 10_frames, 20_dims]`).
*   This provides the robot with "Object Permanence." If the ball disappears in Frame 10, the LSTM's hidden state remembers it was there in Frames 1-9 and maintains the trajectory threat context.

---

## 5. Physics-Informed Loss Functions (Optional but Recommended)
To further combat camera jitter, consider adding a temporal smoothing penalty to the Loss Function when training the Encoders. If the network predicts `Defensive Urgency = 0.9` in frame $t$, and `0.1` in frame $t+1$, the loss function should heavily penalize that delta. The embeddings should flow smoothly regardless of input noise.
