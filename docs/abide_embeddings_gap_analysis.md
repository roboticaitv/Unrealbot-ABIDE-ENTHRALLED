# ABIDE Embeddings & Vision Pipeline: Gap Analysis & Training Strategies

To achieve true autonomous, robust behavior under the ABIDE framework, the physical reality parsed by the Computer Vision (CV) pipeline must map perfectly into the semantic embedding space expected by the neural networks (`NET_A`, `NET_B`, `NET_C`, and `NET_T`). 

Currently, there are critical disconnects between what the camera actually sees, what the physics math computes, and what the neural networks assume is happening. Below is a detailed gap analysis and a roadmap for synthetic training to fix these issues.

---

## 1. The Reality Gap (CV vs. Embeddings)

### Gap A: The Object Permanence Problem
*   **The Vision Reality:** The camera frequently loses the ball or enemy due to motion blur, lighting, or physical occlusion (e.g., an enemy blocks the ball). When this happens, `state_tracker.py` resets histories and outputs zeros.
*   **The Embedding Expectation:** `NET_C` calculates `Tracking confidence` and `Intercept window`. `NET_A` calculates `Ball trajectory threat`. 
*   **The Problem:** Because the physics state resets to zero the instant an object is occluded, the neural network experiences "teleportation." The robot will completely forget an enemy was charging at it if the enemy is obscured for even one frame.

### Gap B: Absolute vs. Relative Localization
*   **The Vision Reality:** The fisheye camera only provides *relative* distances (e.g., "The ball is 2 meters away at a 30-degree angle"). It does not know where the robot actually is on the physical field (X,Y coordinates).
*   **The Embedding Expectation:** `NET_B` calculates `Localization confidence` and `Field safety`. 
*   **The Problem:** Without absolute coordinates or a memory map, "Field safety" is essentially a guess based on how close the nearest enemy is, rather than understanding that the robot is backed into a corner of the physical arena.

### Gap C: Discrete Derivatives & Jitter
*   **The Vision Reality:** The bounding boxes "shake" due to pixel noise. `state_tracker.py` calculates velocity by subtracting the previous bounding box position from the current one. This discrete derivative turns a 2-pixel jitter into a massive, artificial speed spike.
*   **The Embedding Expectation:** The networks rely heavily on `enemy_velocity_norm` and `ball_speed_norm` to output defensive and aggressive behaviors.
*   **The Problem:** The robot will likely twitch or act erratically because the embeddings are constantly spiking due to bounding box noise misclassified as high-speed movement.

### Gap D: The Single-Goal Bias
*   **The Problem:** As noted previously, `shot_opportunity_ego` currently only calculates an intersection path to the `blue_goal`. The embeddings therefore have no concept of "which direction am I defending vs. attacking." 

---

## 2. Recommended Artificial Generation Scenarios

To train the encoders to survive these gaps, we need to upgrade the synthetic generators (`AADFBS_NET_A.py`, etc.) to simulate the messy reality of the CV pipeline.

### Scenario 1: The "Blinking Object" (Simulating Occlusion)
*   **Current Training:** The ball distance smoothly transitions from 0.0 to 1.0.
*   **New Scenario:** During generation, artificially drop the `ball_distance_norm` to `0.0` for 2-3 frames right in the middle of a high-speed approach. 
*   **Goal:** Teach `NET_T` (Temporal Context) to maintain a high `Emergency State` or `Defensive Urgency` even if the primary inputs momentarily vanish.

### Scenario 2: Gaussian Noise Injection (Simulating Jitter)
*   **Current Training:** `enemy_velocity_norm` is a clean, perfect float between 0.0 and 1.0.
*   **New Scenario:** Inject standard Gaussian noise (e.g., `np.random.normal(0, 0.15)`) into the velocity and distance inputs during training. 
*   **Goal:** Force the semantic encoders to decouple from exact scalar values and learn broader trends, preventing the final action policy from twitching every time a bounding box shakes.

### Scenario 3: The "Blind Sided" Edge Case
*   **New Scenario:** Generate states where `ego_pose_confidence` is high (we see the goals) but `enemy_observation_conf` is suddenly `0.0` while `ball_possession_ego` is `1.0`.
*   **Goal:** Teach the robot "Paranoia." If we have the ball but cannot see any enemies, the network should output a high `Exploration capability` (spin around to check blind spots) rather than assuming the field is safe.

---

## 3. Advanced Training Techniques

To truly harness the ABIDE framework, simple feed-forward dense networks (like the current `NET_T`) need architectural upgrades.

**A. Sequence Training for `NET_T` (LSTMs or 1D-CNNs)**
Since `NET_T` is the *Temporal* Context Encoder, it shouldn't just look at a single instantaneous frame of embeddings. It should look at a sliding window of the last 10 frames. 
*   *Technique:* Rebuild `NET_T` using `tf.keras.layers.LSTM` or `Conv1D`. Train it on synthetic *time-series* arrays rather than single rows. This explicitly solves the object permanence problem.

**B. Physics-Informed Loss Functions**
When training the Encoders, penalize the network for sudden, impossible changes.
*   *Technique:* If the network outputs `Ball trajectory threat = 0.9` in frame 1, and `0.1` in frame 2, add a massive penalty to the loss function during training. This forces the embeddings to remain smooth and stable, naturally ignoring the CV jitter.

**C. Curriculum Learning via Oracle Blending**
Instead of training on pure noisy data immediately, start by training the networks on perfect, clean synthetic data (Curriculum 1). Once validation accuracy peaks, slowly blend in the simulated noise, occlusions, and fisheye distortions (Curriculum 2). This prevents the network from failing to converge early on.
