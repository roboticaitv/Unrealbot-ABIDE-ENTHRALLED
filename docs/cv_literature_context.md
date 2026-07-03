# Unrealbot Computer Vision: Architecture & Literature Context

This document outlines the current state of the Unrealbot Computer Vision (CV) pipeline. It is designed to be ingested by NotebookLM to aid in researching state-of-the-art literature for potential solutions, optimizations, and algorithmic upgrades.

## 1. Project Overview & Operational Context
Unrealbot is an autonomous, physical robotics platform operating in a dynamic environment (a competitive physical arena/field). The robot uses on-board vision to perceive the game state (ball, goals, allies, enemies) and feeds this state into a localized neural network ensemble (ONNX) to generate immediate motor commands. 

**Core Objective:** Maintain a high-FPS, low-latency, and highly reliable vision tracking pipeline running entirely on edge hardware.

## 2. Hardware & Performance Constraints
*   **Compute Unit:** Raspberry Pi (ARM Cortex CPU). The system operates on edge compute with strict thermal and power limits.
*   **Sensors:** Dual IMX219 camera sensors (stereo/side-by-side or overlapping field of view).
*   **Lenses:** 200° ultra-wide Fisheye lenses.
*   **Processing Budget:** The vision pipeline must run concurrently with hardware IO, state physics tracking, and 5 separate ONNX neural network inferences. 
*   **Language Environment:** Python (OpenCV, NumPy). Python's Global Interpreter Lock (GIL) currently forces most processing into a sequential execution flow, limiting the use of multi-core concurrency.

## 3. Current Techniques & Pipeline
1.  **Frame Capture:** Continuous background threading pulling YUV420 frames to avoid I/O blocking.
2.  **Color Tracking (Blob Detection):** 
    *   Uses classic OpenCV HSV/YUV thresholds (`cv2.inRange`) to isolate the ball (orange), goals (blue/yellow), and general "unknown" blobs.
    *   Bounding box extraction via contour analysis.
3.  **Fiducial Tracking:**
    *   Uses ArUco markers to definitively identify "Ally" robots.
    *   Runs intermittently (e.g., every 5-10 frames) to save CPU cycles.
4.  **Spatial Deduction (Hitbox Math):**
    *   If a color blob does not geometrically overlap with a known Ally ArUco position, it is heuristically classified as an "Enemy" threat.
5.  **Distance Estimation (Equidistant Model):**
    *   Uses an equidistant lens projection formula: `distance = known_size / (2 * tan(apparent_size_px / (2 * f_eq)))`.

## 4. Primary Challenges & Roadblocks (Focus Areas for Literature)

### A. Non-Linear Fisheye Distortion & Distance Estimation
**The Problem:** The 200° fisheye lenses introduce massive radial distortion. An object (like a ball) appears significantly smaller (fewer pixels) at the edge of the lens than it does in the exact center, even if it is the exact same physical distance from the camera. The current equidistant formula assumes linear pixel density, leading to wildly inaccurate distance estimations at the periphery.
**Literature Keywords:** *Fisheye lens calibration, omnidirectional camera distance estimation, radial distortion correction, un-distortion-free object detection, spherical camera models.*

### B. Lighting Vulnerability & Color Space Brittleness
**The Problem:** Hardcoded HSV/YUV bounds are highly sensitive to real-world lighting changes. A shadow cast over the field or a change in room lighting completely breaks the blob tracking. Dynamic thresholding or auto-exposure compensation is missing.
**Literature Keywords:** *Illumination invariant color tracking, dynamic color thresholding, robust blob detection in varying illumination, HSV shadow suppression, adaptive background subtraction.*

### C. Motion Blur & Fiducial Loss
**The Problem:** ArUco markers are highly susceptible to motion blur. When robots move fast or rotate quickly, the ArUco detector fails. Because the system relies on ArUco to classify allies, a blurred ally is suddenly perceived as a generic blob, causing the spatial deduction algorithm to classify it as an "Enemy."
**Literature Keywords:** *Motion blur robust fiducial tracking, high-speed marker detection, optical flow for marker tracking, Kalman filtering for occluded tracking, blurred ArUco recovery.*

### D. Temporal Jitter & State Physics Instability
**The Problem:** Bounding boxes "shake" frame-by-frame due to noise. Because we calculate velocity based on the frame-to-frame delta of these bounding boxes, the resulting velocity and acceleration vectors are incredibly noisy and spike wildly.
**Literature Keywords:** *Bounding box temporal smoothing, Kalman filters for 2D object tracking, Alpha-Beta filters, Optical Flow bounding box stabilization, noise reduction in discrete derivatives.*

### E. Edge Compute Orchestration
**The Problem:** Executing multiple OpenCV operations alongside heavy neural network inferences sequentially in Python creates a massive bottleneck.
**Literature Keywords:** *Raspberry Pi vision optimization, OpenCV Python multiprocessing pipeline, zero-copy memory Python computer vision, heterogeneous edge computing orchestration.*

## 5. Potential Prompts for NotebookLM
When querying the literature via NotebookLM, use the following prompts:
1. *"What are the most computationally lightweight methods for correcting radial distortion or estimating distance accurately on a 200-degree fisheye lens without running heavy un-distortion algorithms on the entire frame?"*
2. *"How can we implement illumination-invariant color tracking that adapts to shadows and lighting changes without utilizing heavy deep learning models?"*
3. *"What strategies exist for tracking ArUco markers or identifying specific robots during high-speed movement when severe motion blur is present?"*
4. *"What are the best practices for smoothing bounding box coordinates and deriving accurate velocity vectors from noisy vision detections in real-time robotics?"*
