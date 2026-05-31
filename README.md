# Unrealbot & ABIDE-ENTHRALLED 

Welcome to the central integration repository for the **Unrealbot** and **ABIDE-ENTHRALLED** project ecosystem. This monorepo serves as the final unified codebase for the robot's hardware control, telemetry, station communication, and AI vision systems.

## 🏗️ Architecture & Components

The project is highly modular, split between high-level computation (AI/Vision), remote telemetry, and low-level embedded hardware control.

### 1. Embedded Firmware (`/Unrealbot_Embedded`)
The core firmware running on the robot's ESP32 coprocessor.
- **Framework:** ESP-IDF (Currently being migrated from Arduino)
- **Responsibilities:** Motor control (via DRV8251), PID loops, hardware interrupts, kinematics, and receiving/executing movement commands.

### 2. Manual Control & Telemetry (`/Controllerinput`)
A Python-based application for remote telemetry and manual control.
- **Responsibilities:** Captures gamepad/keyboard inputs and sends them to the station. Monitors telemetry data coming back from the bot.

### 3. Base Station Bridge (`/Station_ESPNOW`)
Code for the stationary ESP32 base station.
- **Responsibilities:** Acts as a wireless bridge using the ESP-NOW protocol. It receives commands from the `Controllerinput` script via serial and transmits them wirelessly to the `Unrealbot_Embedded` ESP32 on the robot.

### 4. AI & Vision (`/vision`, `/*.py`, etc.)
- **Responsibilities:** Neural network models and computer vision pipelines (e.g., ball and goal recognition) for autonomous decision making.

## 🚀 Getting Started

*(Add detailed setup instructions for ESP-IDF, Python environment, and AI models here as they are developed)*
