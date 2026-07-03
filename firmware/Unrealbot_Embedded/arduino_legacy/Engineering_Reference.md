# Competitive Soccer Robotics: Engineering Reference Document

## 1. Chassis and Drive Systems

### 1.1 Drive Configurations

* **4-Wheel X-Drive (Recommended for Competitions):**
* **Advantage:** Mathematically 41.4% faster ($V_{total} = V_{motor} \cdot \sqrt{2}$). Four points of contact provide superior stability and shoving power against opponents. Leaves the front open for a wide dribbler mechanism.
* **Disadvantage:** Requires 4 motors/ESCs and increases weight.


* **3-Wheel Kiwi (Y) Drive:**
* **Advantage:** Lightweight, mathematically guaranteed ground contact on all three wheels, uses 25% fewer drive components.
* **Disadvantage:** "Glass cannon." Easier to push around, prone to tipping during side impacts, and complex asymmetric vector kinematics.



### 1.2 Motor & Gearbox Selection

* **4x 050 Motors with 90° Gearboxes:** The best choice for X-Drives. Hiding the motors on the perimeter lowers the center of gravity and frees up internal chassis volume for batteries and custom PCBs.
* **3x JGA25-310 (Inline):** Good for raw pushing power but creates a "dead zone" in the center of the robot, raising the center of gravity.

### 1.3 Omni-Wheel Dynamics

* **Dual-Row Heavy Duty (Grooved):** Essential for competitive carpet play. The dual rows provide constant ground contact to eliminate vibration (saving the IMU), while the grooves provide mechanical "bite" into the carpet.
* **Slim Single-Row (e.g., GTF):** Avoid for drive wheels. Causes a "cobblestone" vibration effect and lacks lateral friction.

---

## 2. Kicker System (High Voltage Electromechanics)

### 2.1 The Physics of Solenoids

* **Inductance & Voltage:** Solenoids resist rapid changes in current based on $V = L \frac{di}{dt}$. Dumping high voltage (e.g., 40V to 100V) forces the current to ramp up rapidly, generating a massive magnetic spike before the plunger reaches the end of its stroke.
* **Kinetic Energy:** Energy stored in the capacitor bank is dictated by $E = \frac{1}{2} C V^2$. Doubling the voltage quadruples the stored energy.
* **Plunger Mass:** Do not add dead weight to the plunger. Ideal energy transfer occurs when the striker mass is close to the projectile mass (a golf ball is ~46g, standard plungers are ~35-45g).

### 2.2 Component Upgrades (100V Architecture)

* **The Solenoid:** Upgrade from JF-0530B to **JF-0826B**. The 0826B has a thicker iron core (delays magnetic saturation) and a shorter stroke (maximizes initial magnetic field density).
* **Capacitor Bank:** Use **200V Aluminum Electrolytic Capacitors** (e.g., one 1500µF or three 470µF in parallel). Do **not** wire 63V capacitors in parallel for a 100V system; they will explode.
* **ZVS Boost Module:** Set to 100V and glue the trimpot. It will draw heavy inrush current from the LiPo, requiring thick wire gauges.

### 2.3 Custom PCB Safety Updates (For 100V)

1. **Input Resistor:** Must be a 5W or 10W wirewound cement resistor (e.g., $22\Omega$ to $100\Omega$) to handle the boost module's inrush.
2. **LED Resistor:** Must be upgraded to $10k\Omega$ or $15k\Omega$ and rated for at least **2 Watts** to prevent fires.
3. **Flyback Diode:** Upgrade to a 200V+ rating (e.g., MUR1520 or FES16DT). Remove any series resistor from the flyback path; let the diode dump the spike directly.
4. **Bleeder Resistor:** A mandatory $100k\Omega$, 1W resistor in parallel with the main capacitor bank to drain lethal voltage when the robot is off.

---

## 3. Logic & Control Systems

### 3.1 Line Detection Array (IR)

An architecture designed to offload processing from the ESP32 and guarantee hardware-level reaction times.

1. **Vref Generation:** ESP32 hardware PWM (LEDC) routed through an RC filter ($10k\Omega$ + $1\mu F$) creates a globally adjustable analog reference voltage.
2. **Comparators:** **LM339** (Quad Comparators) compare the IR phototransistor voltage against Vref. Requires pull-up resistors due to open-collector outputs.
3. **I2C Expander:** **MCP23017** reads the LM339 digital outputs. Configured for "Interrupt-on-Change" (INTA/INTB).
4. **FreeRTOS Flow:** Hardware interrupt triggers $\rightarrow$ unblocks FreeRTOS task $\rightarrow$ reads I2C bus $\rightarrow$ commands motor drivers.

### 3.2 Power Switching (MOSFETs)

* **For 40V:** Use **IRLZ48N**. It has logic-level gates and handles 210A peak pulses with a $V_{dss}$ of 55V.
* **For 100V:** Drop the IRLZ series. Use **IRL640** (200V, 18A).
* **For 150V (Goalie Sniper):** Drop MOSFETs entirely. Use **IGBTs** with opto-isolated gate drivers.

### 3.3 System Diagnostics & Monitoring

* **Motor Driver Logic:** ALWAYS include $10k\Omega$ pull-down resistors on PWM/DIR pins. Floating pins during boot cause EMI-induced motor twitching and stripped gears.
* **Debugging LEDs:** Use a single **WS2812B (NeoPixel)** daisy-chain driven by the ESP32's RMT peripheral. It uses only one GPIO pin but communicates infinite system states via color. Attach analog debugging LEDs *in parallel* on the LM339 outputs, never directly on analog sensor lines.
* **Battery Monitoring:** Use the **INA219** (I2C). It acts as a full wattmeter, tracking both voltage drops and instantaneous current spikes, which is vastly superior to the ESP32's noisy internal ADC.

---

## 4. Development & Execution Triage

To avoid wasted engineering hours, build and troubleshoot in this strict order:

1. **Kinematics (IK/DK):** Fix the math so the robot drives perfectly straight and rotates cleanly.
2. **Mechanical Dribbler:** Align the mount and tune the roller. If the robot cannot possess the ball, it cannot win.
3. **Chassis Redesign (Fusion360):** Combine the hollow base, ASA upgrades, battery lowering, and camera mounts into a single unified CAD update.
4. **Sensors:** Install the kicker break-beam and the IR boundary array.
5. **Tuning:** Optimize the high-voltage kicker and implement optical flow / IMU Kalman filters.
6. **Quality of Life:** Custom BMS chargers and hot-swaps (only if time permits).
