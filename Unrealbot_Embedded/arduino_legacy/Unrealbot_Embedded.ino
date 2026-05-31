#include "FUTBOLITO.h"

// Servo Object
#define SERVO_PIN 46  // Choose any free GPIO
SimpleServo kicker(46, 4);
void IRAM_ATTR supervisorTimerISR() {
  if (SupervisorTaskHandle != NULL) {
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(SupervisorTaskHandle, &woken);
    if (woken) portYIELD_FROM_ISR();
  }
}

void IRAM_ATTR controlTimerISR() {
  if (ControlTaskHandle != NULL) {
    BaseType_t woken = pdFALSE;
    vTaskNotifyGiveFromISR(ControlTaskHandle, &woken);
    if (woken) portYIELD_FROM_ISR();
  }
}

void setup() {
  Serial.begin(1000000);
  delay(1000);  // Give serial monitor time to connect
  pinMode(SOLENOID_PIN, OUTPUT);
  // ADC Config
  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  // Init Motors
  motorFL.begin();
  motorFR.begin();
  motorRL.begin();
  motorRR.begin();

  // Init Servo
  kicker.begin();
  Serial.println("System Ready: 4 Motors + 1 Servo");
  Serial.printf("Reset reason: %d\n", esp_reset_reason());
  enc1.begin();
  enc2.begin();
  enc3.begin();

  /*
  // Initialize SPI with custom pins and high speed
  // Fix 2: Global overrides deleted. CS set to -1.
  SPI.begin(SPI_SCK_PIN, SPI_MISO_PIN, SPI_MOSI_PIN, -1);
  Serial.println("\n✓ SPI initialized");

  // Initialize the IMU
  Serial.println("Initializing IMU...");
  int ret = IMU.begin();
  if (ret != 0) {
    Serial.print("ERROR: IMU initialization failed, code: ");
    Serial.println(ret);
    while (1) delay(1000);
  }
  IMU.startAccel(SAMPLE_RATE_HZ, ACCEL_RANGE_G);
  IMU.startGyro(SAMPLE_RATE_HZ, GYRO_RANGE_DPS);
  Serial.println("✓ IMU initialized");

  // Fix 1: Added the Flow sensor initialization back!
  Serial.println("Initializing Flow Sensor...");
  if (!flow.begin()) {
    Serial.println("ERROR: Flow sensor initialization failed");
    while (1) delay(1000);
  }
  Serial.println("✓ Flow sensor initialized");
*/

  sampleQueue = xQueueCreate(1, sizeof(encoder_sample_t));
  commandQueue = xQueueCreate(1, sizeof(robot_command_t));

  if (sampleQueue == NULL || commandQueue == NULL) {
    Serial.println("FATAL: Failed to create Queues!");
    while (1) delay(100);
  }
  if (sampleQueue == NULL) {
    Serial.println("FATAL: Failed to create sampleQueue!");
    while (1) delay(100);
  }
  // Placed on CORE 0: Handles the high-speed 1kHz sensor polling
  xTaskCreatePinnedToCore(
    MotionSupervisor,       // Function to implement the task
    "Supervisor",           // Name of the task
    16384,                  // Stack size in words
    NULL,                   // Task input parameter
    3,                      // Priority (Highest for data gathering)
    &SupervisorTaskHandle,  // Task handle
    1);

  // Placed on CORE 1: Handles the 500Hz Math, PID, and Motor PWM
  xTaskCreatePinnedToCore(
    MotionControl,       // Function to implement the task
    "Control",           // Name of the task
    16384,               // Stack size in words
    NULL,                // Task input parameter
    2,                   // Priority (High, but below supervisor)
    &ControlTaskHandle,  // Task handle
    1);
  MotionSupervisorTimer = timerBegin(1, 80, true);
  timerAttachInterrupt(MotionSupervisorTimer, &supervisorTimerISR, true);
  timerAlarmWrite(MotionSupervisorTimer, SUPERVISOR_US, true);
  timerAlarmEnable(MotionSupervisorTimer);

  MotionControlTimer = timerBegin(0, 80, true);
  timerAttachInterrupt(MotionControlTimer, &controlTimerISR, true);
  timerAlarmWrite(MotionControlTimer, CONTROL_US, true);
  timerAlarmEnable(MotionControlTimer);
  last_snapshot_us = micros();

  // Turn on the Wi-Fi radio in Station Mode
  WiFi.mode(WIFI_STA);

  // FORCE BOTH BOARDS TO CHANNEL 1
  esp_wifi_set_promiscuous(true);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  esp_wifi_set_promiscuous(false);
  // Init ESP-NOW
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }

  // Register callbacks
  esp_now_register_send_cb(esp_now_send_cb_t(OnDataSent));
  esp_now_register_recv_cb(esp_now_recv_cb_t(OnDataRecv));

  // Register peer
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;
  peerInfo.encrypt = false;

  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    return;
  }

  Serial.println("ESP-NOW bridge ready. Type and send to transmit.");
  motorRR.setSpeed(255);
}

//modify mt6701 class to read encoder readings and modify sample queue so it is a struct of 3 readings.
void MotionSupervisor(void *parameters) {
  esp_task_wdt_add(NULL);
  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    esp_task_wdt_reset();
    unitFL.supervisorTick(unitFL.control_did_read);
    unitFR.supervisorTick(unitFR.control_did_read);
    unitRL.supervisorTick(unitRL.control_did_read);
    unitFL.control_did_read = false;
    unitFR.control_did_read = false;
    unitRL.control_did_read = false;

    // Build combined sample and overwrite queue
    encoder_sample_t s;
    s.fl_ticks = unitFL.accumulated_ticks;
    s.fr_ticks = unitFR.accumulated_ticks;
    s.rl_ticks = unitRL.accumulated_ticks;
    xQueueOverwrite(sampleQueue, &s);
  }
}

// Global Odometry State
float current_x = 0.0f, current_y = 0.0f, current_th = 0.0f;

void MotionControl(void *parameters) {
  const float dt = CONTROL_US / 1e6f;
  int32_t prev_fl_ticks = 0, prev_fr_ticks = 0, prev_rl_ticks = 0;
  uint16_t print_divider = 0;

  for (;;) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    esp_task_wdt_reset();
    uint32_t loop_start_us = micros();
    float raw_vel_fl = 0.0f, raw_vel_fr = 0.0f, raw_vel_rl = 0.0f;

    // 1. GET SYNCHRONIZED HARDWARE STATE
    encoder_sample_t current_state = {};
    if (xQueueReceive(sampleQueue, &current_state, 0) == pdTRUE) {
      int32_t delta_fl_ticks = current_state.fl_ticks - prev_fl_ticks;
      int32_t delta_fr_ticks = current_state.fr_ticks - prev_fr_ticks;
      int32_t delta_rl_ticks = current_state.rl_ticks - prev_rl_ticks;

      prev_fl_ticks = current_state.fl_ticks;
      prev_fr_ticks = current_state.fr_ticks;
      prev_rl_ticks = current_state.rl_ticks;

      float dist_fl = (delta_fl_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);
      float dist_fr = (delta_fr_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);
      float dist_rl = (delta_rl_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);

      updateOdometry(dist_fl, dist_fr, dist_rl, current_x, current_y, current_th);

      raw_vel_fl = (delta_fl_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;
      raw_vel_fr = (delta_fr_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;
      raw_vel_rl = (delta_rl_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;

      unitFL.filtered_velocity = unitFL.alpha * raw_vel_fl + (1.0f - unitFL.alpha) * unitFL.filtered_velocity;
      unitFL.angular_velocity_rads = unitFL.filtered_velocity;

      unitFR.filtered_velocity = unitFR.alpha * raw_vel_fr + (1.0f - unitFR.alpha) * unitFR.filtered_velocity;
      unitFR.angular_velocity_rads = unitFR.filtered_velocity;

      unitRL.filtered_velocity = unitRL.alpha * raw_vel_rl + (1.0f - unitRL.alpha) * unitRL.filtered_velocity;
      unitRL.angular_velocity_rads = unitRL.filtered_velocity;
    }

    // 2. DETERMINE DESIRED ROBOT VELOCITY (RC Mode Only!)
    // The "Virtual Carrot" we are dragging around with the controller
    // 2. DETERMINE DESIRED ROBOT VELOCITY (RC Mode Only!)
    static float target_pose_x = 0.0f;
    static float target_pose_y = 0.0f;
    static float target_pose_th = 0.0f;

    // Sync the carrot to the robot on the very first loop so it doesn't bolt away
    static bool first_loop = true;
    if (first_loop) {
      target_pose_x = current_x;
      target_pose_y = current_y;
      target_pose_th = current_th;
      first_loop = false;
    }

    robot_command_t rc_cmd = {};  // Initialize to empty

    if (xQueuePeek(commandQueue, &rc_cmd, 0) == pdTRUE) {
      // FIX 3: THE IDLE SNAP
      // If the joysticks are perfectly centered, snap the carrot exactly to
      // the robot's current position so it doesn't drift away over time.
      if (fabs(rc_cmd.vx) < 0.01f && fabs(rc_cmd.vy) < 0.01f) {
        target_pose_x = current_x;
        target_pose_y = current_y;
      } else {
        target_pose_x += rc_cmd.vx * dt;
        target_pose_y += rc_cmd.vy * dt;
      }
      float dist_to_carrot = hypot(target_pose_x - current_x, target_pose_y - current_y);
      const float MAX_LEASH = 0.15f;

      if (dist_to_carrot > MAX_LEASH) {
        float angle_to_carrot = atan2(target_pose_y - current_y, target_pose_x - current_x);
        target_pose_x = current_x + cosf(angle_to_carrot) * MAX_LEASH;
        target_pose_y = current_y + sinf(angle_to_carrot) * MAX_LEASH;
      }
      // ---------------------------------------------------------

      target_pose_th = rc_cmd.omega;
      target_pose_th = normalize_angle(target_pose_th);
    }

    // --- CALCULATE VELOCITIES NEEDED TO CATCH THE TARGET POSE ---

    // Proportional tuning gains (Adjust these to make it softer or more aggressive)
    const float Kp_xy = 5.0f;
    const float Kp_th = 2.0f;
    // Clamp the maximum output speeds here so it doesn't go dangerously fast if the joystick is held down for too long.
    const float MAX_MPS = 2.0f;  // Max 2 meters per second
    // 1. Find the global error (Where am I vs Where is the carrot?)
    float err_x = target_pose_x - current_x;
    float err_y = target_pose_y - current_y;
    float err_th = normalize_angle(target_pose_th - current_th);

    // 2. Rotate the global error into the Robot's local perspective
    // (If the carrot is +1m North, but I'm facing West, I need to strafe Right!)
    float cos_th = cosf(current_th);
    float sin_th = sinf(current_th);

    float local_err_x = (cos_th * err_x + sin_th * err_y);
    float local_err_y = (-sin_th * err_x + cos_th * err_y);

    // 3. Command velocities proportional to how far away the carrot is
    float target_vx = local_err_x * Kp_xy;
    float target_vy = local_err_y * Kp_xy;
    float target_omega = err_th * Kp_th;


    if (target_vx > MAX_MPS) target_vx = MAX_MPS;
    if (target_vx < -MAX_MPS) target_vx = -MAX_MPS;
    if (target_vy > MAX_MPS) target_vy = MAX_MPS;
    if (target_vy < -MAX_MPS) target_vy = -MAX_MPS;

    // 3. INVERSE KINEMATICS
    WheelTargets targets = omni_ik(target_vx, target_vy, target_omega);

    float target_rads_fl = targets.fl / WHEEL_RADIUS;
    float target_rads_fr = targets.fr / WHEEL_RADIUS;
    float target_rads_rl = targets.rl / WHEEL_RADIUS;

    // 4. PID COMPUTE
    int pwm_fl = unitFL.pid.compute(target_rads_fl, unitFL.angular_velocity_rads, dt);
    int pwm_fr = unitFR.pid.compute(target_rads_fr, unitFR.angular_velocity_rads, dt);
    int pwm_rl = unitRL.pid.compute(target_rads_rl, unitRL.angular_velocity_rads, dt);

    // 5. MOTOR OUTPUT
    unitFL.setSpeed(pwm_fl);
    unitFR.setSpeed(pwm_fr);
    unitRL.setSpeed(pwm_rl);

    // ---------------------------------------------------------
    // SOLENOID KICKER LOGIC (Non-blocking & Edge-Detected)
    // ---------------------------------------------------------
    static uint32_t kick_timer = 0;
    static bool is_kicking = false;
    static uint32_t current_kick_duration = 0;
    static uint8_t previous_kick_strength = 0;  // <--- ADD THIS

    // Only fire if the button is PRESSED NOW, but WASN'T PRESSED last loop
    if (rc_cmd.kick_strength > 0 && previous_kick_strength == 0 && !is_kicking) {
      is_kicking = true;
      kick_timer = millis();

      if (rc_cmd.kick_strength == 1) current_kick_duration = 50;
      if (rc_cmd.kick_strength == 2) current_kick_duration = 75;
      if (rc_cmd.kick_strength == 3) current_kick_duration = 250;

      digitalWrite(SOLENOID_PIN, HIGH);
    }

    // Remember the button state for the next loop
    previous_kick_strength = rc_cmd.kick_strength;

    // Turn off the solenoid exactly when the requested duration finishes
    if (is_kicking && (millis() - kick_timer > current_kick_duration)) {
      digitalWrite(SOLENOID_PIN, LOW);
      is_kicking = false;
    }

    // 7. DEBUG PRINTING (10Hz)
    uint32_t loop_time_us = micros() - loop_start_us;
    print_divider++;
    if (print_divider >= 50) {
      Serial.println(loop_time_us);
      Serial.printf("X:%.2f  Y:%.2f  Th:%.2f\n", current_x, current_y, current_th);
      Serial.printf("Px:%.2f  Tpy:%.2f  TPW:%.2f\n", target_pose_x, target_pose_y, target_pose_th);
      Serial.printf("Vx:%.2f  Vy:%.2f  W:%.2f\n", target_vx, target_vy, target_omega);
      Serial.printf("T:%6.2f|A:%6.2f|PWM: %4d\n", target_rads_fl, unitFL.angular_velocity_rads, pwm_fl);
      Serial.printf("T:%6.2f|A:%6.2f|PWM: %4d\n", target_rads_fr, unitFR.angular_velocity_rads, pwm_fr);
      Serial.printf("T:%6.2f|A:%6.2f|PWM: %4d\n", target_rads_rl, unitRL.angular_velocity_rads, pwm_rl);
      print_divider = 0;
    }
  }
}

void loop() {
  vTaskDelay(pdMS_TO_TICKS(10));
}



/*  unsigned long currentMicros = micros();
  unsigned long currentMillis = millis();

  // ---------------------------------------------------------
  // 1. HIGH SPEED TASK: Read IMU at exactly 800Hz (1250 µs)
  // ---------------------------------------------------------
  if (currentMicros - lastImuTime >= 1250) {
    lastImuTime = currentMicros;

    // UNCOMMENTED data fetch
    int ret = IMU.getDataFromRegisters(imu_data);
    if (ret == 0) {
      sample_count++;
      // We don't print here! It's too fast. We just store it in imu_data.
    }
  }

  // ---------------------------------------------------------
  // 2. LOW SPEED TASK: Flow, Encoders, & Prints at 50Hz (20ms)
  // ---------------------------------------------------------
  if (currentMillis - lastPrintTime >= 20) {
    lastPrintTime = currentMillis;

    // Read Flow Sensor (safe to do at 50Hz)
    flow.readMotionCount(&deltaX, &deltaY);

    // Print Odometry
    Serial.printf("Flow X:%d Y:%d | E1:%ld E2:%ld E3:%ld\n",
                  deltaX, deltaY, enc1.getCount(), enc2.getCount(), enc3.getCount());

    // Print the most recent IMU data
    Serial.printf("IMU AX:%f AY:%f AZ:%f GX:%f GY:%f GZ:%f\n",
                  imu_data.accel_data[0], imu_data.accel_data[1], imu_data.accel_data[2],
                  imu_data.gyro_data[0], imu_data.gyro_data[1], imu_data.gyro_data[2]);
  }

  // ---------------------------------------------------------
  // 3. ASYNCHRONOUS TASKS: Buttons, Serial Comms, Servo
  // ---------------------------------------------------------

  // Sweep servo
  int servoPos = 90 + 45 * sin(currentMillis / 1000.0);
  kicker.write(servoPos);

  // Read incoming Serial commands
  if (Serial.available() > 0) {
    int val = Serial.parseInt();
    while (Serial.available()) Serial.read();  // Clear buffer
    motorFL.setSpeed(val);
    motorFR.setSpeed(val);
    motorRL.setSpeed(val);
    motorRR.setSpeed(val);
  }

  // Handle Interrupt Flags
uint32_t now = millis();

// Handle Interrupt Flags
if (flagA) {
  if (now - lastDebounceA > debounceDelay) {
    Serial.println("Button A Pressed");
    lastDebounceA = now;
  }
  flagA = false;
}

if (flagB) {
  if (now - lastDebounceB > debounceDelay) {
    Serial.println("Button B Pressed");
    lastDebounceB = now;
  }
  flagB = false;
}

if (flagC) {
  if (now - lastDebounceC > debounceDelay) {
    Serial.println("Button C Pressed");
    lastDebounceC = now;
  }
  flagC = false;
}

if (flagD) {
  if (now - lastDebounceD > debounceDelay) {
    Serial.println("Button D Pressed");
    lastDebounceD = now;
  }
  flagD = false;
}
  */