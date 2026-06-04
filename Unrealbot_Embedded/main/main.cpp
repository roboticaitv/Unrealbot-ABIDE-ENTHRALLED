#include <stdio.h>
#include <math.h>
#include <fcntl.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "esp_system.h"
#include "futbolito.h"
#include "spi_sensors.h"
#include "espnow_comms.h"
#include "PMW3901.h"
#include "ICM45686.h"
#include "driver/uart.h"

static const char* TAG = "MAIN";

SPIClass SPI;

// Global Sensors
Custom_PMW3901 flow;
ICM456xx IMU(SPI, IMU_CS_PIN);
inv_imu_sensor_data_t imu_data;

// Global Odometry State
float current_x = 0.0f, current_y = 0.0f, current_th = 0.0f;

TaskHandle_t RobotLoopTaskHandle = NULL;

void RobotLoopTask(void *parameters) {
    uint8_t dma_imu_buf[14];
    
    const float dt = 2000 / 1e6f; // 500Hz = 2ms
    int32_t prev_fl_ticks = 0, prev_fr_ticks = 0, prev_rl_ticks = 0;
    uint16_t print_divider = 0;
    
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(1); // 1ms precise interval
    uint32_t tick_counter = 0;

    // Safety timeout state
    uint64_t last_cmd_time = esp_timer_get_time();
    robot_command_t rc_cmd = {};

    for (;;) {
        // --- PRECISE 1kHz YIELDING SLEEP ---
        // This completely replaces the busy-wait, dropping CPU usage from 100% to near 0%,
        // allowing the IDLE1 task to run and keeping the FreeRTOS Task Watchdog completely happy!
        vTaskDelayUntil(&xLastWakeTime, xFrequency);
        
        tick_counter++;

        // ==========================================
        // 1. 1kHz SUPERVISOR LOGIC (Every 1ms)
        // ==========================================
        // trigger_dma_spi_reads(); // Temporarily disabled
        
        unitFL.supervisorTick(false);
        unitFR.supervisorTick(false);
        unitRL.supervisorTick(false);

        // fetch_dma_spi_reads(dma_imu_buf); // Temporarily disabled
        // TODO: IMU raw data unpacking goes here!

        // ==========================================
        // 2. 500Hz CONTROL LOGIC (Every 2ms)
        // ==========================================
        if (tick_counter % 2 == 0) {
            
            // --- ESP-NOW Command Processing & Safety Timeout ---
            robot_command_t fresh_cmd;
            // Using Peek for a Mailbox pattern. It doesn't remove the item, 
            // so we MUST check the timestamp to see if it's genuinely new.
            if (xQueuePeek(commandQueue, &fresh_cmd, 0) == pdTRUE) {
                if (fresh_cmd.timestamp != rc_cmd.timestamp) {
                    rc_cmd = fresh_cmd;
                    last_cmd_time = esp_timer_get_time();
                }
            }

            // SAFETY TIMEOUT: If no command in 250ms, stop the robot!
            if (esp_timer_get_time() - last_cmd_time > 250000) {
                rc_cmd.mode = 0;
                rc_cmd.arg1 = 0.0f;
                rc_cmd.arg2 = 0.0f;
                rc_cmd.arg3 = 0.0f;
                rc_cmd.kick_strength = 0;
            }

            // --- Read latest encoder states directly ---
            int32_t current_fl = unitFL.accumulated_ticks;
            int32_t current_fr = unitFR.accumulated_ticks;
            int32_t current_rl = unitRL.accumulated_ticks;

            int32_t delta_fl_ticks = current_fl - prev_fl_ticks;
            int32_t delta_fr_ticks = current_fr - prev_fr_ticks;
            int32_t delta_rl_ticks = current_rl - prev_rl_ticks;

            prev_fl_ticks = current_fl;
            prev_fr_ticks = current_fr;
            prev_rl_ticks = current_rl;

            float dist_fl = (delta_fl_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);
            float dist_fr = (delta_fr_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);
            float dist_rl = (delta_rl_ticks / TICKS_PER_REV) * (2.0f * PI * WHEEL_RADIUS);

            updateOdometry(dist_fl, dist_fr, dist_rl, current_x, current_y, current_th);

            float raw_vel_fl = (delta_fl_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;
            float raw_vel_fr = (delta_fr_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;
            float raw_vel_rl = (delta_rl_ticks / TICKS_PER_REV) * (2.0f * PI) / dt;

            unitFL.filtered_velocity = unitFL.alpha * raw_vel_fl + (1.0f - unitFL.alpha) * unitFL.filtered_velocity;
            unitFL.angular_velocity_rads = unitFL.filtered_velocity;

            unitFR.filtered_velocity = unitFR.alpha * raw_vel_fr + (1.0f - unitFR.alpha) * unitFR.filtered_velocity;
            unitFR.angular_velocity_rads = unitFR.filtered_velocity;

            unitRL.filtered_velocity = unitRL.alpha * raw_vel_rl + (1.0f - unitRL.alpha) * unitRL.filtered_velocity;
            unitRL.angular_velocity_rads = unitRL.filtered_velocity;

            // --- Position Control ---
            static float target_pose_x = 0.0f;
            static float target_pose_y = 0.0f;
            static float target_pose_th = 0.0f;
            static bool first_loop = true;
            if (first_loop) {
                target_pose_x = current_x;
                target_pose_y = current_y;
                target_pose_th = current_th;
                first_loop = false;
            }

            if (rc_cmd.mode == 1) { // MODE_ABSOLUTE_POSE
                target_pose_x = rc_cmd.arg1;
                target_pose_y = rc_cmd.arg2;
                target_pose_th = rc_cmd.arg3;
                target_pose_th = normalize_angle(target_pose_th);
            } else { // MODE_VELOCITY
                if (fabs(rc_cmd.arg1) < 0.01f && fabs(rc_cmd.arg2) < 0.01f) {
                    target_pose_x = current_x;
                    target_pose_y = current_y;
                } else {
                    target_pose_x += rc_cmd.arg1 * dt;
                    target_pose_y += rc_cmd.arg2 * dt;
                }

                float dist_to_carrot = hypot(target_pose_x - current_x, target_pose_y - current_y);
                const float MAX_LEASH = 0.15f;

                if (dist_to_carrot > MAX_LEASH) {
                    float angle_to_carrot = atan2(target_pose_y - current_y, target_pose_x - current_x);
                    target_pose_x = current_x + cosf(angle_to_carrot) * MAX_LEASH;
                    target_pose_y = current_y + sinf(angle_to_carrot) * MAX_LEASH;
                }

                target_pose_th = rc_cmd.arg3;
                target_pose_th = normalize_angle(target_pose_th);
            }

            const float Kp_xy = 5.0f;
            const float Kp_th = 2.0f;
            const float MAX_MPS = 2.0f;

            float err_x = target_pose_x - current_x;
            float err_y = target_pose_y - current_y;
            float err_th = normalize_angle(target_pose_th - current_th);

            float cos_th = cosf(current_th);
            float sin_th = sinf(current_th);

            float local_err_x = (cos_th * err_x + sin_th * err_y);
            float local_err_y = (-sin_th * err_x + cos_th * err_y);

            float target_vx = local_err_x * Kp_xy;
            float target_vy = local_err_y * Kp_xy;
            float target_omega = err_th * Kp_th;

            // Positional Deadband (Stop fighting sub-centimeter errors)
            if (fabs(err_x) < 0.01f && fabs(err_y) < 0.01f && fabs(err_th) < 0.03f) {
                target_vx = 0.0f;
                target_vy = 0.0f;
                target_omega = 0.0f;
            }

            if (target_vx > MAX_MPS) target_vx = MAX_MPS;
            if (target_vx < -MAX_MPS) target_vx = -MAX_MPS;
            if (target_vy > MAX_MPS) target_vy = MAX_MPS;
            if (target_vy < -MAX_MPS) target_vy = -MAX_MPS;

            WheelTargets targets = omni_ik(target_vx, target_vy, target_omega);

            float target_rads_fl = targets.fl / WHEEL_RADIUS;
            float target_rads_fr = targets.fr / WHEEL_RADIUS;
            float target_rads_rl = targets.rl / WHEEL_RADIUS;

            int pwm_fl = unitFL.pid.compute(target_rads_fl, unitFL.angular_velocity_rads, dt);
            int pwm_fr = unitFR.pid.compute(target_rads_fr, unitFR.angular_velocity_rads, dt);
            int pwm_rl = unitRL.pid.compute(target_rads_rl, unitRL.angular_velocity_rads, dt);

            // Anti-Twitch: If target is 0, explicitly kill the PWM and wipe the integral windup
            if (fabs(target_rads_fl) < 0.01f) { pwm_fl = 0; unitFL.pid.reset(); }
            if (fabs(target_rads_fr) < 0.01f) { pwm_fr = 0; unitFR.pid.reset(); }
            if (fabs(target_rads_rl) < 0.01f) { pwm_rl = 0; unitRL.pid.reset(); }

            unitFL.setSpeed(pwm_fl);
            unitFR.setSpeed(pwm_fr);
            unitRL.setSpeed(pwm_rl);

            // --- Kicker Logic ---
            static uint32_t kick_timer = 0;
            static bool is_kicking = false;
            static uint32_t current_kick_duration = 0;
            static uint8_t previous_kick_strength = 0;

            if (rc_cmd.kick_strength > 0 && previous_kick_strength == 0 && !is_kicking) {
                is_kicking = true;
                kick_timer = (uint32_t)(esp_timer_get_time() / 1000ULL);

                if (rc_cmd.kick_strength == 1) current_kick_duration = 50;
                if (rc_cmd.kick_strength == 2) current_kick_duration = 75;
                if (rc_cmd.kick_strength == 3) current_kick_duration = 250;

                gpio_set_level((gpio_num_t)SOLENOID_PIN, 1);
            }
            previous_kick_strength = rc_cmd.kick_strength;

            if (is_kicking && ((uint32_t)(esp_timer_get_time() / 1000ULL) - kick_timer > current_kick_duration)) {
                gpio_set_level((gpio_num_t)SOLENOID_PIN, 0);
                is_kicking = false;
            }

            // --- Logging ---
            print_divider++;
            if (print_divider >= 50) {
                ESP_LOGI(TAG, "X:%.2f Y:%.2f Th:%.2f", current_x, current_y, current_th);
                print_divider = 0;
            }
        }

        // ==========================================
        // 3. 100Hz OPTICAL FLOW LOGIC (Every 10ms)
        // ==========================================
        if (tick_counter % 10 == 0) {
            // int16_t deltaX, deltaY;
            // flow.readMotionCount(&deltaX, &deltaY); // Temporarily disabled
            // Integrate optical flow here!
        }

        // Safe tick reset to prevent any crazy rollover math bugs
        if (tick_counter >= 1000) {
            tick_counter = 0;
        }
    }
}

void TelemetryTask(void *pvParameters) {
    TickType_t xLastWakeTime = xTaskGetTickCount();
    const TickType_t xFrequency = pdMS_TO_TICKS(20); // 50Hz Uplink

    telemetry_packet_t t_packet;
    t_packet.header = 'T';

    while (1) {
        vTaskDelayUntil(&xLastWakeTime, xFrequency);

        // Fetch Odometry
        t_packet.pose_x = current_x;
        t_packet.pose_y = current_y;
        t_packet.pose_th = current_th;

        // Fetch Kinematics
        float v_fl = unitFL.angular_velocity_rads * WHEEL_RADIUS;
        float v_fr = unitFR.angular_velocity_rads * WHEEL_RADIUS;
        float v_rl = unitRL.angular_velocity_rads * WHEEL_RADIUS;

        t_packet.vel_x = (-v_fl - v_fr + 2.0f * v_rl) / 3.0f;
        t_packet.vel_y = (-sqrt(3.0f) * v_fl + sqrt(3.0f) * v_fr) / 3.0f;
        t_packet.omega = (v_fl + v_fr + v_rl) / (3.0f * ROBOT_RADIUS);

        // Read Motor Currents for Hardware Fault Embeddings
        unitFL.motor->updateCurrentSense();
        unitFR.motor->updateCurrentSense();
        unitRL.motor->updateCurrentSense();
        
        t_packet.current_fl = unitFL.motor->getCurrent();
        t_packet.current_fr = unitFR.motor->getCurrent();
        t_packet.current_rl = unitRL.motor->getCurrent();

        // Placeholders for IMU / Flow / IR
        t_packet.accel_x = 0.0f;
        t_packet.accel_y = 0.0f;
        t_packet.gyro_z = 0.0f;
        t_packet.sensors = 0;
        
        t_packet.timestamp = esp_timer_get_time() / 1000;

        uint8_t encoded[128];
        size_t enc_len = cobs_encode((uint8_t*)&t_packet, sizeof(telemetry_packet_t), encoded);
        encoded[enc_len++] = 0x00;
        uart_write_bytes(UART_NUM_0, encoded, enc_len);
    }
}

void SerialCommandTask(void *pvParameters) {
    robot_command_t last_cmd = {};
    bool serial_active = false;

    uint8_t rx_buffer[128];
    size_t rx_idx = 0;

    while (1) {
        uint8_t byte;
        int len = uart_read_bytes(UART_NUM_0, &byte, 1, pdMS_TO_TICKS(10));
        
        if (len == 1) {
            if (byte == 0x00) {
                if (rx_idx > 0) {
                    uint8_t decoded[128];
                    size_t dec_len = cobs_decode(rx_buffer, rx_idx, decoded);
                    
                    if (dec_len > 0) {
                        uint8_t header = decoded[0];
                        if ((header == 'M' || header == 'm') && dec_len == 17) {
                            float payload[4];
                            memcpy(payload, decoded + 1, 16);
                            last_cmd.mode = 0;
                            last_cmd.arg1 = payload[0]; last_cmd.arg2 = payload[1]; last_cmd.arg3 = payload[2];
                            last_cmd.kick_strength = (uint8_t)payload[3];
                            serial_active = true;
                        } else if ((header == 'P' || header == 'p') && dec_len == 17) {
                            float payload[4];
                            memcpy(payload, decoded + 1, 16);
                            last_cmd.mode = 1;
                            last_cmd.arg1 = payload[0]; last_cmd.arg2 = payload[1]; last_cmd.arg3 = payload[2];
                            last_cmd.kick_strength = 0;
                            serial_active = true;
                        } else if (header == 'E' && dec_len == sizeof(msg_embeddings_t)) {
                            msg_embeddings_t emb;
                            memcpy(&emb, decoded, sizeof(msg_embeddings_t));
                            espnow_broadcast_embeddings(&emb);
                        } else if (header == 'S' || header == 's') {
                            serial_active = false;
                        }
                    }
                    rx_idx = 0;
                }
            } else {
                if (rx_idx < sizeof(rx_buffer)) {
                    rx_buffer[rx_idx++] = byte;
                } else {
                    rx_idx = 0; // Overflow, drop packet
                }
            }
        } else {
            // Timeout hit, feed the safety watchdog
            if (serial_active) {
                last_cmd.timestamp = esp_timer_get_time();
                xQueueOverwrite(commandQueue, &last_cmd);
            }
        }
    }
}

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting Unrealbot Embedded ESP-IDF (Single Task Architecture)!");
    
    // Setup GPIO for Solenoid+
    gpio_reset_pin((gpio_num_t)SOLENOID_PIN);
    gpio_set_direction((gpio_num_t)SOLENOID_PIN, GPIO_MODE_OUTPUT);
    
    // Install UART driver for COBS telemetry / command stream
    uart_driver_install(UART_NUM_0, 2048, 2048, 0, NULL, 0);

    setup_robot(); 
    // setup_spi(); // Temporarily disabled
    setup_espnow();
    
    // Initialize IMU
    ESP_LOGI(TAG, "Initializing IMU... (Temporarily Disabled)");
    /*
    int ret = IMU.begin();
    if (ret != 0) {
        ESP_LOGE(TAG, "IMU init failed: %d", ret);
    } else {
        IMU.startAccel(100, 16);
        IMU.startGyro(100, 2000);
        ESP_LOGI(TAG, "IMU initialized.");
    }
    */
    
    // Initialize Flow Sensor
    ESP_LOGI(TAG, "Initializing Flow Sensor... (Temporarily Disabled)");
    /*
    if (!flow.begin()) {
        ESP_LOGE(TAG, "Flow sensor init failed.");
    } else {
        ESP_LOGI(TAG, "Flow sensor initialized.");
    }
    */

    // Launch single massive robot loop pinned to Core 1 at highest non-system priority
    xTaskCreatePinnedToCore(RobotLoopTask, "RobotLoop", 32768, NULL, configMAX_PRIORITIES - 2, &RobotLoopTaskHandle, 1);

    // Launch Serial CLI Task pinned to Core 0 (shares CPU with WiFi/ESP-NOW)
    xTaskCreatePinnedToCore(SerialCommandTask, "SerialCLI", 8192, NULL, 5, NULL, 0);

    // Launch Telemetry Task pinned to Core 0
    xTaskCreatePinnedToCore(TelemetryTask, "Telemetry", 4096, NULL, 4, NULL, 0);

    // FreeRTOS idle loop for app_main
    while (1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
