#pragma once

#include "motor_control.h"
#include "encoder_pcnt.h"
#include "motor_unit.h"
#include "kinematics.h"
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/task.h"

// Hardware globals
extern DRV8251 motorFL;
extern DRV8251 motorFR;
extern DRV8251 motorRL;
extern DRV8251 motorRR;

extern EncoderPCNT enc1;
extern EncoderPCNT enc2;
extern EncoderPCNT enc3;

extern MotorUnit unitFL;
extern MotorUnit unitFR;
extern MotorUnit unitRL;

extern SimpleServo kicker;

// SPI Pins
#define SPI_MOSI_PIN 6
#define SPI_MISO_PIN 4
#define SPI_SCK_PIN 7
#define IMU_CS_PIN 5
#define FLOW_CS_PIN 10
#define SPI_CLOCK_HZ 10000000 // 10 MHz

#define SOLENOID_PIN 18

#define TICKS_PER_REV 37768.0f
#define WHEEL_RADIUS 0.06f
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// Queues & Types
struct encoder_sample_t {
  int32_t fl_ticks;
  int32_t fr_ticks;
  int32_t rl_ticks;
};

struct robot_command_t {
  uint8_t mode; // 0 = Velocity, 1 = Absolute Pose
  float arg1;
  float arg2;
  float arg3;
  uint8_t kick_strength;
  uint32_t timestamp;
};

struct telemetry_packet_t {
    uint8_t header; // Always 'T'
    float pose_x;
    float pose_y;
    float pose_th;
    float vel_x;
    float vel_y;
    float omega;
    float current_fl;
    float current_fr;
    float current_rl;
    float accel_x;
    float accel_y;
    float gyro_z;
    uint8_t sensors;
    uint32_t timestamp;
} __attribute__((packed));

struct msg_embeddings_t {
    uint8_t header; // Always 'E'
    float data[17];
} __attribute__((packed));

extern QueueHandle_t commandQueue;

extern TaskHandle_t SupervisorTaskHandle;
extern TaskHandle_t ImuTaskHandle;

// COBS implementation
inline size_t cobs_encode(const uint8_t * ptr, size_t length, uint8_t * dst) {
    size_t read_index = 0;
    size_t write_index = 1;
    size_t code_index = 0;
    uint8_t code = 1;

    while (read_index < length) {
        if (ptr[read_index] == 0) {
            dst[code_index] = code;
            code = 1;
            code_index = write_index++;
            read_index++;
        } else {
            dst[write_index++] = ptr[read_index++];
            code++;
            if (code == 0xFF) {
                dst[code_index] = code;
                code = 1;
                code_index = write_index++;
            }
        }
    }
    dst[code_index] = code;
    return write_index;
}

inline size_t cobs_decode(const uint8_t * ptr, size_t length, uint8_t * dst) {
    size_t read_index = 0;
    size_t write_index = 0;
    uint8_t code = 0;
    uint8_t i = 0;

    while (read_index < length) {
        code = ptr[read_index];
        if (read_index + code > length && code != 1) return 0;
        read_index++;
        for (i = 1; i < code; i++) {
            dst[write_index++] = ptr[read_index++];
        }
        if (code != 0xFF && read_index != length) {
            dst[write_index++] = '\0';
        }
    }
    return write_index;
}

// Function prototypes
void setup_robot();
void MotionSupervisor(void *parameters);
void MotionControl(void *parameters);
