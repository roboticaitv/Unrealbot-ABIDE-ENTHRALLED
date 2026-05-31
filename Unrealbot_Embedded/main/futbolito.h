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

extern QueueHandle_t commandQueue;

extern TaskHandle_t SupervisorTaskHandle;
extern TaskHandle_t ControlTaskHandle;

// Function prototypes
void setup_robot();
void MotionSupervisor(void *parameters);
void MotionControl(void *parameters);
