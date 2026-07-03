#ifndef FUTBOLITO_H
#define FUTBOLITO_H
#include <ESP32Servo.h>  // Requires "ESP32Servo" library by Kevin Harrington
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\Motor_Control.h"
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\MT6701.h"
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\PMW3901.h"
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\PID.h"
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\Motor_Unit.h"
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\OmniKinematics.h"
// --- Instantiation ---
// DRV8251(IN1, IN2, SensePin, PWM_Ch)

DRV8251 motorFL(13, 14, -1, 0);  // Channel 0
DRV8251 motorFR(3, 8, -1, 1);  // Channel 1
DRV8251 motorRL(11, 12, -1, 2);   // Channel 2
DRV8251 motorRR(9, 10, -1, 3);    // Channel 3

EncoderPCNT enc1(PCNT_UNIT_0, (gpio_num_t)1, (gpio_num_t)2);
EncoderPCNT enc2(PCNT_UNIT_1, (gpio_num_t)37, (gpio_num_t)38);
EncoderPCNT enc3(PCNT_UNIT_2, (gpio_num_t)35, (gpio_num_t)36);

MotorUnit unitFL(&motorFL, &enc1, 0.884f, 38.6f, 0.0f);  // pins 11,12
MotorUnit unitFR(&motorFR, &enc2, 0.880f, 38.9f, 0.0f);  // pins 13,14
MotorUnit unitRL(&motorRL, &enc3, 0.865f, 37.1f, 0.0f);  // pins 3,8

#define SPI_MOSI_PIN 6  // Master Out Slave In
#define SPI_MISO_PIN 4  // Master In Slave Out
#define SPI_SCK_PIN 7   // Serial Clock
#define IMU_CS_PIN 5    // IMU Chip Select
#define FLOW_CS_PIN 10  // PMW3901 Chip Select

const int pinA = -1;
const int pinB = -1;
const int pinC = 17;
const int pinD = 18;

volatile bool flagA = false;
volatile bool flagB = false;
volatile bool flagC = false;
volatile bool flagD = false;

volatile uint32_t lastMillisA = 0;
volatile uint32_t lastMillisB = 0;
volatile uint32_t lastMillisC = 0;
volatile uint32_t lastMillisD = 0;
uint32_t lastDebounceA = 0;
uint32_t lastDebounceB = 0;
uint32_t lastDebounceC = 0;
uint32_t lastDebounceD = 0;

const uint32_t debounceDelay = 50;  // ms

void IRAM_ATTR isrA();
void IRAM_ATTR isrB();
void IRAM_ATTR isrC();
void IRAM_ATTR isrD();

#include "ICM45686.h"

// ---------------------------------------------------------
// CUSTOM SPI PINS FOR ESP32-S3
// ---------------------------------------------------------
#define SPI_MOSI_PIN 6  // Master Out Slave In
#define SPI_MISO_PIN 4  // Master In Slave Out
#define SPI_SCK_PIN 7   // Serial Clock
#define IMU_CS_PIN 5    // IMU Chip Select
#define FLOW_CS_PIN 10  // PMW3901 Chip Select
// ---------------------------------------------------------
// PERFORMANCE SETTINGS
// ---------------------------------------------------------
#define SAMPLE_RATE_HZ 800  // 800 Hz
#define SPI_CLOCK_HZ 10000000
#define ACCEL_RANGE_G 16     // ±16G range
#define GYRO_RANGE_DPS 2000  // ±2000 dps range

// Create IMU object for SPI mode
ICM456xx IMU(SPI, IMU_CS_PIN);

// Performance monitoring
unsigned long sample_count = 0;
unsigned long last_stats_print = 0;
unsigned long max_loop_time = 0;

Custom_PMW3901 flow(FLOW_CS_PIN, &SPI);  // Pass the SPI bus by reference

// --- Global Variables for Sensors ---
inv_imu_sensor_data_t imu_data;
int16_t deltaX, deltaY;

// Timer globals
unsigned long lastImuTime = 0;
unsigned long lastPrintTime = 0;

// --- TIMING VARIABLES ---
const uint64_t SUPERVISOR_US = 500;  // 700us = 1.4kHz
const uint64_t CONTROL_US = 2000;    // 2000 us = 500Hz

// Task Handles
TaskHandle_t SupervisorTaskHandle = NULL;
TaskHandle_t ControlTaskHandle = NULL;

// Hardware Timers
hw_timer_t* MotionSupervisorTimer = NULL;
hw_timer_t* MotionControlTimer = NULL;

portMUX_TYPE accumMux = portMUX_INITIALIZER_UNLOCKED;

// Add to globals
#define TICKS_PER_REV 37768.0f                // Adjust to your encoder PPR * 4 for quadrature
#define SUPERVISOR_DT (SUPERVISOR_US / 1e6f)  // Period in seconds = 0.0001s
#define WHEEL_RADIUS 0.06f  

uint64_t last_snapshot_us = 0; // <-- Fixes the setup() error
struct encoder_sample_t {
  int32_t fl_ticks;
  int32_t fr_ticks;
  int32_t rl_ticks;
};
// The unified command structure
struct robot_command_t {
  float vx;     // Target X velocity (m/s)
  float vy;     // Target Y velocity (m/s)
  float omega;  // Target rotational velocity (rad/s)
  uint8_t kick_strength;
  uint32_t timestamp;
};

QueueHandle_t sampleQueue;
QueueHandle_t commandQueue;
// Single global flag, no mutex needed (atomic on ESP32 for uint8)
volatile bool control_did_read = false;

#include <esp_now.h>
#include <WiFi.h>

uint8_t broadcastAddress[] = { 0x80, 0xb5, 0x4e, 0xc6, 0x8c, 0x28 };
esp_now_peer_info_t peerInfo;
// Structure for the message - must match on both devices
typedef struct struct_message {
  char data[250];  // Generic char buffer for serial data
} struct_message;

struct_message outgoingMsg;
struct_message incomingMsg;

enum RobotMode { MODE_MANUAL,
                 MODE_POSITION };
volatile RobotMode currentMode = MODE_MANUAL;

// Message Type 1: RC Commands
// Update this on the Station ESP32!
typedef struct __attribute__((packed)) msg_manual_t {
  char type;  
  float vx;
  float vy;
  float omega;
  uint8_t kick_strength; 
} msg_manual_t;

// Message Type 2: Ally Status
typedef struct __attribute__((packed)) msg_ally_t {
  char type;  // Will always be 'A'
} msg_ally_t;

int SOLENOID_PIN = 18; 

#include <esp_wifi.h>
#include "C:\Users\caser\OneDrive\Documents\Arduino\futbolito\Unrealbot_Embedded\ESPNOW.h"
#include <math.h>
#include "esp_task_wdt.h"
#endif