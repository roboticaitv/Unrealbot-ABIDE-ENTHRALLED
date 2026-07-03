#include "futbolito.h"

DRV8251 motorFL(7, 6, -1, LEDC_CHANNEL_0);   // M1
DRV8251 motorFR(15, 16, -1, LEDC_CHANNEL_1); // M2
DRV8251 motorRL(4, 5, -1, LEDC_CHANNEL_2);   // M3
DRV8251 motorRR(13, 14, -1, LEDC_CHANNEL_3); // M4

EncoderPCNT enc1(0, (gpio_num_t)11, (gpio_num_t)12); // ENC1
EncoderPCNT enc2(1, (gpio_num_t)10, (gpio_num_t)9);  // ENC2
EncoderPCNT enc3(2, (gpio_num_t)18, (gpio_num_t)17); // ENC3

MotorUnit unitFL(&motorFL, &enc1, 0.884f, 38.6f, 0.0f);
MotorUnit unitFR(&motorFR, &enc2, 0.880f, 38.9f, 0.0f);
MotorUnit unitRL(&motorRL, &enc3, 0.865f, 37.1f, 0.0f);

SimpleServo kicker(46, LEDC_CHANNEL_4);

QueueHandle_t commandQueue;

void setup_robot() {
    setup_motor_timers();
    
    // MotorUnit::begin() automatically calls motor->begin() and encoder->begin()!
    // We only need to manually begin() hardware that is NOT part of a MotorUnit.
    motorRR.begin(); // No encoder, so no MotorUnit for RR

    unitFL.begin();
    unitFR.begin();
    unitRL.begin();

    kicker.begin();

    gpio_reset_pin((gpio_num_t)SOLENOID_PIN);
    gpio_set_direction((gpio_num_t)SOLENOID_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)SOLENOID_PIN, 0);

    commandQueue = xQueueCreate(1, sizeof(robot_command_t));
}
