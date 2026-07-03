#pragma once

#include "motor_control.h"
#include "encoder_pcnt.h"
#include "pid.h"

struct MotorUnit {
  DRV8251* motor;
  EncoderPCNT* encoder;
  PID          pid;

  // Encoder state
  int32_t accumulated_ticks = 0;
  int32_t last_hardware_count = 0;
  bool control_did_read     = false;

  // Velocity filter
  float filtered_velocity   = 0.0f;
  float alpha               = 0.3f;

  // Latest computed velocity
  float angular_velocity_rads = 0.0f;

  MotorUnit(DRV8251* m, EncoderPCNT* e, float kp, float ki, float kd)
    : motor(m), encoder(e), pid(kp, ki, kd, 255.0f, 255.0f) {}

  void begin() {
    motor->begin();
    encoder->begin();
  }

  // Call from supervisor task (CLEANED UP!)
  void supervisorTick(bool did_read) {
    int32_t current_count = encoder->getCount();
    int16_t delta = (int16_t)(current_count - last_hardware_count);
    last_hardware_count = current_count;
    accumulated_ticks += delta;
  }

  void setSpeed(int pwm) {
    motor->setSpeed(pwm);
  }
};
