#ifndef MOTOR_UNIT_H
#define MOTOR_UNIT_H

struct MotorUnit {
  DRV8251* motor;
  EncoderPCNT* encoder;
  PID          pid;

  // Encoder state
  int32_t accumulated_ticks = 0;
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
    int16_t delta = (int16_t)encoder->getCount();
    encoder->clear();
    accumulated_ticks += delta;
  }

  void setSpeed(int pwm) {
    motor->setSpeed(pwm);
  }
};

#endif