#ifndef MOTOR_CONTROL_H
#define MOTOR_CONTROL_H
// Configuration
#define PWM_FREQ 20000
#define PWM_RES 8
#define IPROPI_R_OHMS 1500.0f
#define IPROPI_RATIO 1000.0f

class DRV8251 {
private:
  int _in1_pin;
  int _in2_pin;
  int _sense_pin;
  uint8_t _pwm_ch;
  int _last_dir = 0;  // 0=Stop, 1=Fwd, -1=Rev

  // ---> THE DEADBAND VARIABLE <---
  // Adjust this number (e.g., 15-35) until the motors perfectly
  // overcome their static friction when given a PWM command of 1.
  int _deadband = 25; 

  // Current Sensing Variables
  uint32_t _ipropi_acc = 0;
  uint16_t _ipropi_samples = 0;
  float _motor_current_A = 0.0f;

  // constants defined inside class to avoid conflicts
  static constexpr float R_OHMS = 1500.0f;
  static constexpr float RATIO = 1000.0f;

public:
  DRV8251(int in1, int in2, int sensePin, uint8_t pwmCh) {
    _in1_pin = in1;
    _in2_pin = in2;
    _sense_pin = sensePin;
    _pwm_ch = pwmCh;
  }

  void begin() {
    pinMode(_in1_pin, OUTPUT);
    pinMode(_in2_pin, OUTPUT);

    // Only set up sense pin if it's a valid pin (not -1)
    if (_sense_pin >= 0) {
      pinMode(_sense_pin, INPUT);
    }

    ledcSetup(_pwm_ch, 20000, 8);  // 20kHz, 8-bit
    stop();                        
  }

  void stop() {
    // THE FIX: If we are already stopped, do nothing and return immediately!
    if (_last_dir == 0) return;

    // Detach PWM from both pins to ensure they are standard GPIOs
    ledcDetachPin(_in1_pin);
    ledcDetachPin(_in2_pin);

    // Force both LOW (Coast Mode)
    digitalWrite(_in1_pin, LOW);
    digitalWrite(_in2_pin, LOW);

    _last_dir = 0;
  }

  void updateCurrentSense() {
    if (_sense_pin < 0) return;  // Skip if disabled

    _ipropi_acc += analogRead(_sense_pin);
    _ipropi_samples++;

    if (_ipropi_samples >= 64) {
      float adc_avg = (float)_ipropi_acc / _ipropi_samples;
      float v_ipropi = (adc_avg / 4095.0f) * 3.3f;
      _motor_current_A = (v_ipropi / R_OHMS) * RATIO;

      _ipropi_acc = 0;
      _ipropi_samples = 0;
    }
  }

  void setSpeed(int pwm) {
    pwm = constrain(pwm, -255, 255);

    if (pwm == 0) {
      stop();
      return;
    }

    // ---------------------------------------------------------
    // DEADBAND COMPENSATION (Static Friction Boost)
    // ---------------------------------------------------------
    if (pwm > 0) {
      pwm = map(pwm, 1, 255, _deadband, 255);
    } else if (pwm < 0) {
      // Use abs() to keep the math symmetrical, then re-apply the negative
      pwm = -map(abs(pwm), 1, 255, _deadband, 255); 
    }
    // ---------------------------------------------------------

    if (pwm > 0) {
      // Forward
      if (_last_dir != 1) {
        ledcDetachPin(_in1_pin);  // Ensure IN1 is GPIO
        pinMode(_in1_pin, OUTPUT);
        digitalWrite(_in1_pin, HIGH);  // Hold IN1 HIGH

        ledcAttachPin(_in2_pin, _pwm_ch);  // PWM on IN2
        _last_dir = 1;
      }
      ledcWrite(_pwm_ch, 255 - pwm);  // Inverted PWM
    } else {
      // Reverse
      if (_last_dir != -1) {
        ledcDetachPin(_in2_pin);  // Ensure IN2 is GPIO
        pinMode(_in2_pin, OUTPUT);
        digitalWrite(_in2_pin, HIGH);  // Hold IN2 HIGH

        ledcAttachPin(_in1_pin, _pwm_ch);  // PWM on IN1
        _last_dir = -1;
      }
      ledcWrite(_pwm_ch, 255 - abs(pwm));  // Inverted PWM
    }
  }

  float getCurrent() {
    return _motor_current_A;
  }
};

// --- Global Settings ---
// Ensure we use the correct resolution/attenuation globally
void setupADC() {
  analogReadResolution(12);        // 0–4095
  analogSetAttenuation(ADC_11db);  // 0–3.3 V
}

class SimpleServo {
private:
  uint8_t _pin;
  uint8_t _channel;

  // Servo Settings (Standard 360 Servo)
  // 50Hz = 20ms period
  // 14-bit resolution = 16383 counts
  const int STOP_US = 1500;
  const int MAX_FWD_US = 2000;
  const int MAX_REV_US = 1000;

public:
  // Constructor
  SimpleServo(uint8_t pin, uint8_t channel) {
    _pin = pin;
    _channel = channel;
  }

  void begin() {
    pinMode(_pin, OUTPUT);

    // Setup Timer: 50 Hz, 14-bit Resolution
    // Note: We use 14-bit because 8-bit is too coarse for servos
    ledcSetup(_channel, 50, 14);
    ledcAttachPin(_pin, _channel);

    stop();
  }

  // Input: -100 (Max Reverse) to 100 (Max Forward)
  void write(int speed) {
    speed = constrain(speed, -100, 100);

    int pulse_us = 0;

    if (speed == 0) {
      pulse_us = STOP_US;
    } else if (speed > 0) {
      pulse_us = map(speed, 0, 100, STOP_US, MAX_FWD_US);
    } else {
      pulse_us = map(speed, -100, 0, MAX_REV_US, STOP_US);
    }

    // Convert Microseconds to Duty Cycle (14-bit)
    // Duty = (pulse_us / 20000us) * 16383
    uint32_t duty = (pulse_us * 16383) / 20000;

    ledcWrite(_channel, duty);
  }

  void stop() {
    write(0);
  }
};

#endif