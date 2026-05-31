#pragma once

#include "driver/ledc.h"
#include "driver/gpio.h"
#include "esp_adc/adc_oneshot.h"

void setup_motor_timers();

class DRV8251 {
private:
    int _in1_pin;
    int _in2_pin;
    int _sense_pin;
    ledc_channel_t _pwm_ch;
    int _last_dir = 0;  // 0=Stop, 1=Fwd, -1=Rev

    int _deadband = 25; 

    // Current Sensing Variables
    uint32_t _ipropi_acc = 0;
    uint16_t _ipropi_samples = 0;
    float _motor_current_A = 0.0f;

    static constexpr float R_OHMS = 1500.0f;
    static constexpr float RATIO = 1000.0f;

    adc_oneshot_unit_handle_t _adc_handle;
    adc_channel_t _adc_chan;

public:
    DRV8251(int in1, int in2, int sensePin, ledc_channel_t pwmCh, adc_oneshot_unit_handle_t adc_handle = nullptr, adc_channel_t adc_chan = ADC_CHANNEL_0);
    
    void begin();
    void stop();
    void updateCurrentSense();
    void setSpeed(int pwm);
    float getCurrent();
};

class SimpleServo {
private:
    int _pin;
    ledc_channel_t _channel;

    const int STOP_US = 1500;
    const int MAX_FWD_US = 2000;
    const int MAX_REV_US = 1000;

public:
    SimpleServo(int pin, ledc_channel_t channel);
    void begin();
    void write(int speed);
    void stop();
};
