#include "motor_control.h"
#include <math.h>
#include <algorithm>

static int map_val(int x, int in_min, int in_max, int out_min, int out_max) {
    return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

static int constrain_val(int x, int a, int b) {
    if (x < a) return a;
    if (x > b) return b;
    return x;
}

void setup_motor_timers() {
    // Timer 0 for DC Motors (20kHz, 8-bit)
    ledc_timer_config_t ledc_timer0 = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = LEDC_TIMER_8_BIT,
        .timer_num        = LEDC_TIMER_0,
        .freq_hz          = 20000,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ledc_timer_config(&ledc_timer0);

    // Timer 1 for Servo (50Hz, 14-bit)
    ledc_timer_config_t ledc_timer1 = {
        .speed_mode       = LEDC_LOW_SPEED_MODE,
        .duty_resolution  = LEDC_TIMER_14_BIT,
        .timer_num        = LEDC_TIMER_1,
        .freq_hz          = 50,
        .clk_cfg          = LEDC_AUTO_CLK
    };
    ledc_timer_config(&ledc_timer1);
}

DRV8251::DRV8251(int in1, int in2, int sensePin, ledc_channel_t pwmCh, adc_oneshot_unit_handle_t adc_handle, adc_channel_t adc_chan) {
    _in1_pin = in1;
    _in2_pin = in2;
    _sense_pin = sensePin;
    _pwm_ch = pwmCh;
    _adc_handle = adc_handle;
    _adc_chan = adc_chan;
}

void DRV8251::begin() {
    gpio_reset_pin((gpio_num_t)_in1_pin);
    gpio_set_direction((gpio_num_t)_in1_pin, GPIO_MODE_OUTPUT);
    gpio_reset_pin((gpio_num_t)_in2_pin);
    gpio_set_direction((gpio_num_t)_in2_pin, GPIO_MODE_OUTPUT);

    stop();
}

void DRV8251::stop() {
    if (_last_dir == 0) return;

    // Disconnect PWM
    ledc_stop(LEDC_LOW_SPEED_MODE, _pwm_ch, 0);

    // Force both LOW (Coast Mode)
    gpio_reset_pin((gpio_num_t)_in1_pin);
    gpio_set_direction((gpio_num_t)_in1_pin, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)_in1_pin, 0);

    gpio_reset_pin((gpio_num_t)_in2_pin);
    gpio_set_direction((gpio_num_t)_in2_pin, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)_in2_pin, 0);

    _last_dir = 0;
}

void DRV8251::updateCurrentSense() {
    if (_sense_pin < 0 || _adc_handle == nullptr) return;

    int adc_raw;
    if (adc_oneshot_read(_adc_handle, _adc_chan, &adc_raw) == ESP_OK) {
        _ipropi_acc += adc_raw;
        _ipropi_samples++;

        if (_ipropi_samples >= 64) {
            float adc_avg = (float)_ipropi_acc / _ipropi_samples;
            float v_ipropi = (adc_avg / 4095.0f) * 3.3f; // Assumes 12-bit ADC and 3.3V ref
            _motor_current_A = (v_ipropi / R_OHMS) * RATIO;

            _ipropi_acc = 0;
            _ipropi_samples = 0;
        }
    }
}

void DRV8251::setSpeed(int pwm) {
    pwm = constrain_val(pwm, -255, 255);

    if (pwm == 0) {
        stop();
        return;
    }

    if (pwm > 0) {
        pwm = map_val(pwm, 1, 255, _deadband, 255);
    } else if (pwm < 0) {
        pwm = -map_val(abs(pwm), 1, 255, _deadband, 255); 
    }

    if (pwm > 0) {
        // Forward
        if (_last_dir != 1) {
            ledc_stop(LEDC_LOW_SPEED_MODE, _pwm_ch, 0);
            
            gpio_reset_pin((gpio_num_t)_in1_pin);
            gpio_set_direction((gpio_num_t)_in1_pin, GPIO_MODE_OUTPUT);
            gpio_set_level((gpio_num_t)_in1_pin, 1);

            ledc_channel_config_t ledc_channel = {
                .gpio_num       = _in2_pin,
                .speed_mode     = LEDC_LOW_SPEED_MODE,
                .channel        = _pwm_ch,
                .intr_type      = LEDC_INTR_DISABLE,
                .timer_sel      = LEDC_TIMER_0,
                .duty           = 0,
                .hpoint         = 0
            };
            ledc_channel_config(&ledc_channel);

            _last_dir = 1;
        }
        ledc_set_duty(LEDC_LOW_SPEED_MODE, _pwm_ch, 256 - pwm); // Inverted PWM
        ledc_update_duty(LEDC_LOW_SPEED_MODE, _pwm_ch);
    } else {
        // Reverse
        if (_last_dir != -1) {
            ledc_stop(LEDC_LOW_SPEED_MODE, _pwm_ch, 0);

            gpio_reset_pin((gpio_num_t)_in2_pin);
            gpio_set_direction((gpio_num_t)_in2_pin, GPIO_MODE_OUTPUT);
            gpio_set_level((gpio_num_t)_in2_pin, 1);

            ledc_channel_config_t ledc_channel = {
                .gpio_num       = _in1_pin,
                .speed_mode     = LEDC_LOW_SPEED_MODE,
                .channel        = _pwm_ch,
                .intr_type      = LEDC_INTR_DISABLE,
                .timer_sel      = LEDC_TIMER_0,
                .duty           = 0,
                .hpoint         = 0
            };
            ledc_channel_config(&ledc_channel);

            _last_dir = -1;
        }
        ledc_set_duty(LEDC_LOW_SPEED_MODE, _pwm_ch, 256 - abs(pwm)); // Inverted PWM
        ledc_update_duty(LEDC_LOW_SPEED_MODE, _pwm_ch);
    }
}

float DRV8251::getCurrent() {
    return _motor_current_A;
}

SimpleServo::SimpleServo(int pin, ledc_channel_t channel) {
    _pin = pin;
    _channel = channel;
}

void SimpleServo::begin() {
    ledc_channel_config_t ledc_channel = {
        .gpio_num       = _pin,
        .speed_mode     = LEDC_LOW_SPEED_MODE,
        .channel        = _channel,
        .intr_type      = LEDC_INTR_DISABLE,
        .timer_sel      = LEDC_TIMER_1,
        .duty           = 0,
        .hpoint         = 0
    };
    ledc_channel_config(&ledc_channel);
    stop();
}

void SimpleServo::write(int speed) {
    speed = constrain_val(speed, -100, 100);

    int pulse_us = 0;
    if (speed == 0) {
        pulse_us = STOP_US;
    } else if (speed > 0) {
        pulse_us = map_val(speed, 0, 100, STOP_US, MAX_FWD_US);
    } else {
        pulse_us = map_val(speed, -100, 0, MAX_REV_US, STOP_US);
    }

    uint32_t duty = (pulse_us * 16383) / 20000;
    ledc_set_duty(LEDC_LOW_SPEED_MODE, _channel, duty);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, _channel);
}

void SimpleServo::stop() {
    write(0);
}
