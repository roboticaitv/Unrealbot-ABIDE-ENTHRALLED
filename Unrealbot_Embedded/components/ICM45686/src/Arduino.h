#pragma once

#include <stdint.h>
#include <string.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_rom_sys.h"

#define HIGH 1
#define LOW 0
#define OUTPUT GPIO_MODE_OUTPUT
#define INPUT GPIO_MODE_INPUT
#define RISING 1

typedef void (*ICM456xx_irq_handler)(void);
typedef bool boolean;

static inline void pinMode(int pin, int mode) {
    gpio_reset_pin((gpio_num_t)pin);
    gpio_set_direction((gpio_num_t)pin, (gpio_mode_t)mode);
}

static inline void digitalWrite(int pin, int val) {
    gpio_set_level((gpio_num_t)pin, val);
}

static inline void delayMicroseconds(uint32_t us) {
    esp_rom_delay_us(us);
}

static inline void delay(uint32_t ms) {
    vTaskDelay(pdMS_TO_TICKS(ms));
}

static inline uint32_t millis() {
    return esp_timer_get_time() / 1000;
}

static inline uint32_t micros() {
    return esp_timer_get_time();
}

static inline void attachInterrupt(int pin, void (*handler)(void), int mode) {
    // Dummy for now, we poll in ESP-IDF
}

#ifdef __cplusplus
class Stream {
public:
    virtual size_t write(uint8_t) { return 1; }
    virtual size_t write(const uint8_t *buf, size_t size) { return size; }
    virtual int available() { return 0; }
    virtual int read() { return -1; }
    virtual int peek() { return -1; }
    virtual void flush() {}
    void print(const char* s) {}
    void println(const char* s) {}
    void print(int i, int format = 10) {}
    void println(int i, int format = 10) {}
};

class HardwareSerial : public Stream {
public:
    void begin(unsigned long baud) {}
};

extern HardwareSerial Serial;

class TwoWire {
public:
    void begin() {}
    void setClock(uint32_t clock) {}
    void beginTransmission(uint8_t address) {}
    void write(uint8_t data) {}
    void endTransmission(bool stop = true) {}
    uint16_t requestFrom(uint8_t address, uint16_t length) { return 0; }
    uint8_t read() { return 0; }
};

#endif // __cplusplus
