#pragma once

#include <stdint.h>

#define MSBFIRST 1
#define SPI_MODE3 3

class SPISettings {
public:
    SPISettings(uint32_t clock, uint8_t bitOrder, uint8_t dataMode) {}
};

class SPIClass {
public:
    void begin() {}
    void beginTransaction(SPISettings settings) {}
    uint8_t transfer(uint8_t data) { return 0; }
    void transfer(uint8_t *data, uint32_t size) {}
    void endTransaction() {}
};

extern SPIClass SPI;
