#pragma once
#include <Arduino.h>
#include <SPI.h>

class Custom_PMW3901 {
  private:
    uint8_t _cs;
    SPIClass *_spi;

    // Low level register access
    void registerWrite(uint8_t reg, uint8_t value) {
      reg |= 0x80u;
      _spi->beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE3));
      digitalWrite(_cs, LOW);
      delayMicroseconds(50);
      _spi->transfer(reg);
      _spi->transfer(value);
      delayMicroseconds(50);
      digitalWrite(_cs, HIGH);
      _spi->endTransaction();
      delayMicroseconds(200);
    }

    uint8_t registerRead(uint8_t reg) {
      reg &= ~0x80u;
      _spi->beginTransaction(SPISettings(4000000, MSBFIRST, SPI_MODE3));
      digitalWrite(_cs, LOW);
      delayMicroseconds(50);
      _spi->transfer(reg);
      delayMicroseconds(50);
      uint8_t value = _spi->transfer(0);
      delayMicroseconds(100);
      digitalWrite(_cs, HIGH);
      _spi->endTransaction();
      return value;
    }

    // Performance optimisation registers
    void initRegisters() {
      registerWrite(0x7F, 0x00); registerWrite(0x61, 0xAD); registerWrite(0x7F, 0x03);
      registerWrite(0x40, 0x00); registerWrite(0x7F, 0x05); registerWrite(0x41, 0xB3);
      registerWrite(0x43, 0xF1); registerWrite(0x45, 0x14); registerWrite(0x5B, 0x32);
      registerWrite(0x5F, 0x34); registerWrite(0x7B, 0x08); registerWrite(0x7F, 0x06);
      registerWrite(0x44, 0x1B); registerWrite(0x40, 0xBF); registerWrite(0x4E, 0x3F);
      registerWrite(0x7F, 0x08); registerWrite(0x65, 0x20); registerWrite(0x6A, 0x18);
      registerWrite(0x7F, 0x09); registerWrite(0x4F, 0xAF); registerWrite(0x5F, 0x40);
      registerWrite(0x48, 0x80); registerWrite(0x49, 0x80); registerWrite(0x57, 0x77);
      registerWrite(0x60, 0x78); registerWrite(0x61, 0x78); registerWrite(0x62, 0x08);
      registerWrite(0x63, 0x50); registerWrite(0x7F, 0x0A); registerWrite(0x45, 0x60);
      registerWrite(0x7F, 0x00); registerWrite(0x4D, 0x11); registerWrite(0x55, 0x80);
      registerWrite(0x74, 0x1F); registerWrite(0x75, 0x1F); registerWrite(0x4A, 0x78);
      registerWrite(0x4B, 0x78); registerWrite(0x44, 0x08); registerWrite(0x45, 0x50);
      registerWrite(0x64, 0xFF); registerWrite(0x65, 0x1F); registerWrite(0x7F, 0x14);
      registerWrite(0x65, 0x60); registerWrite(0x66, 0x08); registerWrite(0x63, 0x78);
      registerWrite(0x7F, 0x15); registerWrite(0x48, 0x58); registerWrite(0x7F, 0x07);
      registerWrite(0x41, 0x0D); registerWrite(0x43, 0x14); registerWrite(0x4B, 0x0E);
      registerWrite(0x45, 0x0F); registerWrite(0x44, 0x42); registerWrite(0x4C, 0x80);
      registerWrite(0x7F, 0x10); registerWrite(0x5B, 0x02); registerWrite(0x7F, 0x07);
      registerWrite(0x40, 0x41); registerWrite(0x70, 0x00);
      delay(100);
      registerWrite(0x32, 0x44); registerWrite(0x7F, 0x07); registerWrite(0x40, 0x40);
      registerWrite(0x7F, 0x06); registerWrite(0x62, 0xf0); registerWrite(0x63, 0x00);
      registerWrite(0x7F, 0x0D); registerWrite(0x48, 0xC0); registerWrite(0x6F, 0xd5);
      registerWrite(0x7F, 0x00); registerWrite(0x5B, 0xa0); registerWrite(0x4E, 0xA8);
      registerWrite(0x5A, 0x50); registerWrite(0x40, 0x80);
    }

  public:
    // Constructor now takes the SPI object pointer
    Custom_PMW3901(uint8_t cspin, SPIClass *spiPort = &SPI) : _cs(cspin), _spi(spiPort) { }

    boolean begin(void) {
      pinMode(_cs, OUTPUT);
      
      // Make sure the SPI bus is reset
      digitalWrite(_cs, HIGH);
      delay(1);
      digitalWrite(_cs, LOW);
      delay(1);
      digitalWrite(_cs, HIGH);
      delay(1);

      // Power on reset
      registerWrite(0x3A, 0x5A);
      delay(5);
      
      // Test the SPI communication
      uint8_t chipId = registerRead(0x00);
      uint8_t dIpihc = registerRead(0x5F);

      if (chipId != 0x49 && dIpihc != 0xB8) return false;

      // Reading the motion registers one time
      registerRead(0x02); registerRead(0x03); registerRead(0x04);
      registerRead(0x05); registerRead(0x06);
      delay(1);

      initRegisters();
      return true;
    }

    void readMotionCount(int16_t *deltaX, int16_t *deltaY) {
      registerRead(0x02);
      *deltaX = ((int16_t)registerRead(0x04) << 8) | registerRead(0x03);
      *deltaY = ((int16_t)registerRead(0x06) << 8) | registerRead(0x05);
    }
};