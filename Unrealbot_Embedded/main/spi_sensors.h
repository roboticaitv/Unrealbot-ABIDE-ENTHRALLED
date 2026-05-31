#pragma once

#include <stdint.h>
#include "driver/spi_master.h"

extern spi_device_handle_t imu_spi_handle;
extern spi_device_handle_t flow_spi_handle;

void setup_spi();

// Synchronous calls for initialization
void imu_spi_write(uint8_t reg, const uint8_t *wbuffer, uint32_t wlen);
void imu_spi_read(uint8_t reg, uint8_t *rbuffer, uint32_t rlen);

void flow_spi_write(uint8_t reg, uint8_t value);
uint8_t flow_spi_read(uint8_t reg);

// Asynchronous DMA call for the high speed loop
void trigger_dma_spi_reads();
bool fetch_dma_spi_reads(uint8_t* imu_data);
