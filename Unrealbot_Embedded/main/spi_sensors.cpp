#include "spi_sensors.h"
#include "futbolito.h"
#include "driver/gpio.h"
#include <string.h>
#include <stdlib.h>

spi_device_handle_t imu_spi_handle;
spi_device_handle_t flow_spi_handle;

static spi_transaction_t imu_trans;
static spi_transaction_t flow_trans;

static uint8_t imu_rx_buf[14];
static uint8_t flow_rx_buf[5];

void setup_spi() {
    spi_bus_config_t buscfg = {};
    buscfg.miso_io_num = SPI_MISO_PIN;
    buscfg.mosi_io_num = SPI_MOSI_PIN;
    buscfg.sclk_io_num = SPI_SCK_PIN;
    buscfg.quadwp_io_num = -1;
    buscfg.quadhd_io_num = -1;
    buscfg.max_transfer_sz = 64;

    // Enable DMA
    spi_bus_initialize(SPI2_HOST, &buscfg, SPI_DMA_CH_AUTO);

    spi_device_interface_config_t imu_devcfg = {};
    imu_devcfg.clock_speed_hz = SPI_CLOCK_HZ;
    imu_devcfg.mode = 3; 
    imu_devcfg.spics_io_num = IMU_CS_PIN;
    imu_devcfg.queue_size = 2;
    
    spi_bus_add_device(SPI2_HOST, &imu_devcfg, &imu_spi_handle);

    spi_device_interface_config_t flow_devcfg = {};
    flow_devcfg.clock_speed_hz = 4000000; // 4 MHz
    flow_devcfg.mode = 3;
    flow_devcfg.spics_io_num = -1; // Manual CS
    flow_devcfg.queue_size = 2;
    
    spi_bus_add_device(SPI2_HOST, &flow_devcfg, &flow_spi_handle);

    gpio_reset_pin((gpio_num_t)FLOW_CS_PIN);
    gpio_set_direction((gpio_num_t)FLOW_CS_PIN, GPIO_MODE_OUTPUT);
    gpio_set_level((gpio_num_t)FLOW_CS_PIN, 1);
}

void imu_spi_write(uint8_t reg, const uint8_t *wbuffer, uint32_t wlen) {
    spi_transaction_t t = {};
    t.length = 8 + wlen * 8;
    
    uint8_t* tx_data = (uint8_t*)malloc(1 + wlen);
    tx_data[0] = reg;
    memcpy(&tx_data[1], wbuffer, wlen);

    t.tx_buffer = tx_data;
    spi_device_transmit(imu_spi_handle, &t);
    free(tx_data);
}

void imu_spi_read(uint8_t reg, uint8_t *rbuffer, uint32_t rlen) {
    spi_transaction_t t = {};
    t.length = 8 + rlen * 8;
    t.rxlength = 8 + rlen * 8;
    
    uint8_t* tx_data = (uint8_t*)calloc(1 + rlen, 1);
    uint8_t* rx_data = (uint8_t*)calloc(1 + rlen, 1);
    
    tx_data[0] = reg | 0x80; // SPI_READ

    t.tx_buffer = tx_data;
    t.rx_buffer = rx_data;
    
    spi_device_transmit(imu_spi_handle, &t);
    
    memcpy(rbuffer, &rx_data[1], rlen);
    free(tx_data);
    free(rx_data);
}

#include "esp_rom_sys.h"

void flow_spi_write(uint8_t reg, uint8_t value) {
    spi_transaction_t t = {};
    t.length = 8;
    
    uint8_t tx_reg = (uint8_t)(reg | 0x80);
    
    gpio_set_level((gpio_num_t)FLOW_CS_PIN, 0);
    esp_rom_delay_us(50);
    
    t.tx_buffer = &tx_reg;
    spi_device_transmit(flow_spi_handle, &t);
    
    t.tx_buffer = &value;
    spi_device_transmit(flow_spi_handle, &t);
    
    esp_rom_delay_us(50);
    gpio_set_level((gpio_num_t)FLOW_CS_PIN, 1);
    esp_rom_delay_us(200);
}

uint8_t flow_spi_read(uint8_t reg) {
    spi_transaction_t t = {};
    t.length = 8;
    
    uint8_t tx_reg = (uint8_t)(reg & ~0x80);
    uint8_t rx_val = 0;
    
    gpio_set_level((gpio_num_t)FLOW_CS_PIN, 0);
    esp_rom_delay_us(50);
    
    t.tx_buffer = &tx_reg;
    spi_device_transmit(flow_spi_handle, &t);
    
    esp_rom_delay_us(50);
    
    t.tx_buffer = NULL;
    t.rx_buffer = &rx_val;
    spi_device_transmit(flow_spi_handle, &t);
    
    esp_rom_delay_us(50); // Optional extra delay before CS high
    gpio_set_level((gpio_num_t)FLOW_CS_PIN, 1);
    
    return rx_val;
}

void trigger_dma_spi_reads() {
    // IMU: Queue reading of ACCEL and GYRO data (14 bytes) starting at register 0x1F (ACCEL_DATA_X1)
    memset(&imu_trans, 0, sizeof(spi_transaction_t));
    imu_trans.length = 14 * 8;
    imu_trans.tx_buffer = NULL;
    imu_trans.rx_buffer = imu_rx_buf;

    spi_device_queue_trans(imu_spi_handle, &imu_trans, portMAX_DELAY);
    
    /* 
    // TODO: DMA for PMW3901 Flow Sensor (currently requires manual CS and delays)
    // FLOW: PMW3901 burst read is from register 0x16 or individual 0x02..0x06
    // Since PMW3901 burst read gives Motion, Obs, dxL, dxH, dyL, dyH, squal...
    // Let's just read 5 bytes starting at 0x02
    memset(&flow_trans, 0, sizeof(spi_transaction_t));
    flow_trans.length = 8 + 5 * 8;
    flow_trans.rxlength = 8 + 5 * 8;

    static uint8_t flow_tx[6] = { 0x02 & ~0x80 }; 
    flow_trans.tx_buffer = flow_tx;
    flow_trans.rx_buffer = flow_rx_buf;

    spi_device_queue_trans(flow_spi_handle, &flow_trans, portMAX_DELAY);
    */
}

bool fetch_dma_spi_reads(uint8_t* imu_data) {
    spi_transaction_t *ret_trans;

    // Wait for IMU
    spi_device_get_trans_result(imu_spi_handle, &ret_trans, portMAX_DELAY);
    memcpy(imu_data, &imu_rx_buf[1], 14);

    /*
    // Wait for Flow
    spi_device_get_trans_result(flow_spi_handle, &ret_trans, portMAX_DELAY);
    memcpy(flow_data, &flow_rx_buf[1], 5);
    */

    return true;
}
