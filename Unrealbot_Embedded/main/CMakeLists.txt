idf_component_register(SRCS "main.cpp" "motor_control.cpp" "futbolito.cpp" "spi_sensors.cpp" "espnow_comms.cpp"
                    INCLUDE_DIRS "."
                    PRIV_REQUIRES ICM45686 driver esp_timer esp_adc esp_wifi nvs_flash esp_netif)
