idf_component_register(SRCS "src/ICM45686.cpp"
                            "src/imu/inv_imu_transport.c"
                            "src/imu/inv_imu_driver.c"
                            "src/imu/inv_imu_driver_advanced.c"
                            "src/imu/inv_imu_driver_aux1.c"
                            "src/imu/inv_imu_edmp.c"
                            "src/imu/inv_imu_edmp_compass.c"
                            "src/imu/inv_imu_edmp_wearable.c"
                            "src/imu/inv_imu_i2cm.c"
                            "src/imu/inv_imu_selftest.c"
                            "src/inv_time.c"
                    INCLUDE_DIRS "src" "src/imu" "src/Invn"
                    PRIV_REQUIRES spi_flash
                    REQUIRES driver esp_timer)
