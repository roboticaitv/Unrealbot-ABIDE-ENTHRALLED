#pragma once

#include "driver/pulse_cnt.h"
#include "driver/gpio.h"
#include "esp_log.h"

class EncoderPCNT {
private:
    pcnt_unit_handle_t pcnt_unit = NULL;
    pcnt_channel_handle_t pcnt_chan_a = NULL;
    pcnt_channel_handle_t pcnt_chan_b = NULL;
    gpio_num_t pinA;
    gpio_num_t pinB;

public:
    // We keep the first argument (unit_num) to prevent breaking legacy code,
    // but the new API handles unit allocation dynamically.
    EncoderPCNT(int unit_num_ignored, gpio_num_t a, gpio_num_t b)
        : pinA(a), pinB(b) {}

    void begin() {
        pcnt_unit_config_t unit_config = {};
        unit_config.high_limit = 32767;
        unit_config.low_limit = -32768;
        
        ESP_ERROR_CHECK(pcnt_new_unit(&unit_config, &pcnt_unit));

        // Setup 1us glitch filter for noisy magnetic environments
        pcnt_glitch_filter_config_t filter_config = {};
        filter_config.max_glitch_ns = 1000;
        ESP_ERROR_CHECK(pcnt_unit_set_glitch_filter(pcnt_unit, &filter_config));

        // Create Channel A
        pcnt_chan_config_t chan_a_config = {};
        chan_a_config.edge_gpio_num = pinA;
        chan_a_config.level_gpio_num = pinB;
        ESP_ERROR_CHECK(pcnt_new_channel(pcnt_unit, &chan_a_config, &pcnt_chan_a));

        // Create Channel B
        pcnt_chan_config_t chan_b_config = {};
        chan_b_config.edge_gpio_num = pinB;
        chan_b_config.level_gpio_num = pinA;
        ESP_ERROR_CHECK(pcnt_new_channel(pcnt_unit, &chan_b_config, &pcnt_chan_b));

        // Set edge and level actions for X4 Quadrature Decoding
        
        // Channel A (Pulse = A, Ctrl = B)
        ESP_ERROR_CHECK(pcnt_channel_set_edge_action(pcnt_chan_a, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
        ESP_ERROR_CHECK(pcnt_channel_set_level_action(pcnt_chan_a, PCNT_CHANNEL_LEVEL_ACTION_KEEP, PCNT_CHANNEL_LEVEL_ACTION_INVERSE));

        // Channel B (Pulse = B, Ctrl = A)
        ESP_ERROR_CHECK(pcnt_channel_set_edge_action(pcnt_chan_b, PCNT_CHANNEL_EDGE_ACTION_INCREASE, PCNT_CHANNEL_EDGE_ACTION_DECREASE));
        ESP_ERROR_CHECK(pcnt_channel_set_level_action(pcnt_chan_b, PCNT_CHANNEL_LEVEL_ACTION_INVERSE, PCNT_CHANNEL_LEVEL_ACTION_KEEP));

        // Enable and start
        ESP_ERROR_CHECK(pcnt_unit_enable(pcnt_unit));
        ESP_ERROR_CHECK(pcnt_unit_clear_count(pcnt_unit));
        ESP_ERROR_CHECK(pcnt_unit_start(pcnt_unit));
    }

    int32_t getCount() {
        int count = 0;
        pcnt_unit_get_count(pcnt_unit, &count);
        return (int32_t)count;
    }

    void clear() {
        pcnt_unit_clear_count(pcnt_unit);
    }
};
