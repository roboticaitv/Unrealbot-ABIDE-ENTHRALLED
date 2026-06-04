#include "espnow_comms.h"
#include "futbolito.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include "driver/uart.h"
#include <string.h>

static const char* TAG = "ESPNOW";

struct msg_manual_t {
    char packet_type;
    float vx;
    float vy;
    float omega;
    float kick_strength;
} __attribute__((packed));

struct msg_pose_t {
    char packet_type;
    float x;
    float y;
    float theta;
} __attribute__((packed));

void OnDataSent(const esp_now_send_info_t *tx_info, esp_now_send_status_t status) {
    ESP_LOGI(TAG, "Send status: %s", status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

void OnDataRecv(const esp_now_recv_info_t *esp_now_info, const uint8_t *incomingData, int len) {
    if (len == 0) return;

    char packet_type = (char)incomingData[0];

    if (packet_type == 'M' && len == sizeof(msg_manual_t)) {
        msg_manual_t rc_data;
        memcpy(&rc_data, incomingData, sizeof(msg_manual_t));

        robot_command_t cmd;
        cmd.mode = 0; // MODE_VELOCITY
        cmd.arg1 = rc_data.vx;
        cmd.arg2 = rc_data.vy;
        cmd.arg3 = rc_data.omega;
        cmd.kick_strength = rc_data.kick_strength;
        cmd.timestamp = esp_timer_get_time();

        xQueueOverwrite(commandQueue, &cmd);
    } 
    else if (packet_type == 'P' && len == sizeof(msg_pose_t)) {
        msg_pose_t p_data;
        memcpy(&p_data, incomingData, sizeof(msg_pose_t));

        robot_command_t cmd;
        cmd.mode = 1; // MODE_ABSOLUTE_POSE
        cmd.arg1 = p_data.x;
        cmd.arg2 = p_data.y;
        cmd.arg3 = p_data.theta;
        cmd.kick_strength = 0;
        cmd.timestamp = esp_timer_get_time();

        xQueueOverwrite(commandQueue, &cmd);
    } 
    else if (packet_type == 'E' && len == sizeof(msg_embeddings_t)) {
        // Forward Allied Embeddings directly to Pi over USB, encoded with COBS
        uint8_t encoded[128];
        size_t enc_len = cobs_encode(incomingData, len, encoded);
        encoded[enc_len++] = 0x00;
        uart_write_bytes(UART_NUM_0, encoded, enc_len);
    }
}

void espnow_broadcast_embeddings(msg_embeddings_t* packet) {
    uint8_t broadcast_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_send(broadcast_mac, (uint8_t *)packet, sizeof(msg_embeddings_t));
}

void setup_espnow() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_register_send_cb(OnDataSent));
    ESP_ERROR_CHECK(esp_now_register_recv_cb(OnDataRecv));

    // Register Broadcast Peer
    uint8_t broadcast_mac[6] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};
    esp_now_peer_info_t peerInfo = {};
    memcpy(peerInfo.peer_addr, broadcast_mac, 6);
    peerInfo.channel = 0;  
    peerInfo.encrypt = false;
    ESP_ERROR_CHECK(esp_now_add_peer(&peerInfo));

    ESP_LOGI(TAG, "ESP-NOW Initialized (Broadcast Peer Added).");
}
