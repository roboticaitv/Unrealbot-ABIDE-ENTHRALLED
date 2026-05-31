#include "espnow_comms.h"
#include "futbolito.h"
#include "esp_now.h"
#include "esp_wifi.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_mac.h"
#include "esp_timer.h"
#include <string.h>

static const char* TAG = "ESPNOW";

static uint8_t station_mac[6] = {0};
static bool station_mac_set = false;

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

    // Automatically register the Station as a peer so we can send telemetry back!
    if (!station_mac_set) {
        memcpy(station_mac, esp_now_info->src_addr, 6);
        esp_now_peer_info_t peerInfo = {};
        memcpy(peerInfo.peer_addr, station_mac, 6);
        peerInfo.channel = 0;  
        peerInfo.encrypt = false;
        if (!esp_now_is_peer_exist(station_mac)) {
            esp_now_add_peer(&peerInfo);
        }
        station_mac_set = true;
        ESP_LOGI(TAG, "Registered Station MAC for Telemetry Uplink.");
    }

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
}

void espnow_send_telemetry(telemetry_packet_t* packet) {
    if (station_mac_set) {
        esp_now_send(station_mac, (uint8_t *)packet, sizeof(telemetry_packet_t));
    }
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

    ESP_LOGI(TAG, "ESP-NOW Initialized.");
}
