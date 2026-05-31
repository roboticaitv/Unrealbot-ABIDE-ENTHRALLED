#ifndef ESPNOW_H
#define ESPNOW_H

// Callback when data is sent via ESP-NOW
void OnDataSent(const uint8_t *mac_addr, esp_now_send_status_t status) {
  Serial.print("[ESP-NOW] Send status: ");
  Serial.println(status == ESP_NOW_SEND_SUCCESS ? "Success" : "Fail");
}

void OnDataRecv(const uint8_t *mac, const uint8_t *incomingData, int len) {
  if (len == 0) return;
  char packet_type = (char)incomingData[0];

  // ROUTE: MANUAL MODE ('M')
  if (packet_type == 'M' && len == sizeof(msg_manual_t)) {
    msg_manual_t rc_data;
    memcpy(&rc_data, incomingData, sizeof(msg_manual_t));

    // Package it for MotionControl
    robot_command_t cmd;
    cmd.vx = rc_data.vx;
    cmd.vy = rc_data.vy;
    cmd.omega = rc_data.omega;
    cmd.kick_strength = rc_data.kick_strength;  // <--- ADD THIS LINE!
    cmd.timestamp = micros();

    xQueueOverwrite(commandQueue, &cmd);
  }
}

#endif