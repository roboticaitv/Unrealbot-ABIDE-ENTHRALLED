#include <Arduino.h>
#include <WiFi.h>
#include <esp_now.h>

// Put your Robot's MAC Address here!
uint8_t robotMacAddress[] = {0x80, 0xb5, 0x4e, 0xc6, 0x8c, 0x28};
//Seeduino UI  e0:72:a1:d7:5f:98
//Car e0:72:a1:d7:5f:98
// EXACT same struct as the robot
typedef struct __attribute__((packed)) msg_manual_t {
  char type;  
  float vx;
  float vy;
  float omega;
  float kick_strength;  
} msg_manual_t;

// We know the struct is exactly 17 bytes long
const int PACKET_SIZE = sizeof(msg_manual_t);
uint8_t serialBuffer[PACKET_SIZE];

void setup() {
  Serial.begin(1000000);

  // Initialize Wi-Fi and ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    while (1) delay(10);
  }

  // Register the peer (Your Robot)
  esp_now_peer_info_t peerInfo = {};
  memcpy(peerInfo.peer_addr, robotMacAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  
  if (esp_now_add_peer(&peerInfo) != ESP_OK) {
    Serial.println("Failed to add peer");
    while(1) delay(10);
  }
}

void loop() {
  // If we have enough bytes in the Serial buffer to form a complete struct
  if (Serial.available() >= PACKET_SIZE) {
    
    // Check the very first byte to ensure alignment
    if (Serial.peek() == 'M') {
      
      // Scoop up the exact bytes needed
      Serial.readBytes(serialBuffer, PACKET_SIZE);
      
      // Fire it directly over Wi-Fi
      esp_now_send(robotMacAddress, serialBuffer, PACKET_SIZE);
      
    } else {
      // If the first byte isn't 'M', we lost alignment. +9
      // Throw away one byte and try again next loop to sync up.
      Serial.read(); 
    }
  }
}
