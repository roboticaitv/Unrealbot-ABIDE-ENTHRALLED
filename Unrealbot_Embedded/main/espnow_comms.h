#pragma once
#include "futbolito.h"

void setup_espnow();
void espnow_send_telemetry(telemetry_packet_t* packet);
