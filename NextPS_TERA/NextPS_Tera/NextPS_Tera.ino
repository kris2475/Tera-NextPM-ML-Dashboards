
#include <Arduino.h>

HardwareSerial SensorSerial(2);

const int SENSOR_RX = 16;  // ESP32 RX2 <- sensor TX
const int SENSOR_TX = 17;  // ESP32 TX2 -> sensor RX

const uint8_t request[] = {0x81, 0x31, 0x4E};
const size_t FRAME_SIZE = 24;

uint32_t readU32(const uint8_t *p) {
  return ((uint32_t)p[0] << 24) |
         ((uint32_t)p[1] << 16) |
         ((uint32_t)p[2] << 8)  |
          (uint32_t)p[3];
}

void setup() {
  Serial.begin(115200);
  SensorSerial.begin(115200, SERIAL_8E1,
                    SENSOR_RX, SENSOR_TX);

  Serial.println("\nNextPS particle counter test");
  delay(10000);  // Sensor warm-up
}

void loop() {
  while (SensorSerial.available()) {
    SensorSerial.read();
  }

  SensorSerial.write(request, sizeof(request));
  SensorSerial.flush();

  uint8_t frame[FRAME_SIZE];
  size_t n = 0;
  unsigned long start = millis();

  while (n < FRAME_SIZE && millis() - start < 1500) {
    if (SensorSerial.available()) {
      frame[n++] = SensorSerial.read();
    }
    delay(1);
  }

  if (n != FRAME_SIZE) {
    Serial.println("Incomplete response or timeout");
    delay(2000);
    return;
  }

  uint8_t sum = 0;
  for (size_t i = 0; i < FRAME_SIZE; i++) {
    sum += frame[i];
  }

  if (frame[0] != 0x81 || frame[1] != 0x31) {
    Serial.println("Unexpected response header");
    delay(2000);
    return;
  }

  if (sum != 0) {
    Serial.println("Checksum error");
    delay(2000);
    return;
  }

  Serial.printf("Sensor state: 0x%02X\n", frame[2]);

  const char *labels[] = {
    "> 0.3 um", "> 0.5 um", "> 1.0 um",
    "> 2.5 um", "> 5.0 um"
  };

  for (int i = 0; i < 5; i++) {
    uint32_t count = readU32(&frame[3 + i * 4]);
    Serial.printf("%s: %lu particles/m3\n",
                  labels[i], (unsigned long)count);
  }

  Serial.println("------------------------");
  delay(2000);
}


