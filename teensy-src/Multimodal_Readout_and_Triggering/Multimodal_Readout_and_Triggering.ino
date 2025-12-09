// ───────────────────────────────────────────────────────────────────────────────
//   Teensy 4.1 – FSR + ECG + GSR ADC streaming + heartbeat LED + command‐triggered pin
// Original code by José Torres, extended by Paul Rüsing. @INI UZH / ETH Zurich
// ───────────────────────────────────────────────────────────────────────────────
const int HEARTBEAT_PIN = LED_BUILTIN;
const int TRIG_PIN      = 28;
const int FSR_AN_PIN    = A10;
const int ECG_AN_PIN    = A2;
const int GSR_AN_PIN    = A4;


// heartbeat config
const unsigned long HEARTBEAT_PERIOD = 250;  // ms toggle interval
unsigned long lastHeartbeat = 0;
bool heartbeatState = false;


void setup() {
  pinMode(HEARTBEAT_PIN, OUTPUT);
  pinMode(TRIG_PIN, OUTPUT);


  Serial.begin(115200);  // ⬅️ Increased baud rate
  delay(1000);           // Let host settle
}


void loop() {
  unsigned long now = millis();


  // ── 1) Non-blocking heartbeat
  if (now - lastHeartbeat >= HEARTBEAT_PERIOD) {
    heartbeatState = !heartbeatState;
    digitalWrite(HEARTBEAT_PIN, heartbeatState);
    lastHeartbeat = now;
  }


  // ── 2) Read FSR sensor and send structured message
  int raw = analogRead(FSR_AN_PIN);
  float voltage = raw * (3.3 / 1023.0);
  Serial.print("FSR:");
  Serial.println(voltage, 6);  // six decimal precision


  // ── 3) Read FSR sensor and send structured message
  raw = analogRead(ECG_AN_PIN);
  voltage = raw * (3.3 / 1023.0);
  Serial.print("ECG:");
  Serial.println(voltage, 6);  // six decimal precision


  // ── 4) Read FSR sensor and send structured message
  raw = analogRead(GSR_AN_PIN);
  voltage = raw * (3.3 / 1023.0);
  Serial.print("GSR:");
  Serial.println(voltage, 6);  // six decimal precision


  // ── 5) Command trigger
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'A') digitalWrite(TRIG_PIN, HIGH);
    else if (cmd == 'B') digitalWrite(TRIG_PIN, LOW);
  }
}