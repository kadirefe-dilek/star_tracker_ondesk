// DORT DURUMLU QUADRATURE ENCODER OKUMA (INTERRUPTSIZ + DEBOUNCE + mm + KALIBRASYON)

// --- Ayarlar ve Pin Tanımları ---
#define A_PIN 7
#define B_PIN 14

// Her count'un temsil ettiği fiziksel mesafe (mm)
// Senin kullandığın temel scale: 0.00001 * 100 = 0.001 mm / count (1 µm)
const float MM_PER_COUNT = 0.00001f;

// 20 mm götürdüğünde yaklaşık 19.7656 mm ölçtüğünü söyledin.
// Bunu düzeltmek için scale correction katsayısı:
const float SCALE_CORRECTION = 20.0f / 19.7656f;  // yaklaşık 1.0118

// Debounce filtresi için bekleme süresi (us)
const unsigned long DEBOUNCE_US = 90; 

// --- State Machine Globals ---
long position = 0;
unsigned long illegalTransitions = 0;

// İleri/Geri durum geçiş tablosu (lookup table)
const int8_t transitionTable[4][4] = {
  // new:   00   01   10   11
  /00/ {  0,  -1,  +1,   0 },
  /01/ { +1,   0,   0,  -1 },
  /10/ { -1,   0,   0,  +1 },
  /11/ {  0,  +1,  -1,   0 }
};

// Son onaylanmış (filtrelenmiş) (A,B) durumu
uint8_t lastState;

// --- Debounce Filtresi ---

struct DebouncedChannel {
  byte pin;
  bool stableState;         // son onaylı (debounced) durum
  bool lastRaw;             // son okunan ham değer
  unsigned long lastChange; // ham değişimin zamanı
};

DebouncedChannel chA = { A_PIN, false, false, 0 };
DebouncedChannel chB = { B_PIN, false, false, 0 };

void updateDebounced(DebouncedChannel &ch) {
  bool raw = digitalRead(ch.pin);
  unsigned long now = micros();

  if (raw != ch.lastRaw) {
    ch.lastRaw = raw;
    ch.lastChange = now;
  }

  if (raw != ch.stableState) {
    if ((now - ch.lastChange) > DEBOUNCE_US) {
      ch.stableState = raw;
    }
  }
}

// --- Ana Arduino Kodları ---

void setup() {
  pinMode(A_PIN, INPUT_PULLUP);
  pinMode(B_PIN, INPUT_PULLUP);

  Serial.begin(115200);
  Serial.println("4-state quadrature encoder (DEBOUNCED + mm + CALIBRATED) basladi.");

  bool initialA = digitalRead(A_PIN);
  bool initialB = digitalRead(B_PIN);

  chA.stableState = initialA;
  chA.lastRaw     = initialA;
  chA.lastChange  = micros();

  chB.stableState = initialB;
  chB.lastRaw     = initialB;
  chB.lastChange  = micros();

  lastState = (initialA << 1) | initialB;
}

void loop() {
  static unsigned long lastPrint = 0;
  const unsigned long printPeriod = 100; // ms

  // 1) DEBOUNCE
  updateDebounced(chA);
  updateDebounced(chB);

  // 2) STATE MACHINE
  uint8_t a = chA.stableState;
  uint8_t b = chB.stableState;
  uint8_t currState = (a << 1) | b;

  if (currState != lastState) {
    int8_t step = transitionTable[lastState][currState];

    if (step == 0) {
      illegalTransitions++;
    } else {
      // Yön düzeltmesi: ileri hareket pozitif olsun diye tersliyoruz
      position -= step;
    }

    lastState = currState;
  }

  // 3) YAZDIRMAYI YAVASLAT + mm'ye çevir + kalibrasyon uygula
  unsigned long now = millis();
  if (now - lastPrint >= printPeriod) {
    lastPrint = now;

    // Temel scale: position * MM_PER_COUNT * 100
    // Düzeltme: SCALE_CORRECTION ile çarp
    float distance_mm = position * MM_PER_COUNT * 100.0f * SCALE_CORRECTION;

    // Tek satırda güncelle
    Serial.print("\rpos = ");
    Serial.print(position);
    Serial.println("  | mesafe = ");
    Serial.print(distance_mm, 4);
    Serial.print(" mm  | illegal = ");
    Serial.print(illegalTransitions);
    Serial.print("        "); // eski karakterleri temizlemek için boşluk
  }
}