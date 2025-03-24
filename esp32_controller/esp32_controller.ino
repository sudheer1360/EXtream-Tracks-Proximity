#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Act";
const char* password = "Madhumakeskilled";

// API endpoint with your actual IP address
const char* api_url = "http://192.168.99.238:3456/api/latest_detections";

// Pin definitions
const int BUZZER_PIN = 13;
const int LED_RED_PIN = 12;
const int LED_GREEN_PIN = 14;

// Timing variables
unsigned long lastApiCheck = 0;
const unsigned long API_CHECK_INTERVAL = 500;  // Check every 500ms
const unsigned long BUZZER_TIMEOUT = 2000;     // 2 seconds max buzzer on time
unsigned long buzzerStartTime = 0;
bool buzzerActive = false;

unsigned long lastWarningTime = 0;
const unsigned long WARNING_COOLDOWN = 3000;  // 3 seconds cooldown between warnings
bool isPersonPresent = false;  // Track person detection state

void setup_wifi() {
    WiFi.begin(ssid, password);
    Serial.print("Connecting to WiFi");
    
    int attempts = 0;
    while (WiFi.status() != WL_CONNECTED && attempts < 20) {
        delay(500);
        Serial.print(".");
        attempts++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nConnected to WiFi");
        Serial.println("IP address: " + WiFi.localIP().toString());
    } else {
        Serial.println("\nFailed to connect to WiFi");
    }
}

void setup() {
    Serial.begin(115200);
    
    // Initialize pins
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_GREEN_PIN, OUTPUT);
    
    // Set initial LED states - Start with everything off
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_GREEN_PIN, LOW);
    digitalWrite(BUZZER_PIN, LOW);
    
    // Connect to WiFi
    setup_wifi();
}

void activateWarning() {
    digitalWrite(BUZZER_PIN, HIGH);
    digitalWrite(LED_RED_PIN, HIGH);
    digitalWrite(LED_GREEN_PIN, LOW);
    buzzerActive = true;
    buzzerStartTime = millis();
    Serial.println("Warning activated!");
}

void deactivateWarning() {
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_GREEN_PIN, HIGH);
    buzzerActive = false;
    Serial.println("Warning deactivated!");
}

void checkDetections() {
    if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        
        Serial.println("Making API request...");
        http.begin(api_url);
        int httpCode = http.GET();
        
        if (httpCode == HTTP_CODE_OK) {
            String payload = http.getString();
            
            StaticJsonDocument<1024> doc;
            DeserializationError error = deserializeJson(doc, payload);
            
            if (!error) {
                bool currentPersonDetected = false;
                
                if (doc.containsKey("objects")) {
                    JsonArray objects = doc["objects"];
                    
                    for (JsonVariant obj : objects) {
                        const char* name = obj["name"];
                        float confidence = obj["confidence"];
                        
                        if (strcmp(name, "person") == 0 && confidence > 0.6) {  // Added confidence threshold
                            Serial.printf("Person detected with confidence: %.2f%%\n", confidence * 100);
                            currentPersonDetected = true;
                            break;
                        }
                    }
                }
                
                unsigned long currentTime = millis();
                
                // State change detection
                if (currentPersonDetected != isPersonPresent) {
                    isPersonPresent = currentPersonDetected;
                    
                    if (isPersonPresent) {
                        // Person appeared
                        if (currentTime - lastWarningTime >= WARNING_COOLDOWN) {
                            activateWarning();
                            lastWarningTime = currentTime;
                            Serial.println("Person appeared - Activating warning");
                        }
                    } else {
                        // Person disappeared
                        deactivateWarning();
                        Serial.println("Person disappeared - Deactivating warning");
                    }
                }
            } else {
                Serial.println("JSON parsing failed!");
            }
        } else {
            Serial.printf("HTTP request failed with error: %d\n", httpCode);
        }
        
        http.end();
    } else {
        Serial.println("WiFi disconnected. Attempting to reconnect...");
        setup_wifi();
    }
}

void loop() {
    unsigned long currentMillis = millis();
    
    // Check API periodically
    if (currentMillis - lastApiCheck >= API_CHECK_INTERVAL) {
        lastApiCheck = currentMillis;
        checkDetections();
    }
    
    // Handle buzzer timeout only if warning is active
    if (buzzerActive && (currentMillis - buzzerStartTime >= BUZZER_TIMEOUT)) {
        digitalWrite(BUZZER_PIN, LOW);  // Turn off only the buzzer
        buzzerActive = false;
        Serial.println("Buzzer timeout - Buzzer turned off but keeping LED state");
    }
    
    delay(100);  // Reduced delay to be more responsive
}




