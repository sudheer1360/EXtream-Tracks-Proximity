#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

// WiFi credentials
const char* ssid = "Act";
const char* password = "Madhumakeskilled";

// MQTT Broker settings
const char* mqtt_broker = "broker.hivemq.com";
const int mqtt_port = 1883;
const char* mqtt_topic_buzzer = "road_safety/buzzer";
const char* mqtt_topic_led = "road_safety/led";
const char* mqtt_client_id = "ESP32Client_"; // Will be appended with random number

// Pin definitions
const int BUZZER_PIN = 13;
const int LED_RED_PIN = 12;
const int LED_GREEN_PIN = 14;

// Timing variables for reconnection
unsigned long lastReconnectAttempt = 0;
const long reconnectInterval = 5000; // 5 seconds between reconnection attempts

// MQTT client
WiFiClient espClient;
PubSubClient client(espClient);

// Generate random client ID
String getRandomClientId() {
    String clientId = mqtt_client_id;
    clientId += String(random(0xffff), HEX);
    return clientId;
}

void setup() {
    Serial.begin(115200);
    randomSeed(micros()); // Initialize random seed
    
    // Initialize pins
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(LED_RED_PIN, OUTPUT);
    pinMode(LED_GREEN_PIN, OUTPUT);
    
    // Set initial states
    digitalWrite(BUZZER_PIN, LOW);
    digitalWrite(LED_RED_PIN, LOW);
    digitalWrite(LED_GREEN_PIN, HIGH);
    
    // Connect to WiFi
    setup_wifi();
    
    // Configure MQTT
    client.setServer(mqtt_broker, mqtt_port);
    client.setCallback(callback);
    client.setKeepAlive(60); // Keep alive for 60 seconds
    
    // Initial connection attempt
    connectMQTT();
}

void setup_wifi() {
    delay(10);
    Serial.println("Connecting to WiFi...");
    
    WiFi.mode(WIFI_STA); // Set WiFi to station mode
    WiFi.begin(ssid, password);
    
    int attempt = 0;
    while (WiFi.status() != WL_CONNECTED && attempt < 20) {
        delay(500);
        Serial.print(".");
        attempt++;
    }
    
    if (WiFi.status() == WL_CONNECTED) {
        Serial.println("\nWiFi connected");
        Serial.println("IP address: ");
        Serial.println(WiFi.localIP());
    } else {
        Serial.println("\nWiFi connection failed!");
        ESP.restart(); // Restart ESP32 if WiFi connection fails
    }
}

void callback(char* topic, byte* payload, unsigned int length) {
    Serial.println("\n--- MQTT Message Received ---");
    Serial.print("Topic: ");
    Serial.println(topic);
    Serial.print("Payload: ");
    for (int i = 0; i < length; i++) {
        Serial.print((char)payload[i]);
    }
    Serial.println();
    
    StaticJsonDocument<200> doc;
    char message[length + 1];
    memcpy(message, payload, length);
    message[length] = '\0';
    
    DeserializationError error = deserializeJson(doc, message);
    
    if (error) {
        Serial.print("Failed to parse JSON: ");
        Serial.println(error.c_str());
        return;
    }
    
    // Handle buzzer control
    if (strcmp(topic, mqtt_topic_buzzer) == 0) {
        bool state = doc["state"];
        Serial.print("Setting buzzer state to: ");
        Serial.println(state ? "ON" : "OFF");
        
        digitalWrite(BUZZER_PIN, state ? HIGH : LOW);
        if (state) {
            tone(BUZZER_PIN, 2000); // 2000Hz frequency
        } else {
            noTone(BUZZER_PIN);
        }
        
        Serial.println("Buzzer state changed");
    }
}

boolean connectMQTT() {
    if (!client.connected()) {
        Serial.println("Attempting MQTT connection...");
        
        String clientId = getRandomClientId();
        
        if (client.connect(clientId.c_str())) {
            Serial.println("Connected to MQTT broker");
            
            // Subscribe to topics
            client.subscribe(mqtt_topic_buzzer);
            client.subscribe(mqtt_topic_led);
            
            // Publish connection message
            String connectMsg = "ESP32 " + clientId + " connected";
            client.publish("road_safety/status", connectMsg.c_str());
            
            return true;
        } else {
            Serial.print("Failed, rc=");
            Serial.print(client.state());
            Serial.println(" Retrying later...");
            return false;
        }
    }
    return true;
}

void checkWiFiConnection() {
    if (WiFi.status() != WL_CONNECTED) {
        Serial.println("WiFi connection lost! Reconnecting...");
        WiFi.disconnect();
        WiFi.begin(ssid, password);
        
        int attempt = 0;
        while (WiFi.status() != WL_CONNECTED && attempt < 20) {
            delay(500);
            Serial.print(".");
            attempt++;
        }
        
        if (WiFi.status() != WL_CONNECTED) {
            Serial.println("\nWiFi reconnection failed! Restarting...");
            ESP.restart();
        }
    }
}

void loop() {
    static unsigned long lastHeartbeat = 0;
    const unsigned long HEARTBEAT_INTERVAL = 5000; // 5 seconds
    
    // Check WiFi connection
    checkWiFiConnection();
    
    // Handle MQTT connection
    if (!client.connected()) {
        unsigned long now = millis();
        if (now - lastReconnectAttempt > reconnectInterval) {
            lastReconnectAttempt = now;
            if (connectMQTT()) {
                lastReconnectAttempt = 0;
            }
        }
    } else {
        client.loop();
        
        // Send heartbeat message every 5 seconds
        unsigned long now = millis();
        if (now - lastHeartbeat > HEARTBEAT_INTERVAL) {
            lastHeartbeat = now;
            client.publish("road_safety/heartbeat", "ESP32 alive");
            Serial.println("Heartbeat sent");
        }
    }
}

