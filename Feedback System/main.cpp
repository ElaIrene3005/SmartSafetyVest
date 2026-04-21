#include <Arduino.h>

// UART CONFIGURATION
// Connect ESP32 RX to Raspberry Pi TX (Pin 8 / GPIO 14)
// Connect ESP32 TX to Raspberry Pi RX (Pin 10 / GPIO 15)
#define RXD1 18   
#define TXD1 17   

// PIN DEFINITIONS: LED ARRAY (L298N #1)
#define LED_L_PWM 1  
#define LED_L_GND 2  
#define LED_R_PWM 10 
#define LED_R_GND 11 

// PIN DEFINITIONS: TAPTIC MOTORS (L298N #2) 
#define MOTOR_L_PWM 42 
#define MOTOR_L_GND 41 
#define MOTOR_R_PWM 40 
#define MOTOR_R_GND 39 

// PWM SETTINGS (LEDC) 
#define CH_LED_L    0
#define CH_LED_R    1
#define CH_MOTOR_L  2
#define CH_MOTOR_R  3

#define PWM_FREQ 500
#define PWM_RES 8
#define MAX_DUTY 254 // Using 254 based on testing success

// TIMING SETTINGS
#define BLINK_INTERVAL 250 // Pulse speed (ms)
#define ACTION_PAUSE   200 // Cooling period between commands (ms)

void stopAll() {
    // 1. Reset all PWM values to zero
    ledcWrite(CH_LED_L, 0);
    ledcWrite(CH_LED_R, 0);
    ledcWrite(CH_MOTOR_L, 0);
    ledcWrite(CH_MOTOR_R, 0);

    // 2. Set GND pins to HIGH for absolute zero potential difference
    digitalWrite(LED_L_GND, HIGH);
    digitalWrite(LED_R_GND, HIGH);
    digitalWrite(MOTOR_L_GND, HIGH);
    digitalWrite(MOTOR_R_GND, HIGH);
    
    Serial.println(">>> System Standby: All Outputs Disabled <<<");
}

void setup() {
    Serial.begin(115200);
    Serial1.begin(9600, SERIAL_8N1, RXD1, TXD1);
    delay(2000);

    // Initialize all pins as OUTPUT
    pinMode(LED_L_PWM, OUTPUT); pinMode(LED_L_GND, OUTPUT);
    pinMode(LED_R_PWM, OUTPUT); pinMode(LED_R_GND, OUTPUT);
    pinMode(MOTOR_L_PWM, OUTPUT); pinMode(MOTOR_L_GND, OUTPUT);
    pinMode(MOTOR_R_PWM, OUTPUT); pinMode(MOTOR_R_GND, OUTPUT);

    // Setup PWM Hardware
    ledcSetup(CH_LED_L, PWM_FREQ, PWM_RES);
    ledcSetup(CH_LED_R, PWM_FREQ, PWM_RES);
    ledcSetup(CH_MOTOR_L, PWM_FREQ, PWM_RES);
    ledcSetup(CH_MOTOR_R, PWM_FREQ, PWM_RES);

    // Attach PWM channels to specific pins
    ledcAttachPin(LED_L_PWM, CH_LED_L);
    ledcAttachPin(LED_R_PWM, CH_LED_R);
    ledcAttachPin(MOTOR_L_PWM, CH_MOTOR_L);
    ledcAttachPin(MOTOR_R_PWM, CH_MOTOR_R);

    // Start in a clean, stopped state
    stopAll(); 
    Serial.println("ESP32 Controller Online - Awaiting UART Commands...");
}

// VIBRATION FUNCTIONS
void activateLeft() {
    Serial.println("Action: LEFT DETECTED");
    digitalWrite(MOTOR_L_GND, LOW);
    digitalWrite(LED_L_GND, LOW);

    ledcWrite(CH_MOTOR_L, MAX_DUTY); // Motor ON
    
    for(int i = 0; i < 3; i++) {
        ledcWrite(CH_LED_L, MAX_DUTY); 
        delay(BLINK_INTERVAL);
        ledcWrite(CH_LED_L, 0);        
        delay(BLINK_INTERVAL);
    }
    
    stopAll(); // Reset state after action finishes
}

void activateRight() {
    Serial.println("Action: RIGHT DETECTED");
    digitalWrite(MOTOR_R_GND, LOW);
    digitalWrite(LED_R_GND, LOW);

    ledcWrite(CH_MOTOR_R, MAX_DUTY); 

    for(int i = 0; i < 3; i++) {
        ledcWrite(CH_LED_R, MAX_DUTY); 
        delay(BLINK_INTERVAL);
        ledcWrite(CH_LED_R, 0);        
        delay(BLINK_INTERVAL);
    }

    stopAll();
}

void activateCenter() {
    Serial.println("Action: CENTER DETECTED (High Priority)");
    
    // Enable all grounds
    digitalWrite(MOTOR_L_GND, LOW); 
    digitalWrite(MOTOR_R_GND, LOW);
    digitalWrite(LED_L_GND, LOW);   
    digitalWrite(LED_R_GND, LOW);

    // Both Motors ON immediately
    ledcWrite(CH_MOTOR_L, MAX_DUTY);
    ledcWrite(CH_MOTOR_R, MAX_DUTY);

    // Both LEDs blink together
    for(int i = 0; i < 3; i++) {
        ledcWrite(CH_LED_L, MAX_DUTY); 
        ledcWrite(CH_LED_R, MAX_DUTY);
        delay(BLINK_INTERVAL);
        
        ledcWrite(CH_LED_L, 0);        
        ledcWrite(CH_LED_R, 0);
        delay(BLINK_INTERVAL);
    }

    stopAll();
}

void loop() {
  if (Serial1.available()) {
    String command = Serial1.readStringUntil('\n');
    command.trim(); 
    
    Serial.print("Command Received: "); 
    Serial.println(command);

    if (command == "LEFT") {
      activateLeft();
    }
    else if (command == "RIGHT") {
      activateRight();
    }
    else if (command == "CENTER") {
      activateCenter();
    }
    else if (command == "STATIC" || command == "STOP") {
      stopAll();
    }
    
    delay(ACTION_PAUSE);
  }
}