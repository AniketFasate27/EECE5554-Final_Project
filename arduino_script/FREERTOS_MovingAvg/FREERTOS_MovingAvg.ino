// ============================================================================
// Smart Motor Health Diagnostics - Level 1 with FreeRTOS
// ============================================================================
// Dual-Core Implementation: 
//   Core 0: Data Acquisition (High Priority)
//   Core 1: Data Processing + Serial Output (Lower Priority)
//
// Team: Sofia Makowska, Jeje Dennis, Aniket Fasate, Madison O'Neil
// Date: November 2025
// ============================================================================

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

// ============================================================================
// CONFIGURATION PARAMETERS
// ============================================================================

// Sampling configuration
const int SAMPLE_RATE_MS = 100;        // 100 Hz sampling rate (adjust as needed)
const int MOVING_AVG_WINDOW = 5;      // Moving average window size (3-10 recommended)

// Task priorities (higher number = higher priority)
const int PRIORITY_ACQUISITION = 3;   // Highest - data acquisition must not be delayed
const int PRIORITY_PROCESSING = 2;    // Medium - process data as it arrives
const int PRIORITY_OUTPUT = 1;        // Lowest - output can wait if needed

// Task stack sizes (in words, not bytes)
const int STACK_SIZE_ACQUISITION = 4096;
const int STACK_SIZE_PROCESSING = 4096;
const int STACK_SIZE_OUTPUT = 2048;

// Queue sizes (number of items each queue can hold)
const int QUEUE_SIZE_RAW = 10;        // Buffer for raw sensor data
const int QUEUE_SIZE_PROCESSED = 10;  // Buffer for processed data

// ============================================================================
// DATA STRUCTURES
// ============================================================================

// Structure to hold raw sensor readings
struct SensorData {
  unsigned long timestamp;    // Time in milliseconds
  float accel_x;             // Acceleration X (m/s²)
  float accel_y;             // Acceleration Y (m/s²)
  float accel_z;             // Acceleration Z (m/s²)
  float gyro_x;              // Angular velocity X (rad/s)
  float gyro_y;              // Angular velocity Y (rad/s)
  float gyro_z;              // Angular velocity Z (rad/s)
  float temperature;         // Temperature (°C)
};

// Structure to hold processed data (raw + smoothed)
struct ProcessedData {
  unsigned long timestamp;
  // Raw values (preserved for fault detection)
  float accel_x_raw;
  float accel_y_raw;
  float accel_z_raw;
  float gyro_x_raw;
  float gyro_y_raw;
  float gyro_z_raw;
  // Smoothed values (for trend analysis)
  float accel_x_smooth;
  float accel_y_smooth;
  float accel_z_smooth;
  float gyro_x_smooth;
  float gyro_y_smooth;
  float gyro_z_smooth;
  float temperature;
};

// ============================================================================
// GLOBAL VARIABLES
// ============================================================================

// Hardware
Adafruit_MPU6050 mpu;

// FreeRTOS Queues for inter-task communication
QueueHandle_t rawDataQueue;           // Acquisition → Processing
QueueHandle_t processedDataQueue;     // Processing → Output

// FreeRTOS Task Handles
TaskHandle_t acquisitionTaskHandle = NULL;
TaskHandle_t processingTaskHandle = NULL;
TaskHandle_t outputTaskHandle = NULL;

// Mutex for I2C bus protection (MPU6050 communication)
SemaphoreHandle_t i2cMutex;

// Moving average circular buffers
float accelX_buffer[MOVING_AVG_WINDOW] = {0};
float accelY_buffer[MOVING_AVG_WINDOW] = {0};
float accelZ_buffer[MOVING_AVG_WINDOW] = {0};
float gyroX_buffer[MOVING_AVG_WINDOW] = {0};
float gyroY_buffer[MOVING_AVG_WINDOW] = {0};
float gyroZ_buffer[MOVING_AVG_WINDOW] = {0};
int bufferIndex = 0;

// Statistics counters
unsigned long totalSamples = 0;
unsigned long droppedSamples = 0;

// ============================================================================
// MOVING AVERAGE ALGORITHM
// ============================================================================

/**
 * Computes moving average by maintaining a circular buffer
 * @param newValue: Latest sensor reading
 * @param buffer: Circular buffer array
 * @param windowSize: Number of samples to average
 * @return: Smoothed value
 */
float movingAverage(float newValue, float* buffer, int windowSize) {
  // Store new value at current position
  buffer[bufferIndex] = newValue;
  
  // Calculate sum of all values in window
  float sum = 0;
  for (int i = 0; i < windowSize; i++) {
    sum += buffer[i];
  }
  
  // Return average
  return sum / windowSize;
}

// ============================================================================
// TASK 1: DATA ACQUISITION (Runs on Core 0)
// ============================================================================
/**
 * High-priority task running on Core 0
 * Reads MPU6050 sensor at precise intervals
 * Sends raw data to processing task via queue
 */
void acquisitionTask(void *parameter) {
  TickType_t xLastWakeTime;
  const TickType_t xFrequency = pdMS_TO_TICKS(SAMPLE_RATE_MS);
  
  // Initialize timing for periodic execution
  xLastWakeTime = xTaskGetTickCount();
  
  Serial.println("[ACQ] Data Acquisition Task started on Core 0");
  Serial.print("[ACQ] Target sampling rate: ");
  Serial.print(1000 / SAMPLE_RATE_MS);
  Serial.println(" Hz");
  
  while (1) {
    // Wait for next cycle (ensures precise timing)
    vTaskDelayUntil(&xLastWakeTime, xFrequency);
    
    SensorData data;
    
    // Take mutex before accessing I2C bus
    if (xSemaphoreTake(i2cMutex, portMAX_DELAY) == pdTRUE) {
      
      // Capture timestamp
      data.timestamp = millis();
      
      // Read sensor data from MPU6050
      sensors_event_t a, g, temp;
      mpu.getEvent(&a, &g, &temp);
      
      // Store readings in structure
      data.accel_x = a.acceleration.x;
      data.accel_y = a.acceleration.y;
      data.accel_z = a.acceleration.z;
      data.gyro_x = g.gyro.x;
      data.gyro_y = g.gyro.y;
      data.gyro_z = g.gyro.z;
      data.temperature = temp.temperature;
      
      // Release mutex
      xSemaphoreGive(i2cMutex);
      
      // Send data to processing task
      if (xQueueSend(rawDataQueue, &data, 0) == pdTRUE) {
        totalSamples++;
      } else {
        // Queue full - data point lost
        droppedSamples++;
        if (droppedSamples % 10 == 0) {
          Serial.print("[ACQ] WARNING: ");
          Serial.print(droppedSamples);
          Serial.println(" samples dropped due to full queue!");
        }
      }
    } else {
      Serial.println("[ACQ] ERROR: Failed to acquire I2C mutex");
    }
  }
}

// ============================================================================
// TASK 2: DATA PROCESSING (Runs on Core 1)
// ============================================================================
/**
 * Medium-priority task running on Core 1
 * Applies moving average filter to smooth sensor data
 * Preserves both raw and smoothed data for analysis
 */
void processingTask(void *parameter) {
  Serial.println("[PROC] Data Processing Task started on Core 1");
  Serial.print("[PROC] Moving average window: ");
  Serial.print(MOVING_AVG_WINDOW);
  Serial.println(" samples");
  
  while (1) {
    SensorData rawData;
    
    // Wait for data from acquisition task (blocking)
    if (xQueueReceive(rawDataQueue, &rawData, portMAX_DELAY) == pdTRUE) {
      
      ProcessedData procData;
      
      // Copy timestamp
      procData.timestamp = rawData.timestamp;
      
      // Store raw values (important for fault detection!)
      procData.accel_x_raw = rawData.accel_x;
      procData.accel_y_raw = rawData.accel_y;
      procData.accel_z_raw = rawData.accel_z;
      procData.gyro_x_raw = rawData.gyro_x;
      procData.gyro_y_raw = rawData.gyro_y;
      procData.gyro_z_raw = rawData.gyro_z;
      procData.temperature = rawData.temperature;
      
      // Apply moving average filter to acceleration data
      procData.accel_x_smooth = movingAverage(rawData.accel_x, accelX_buffer, MOVING_AVG_WINDOW);
      procData.accel_y_smooth = movingAverage(rawData.accel_y, accelY_buffer, MOVING_AVG_WINDOW);
      procData.accel_z_smooth = movingAverage(rawData.accel_z, accelZ_buffer, MOVING_AVG_WINDOW);
      
      // Apply moving average filter to gyroscope data
      procData.gyro_x_smooth = movingAverage(rawData.gyro_x, gyroX_buffer, MOVING_AVG_WINDOW);
      procData.gyro_y_smooth = movingAverage(rawData.gyro_y, gyroY_buffer, MOVING_AVG_WINDOW);
      procData.gyro_z_smooth = movingAverage(rawData.gyro_z, gyroZ_buffer, MOVING_AVG_WINDOW);
      
      // Update buffer index (circular buffer)
      bufferIndex = (bufferIndex + 1) % MOVING_AVG_WINDOW;
      
      // Send processed data to output task
      if (xQueueSend(processedDataQueue, &procData, 0) != pdTRUE) {
        Serial.println("[PROC] WARNING: Output queue full!");
      }
    }
  }
}

// ============================================================================
// TASK 3: DATA OUTPUT (Runs on Core 1)
// ============================================================================
/**
 * Low-priority task running on Core 1
 * Handles serial communication (slow operation)
 * Outputs data in CSV format for Telemetry Viewer
 */
void outputTask(void *parameter) {
  Serial.println("[OUT] Data Output Task started on Core 1");
  Serial.println("[OUT] CSV format: Time,Raw(6),Smooth(6),Temp");
  Serial.println();
  
  // Wait for system to stabilize
  vTaskDelay(pdMS_TO_TICKS(1000));
  
  // Print CSV header for Telemetry Viewer
  Serial.println("Time,Ax_Raw,Ay_Raw,Az_Raw,Gx_Raw,Gy_Raw,Gz_Raw,Ax_Smooth,Ay_Smooth,Az_Smooth,Gx_Smooth,Gy_Smooth,Gz_Smooth,Temp");
  
  while (1) {
    ProcessedData data;
    
    // Wait for processed data (blocking)
    if (xQueueReceive(processedDataQueue, &data, portMAX_DELAY) == pdTRUE) {
      
      // Output timestamp
      Serial.print(data.timestamp);
      Serial.print(",");
      
      // ===== RAW ACCELERATION DATA (m/s²) =====
      Serial.print(data.accel_x_raw, 4);
      Serial.print(",");
      Serial.print(data.accel_y_raw, 4);
      Serial.print(",");
      Serial.print(data.accel_z_raw, 4);
      Serial.print(",");
      
      // ===== RAW GYROSCOPE DATA (rad/s) =====
      Serial.print(data.gyro_x_raw, 4);
      Serial.print(",");
      Serial.print(data.gyro_y_raw, 4);
      Serial.print(",");
      Serial.print(data.gyro_z_raw, 4);
      Serial.print(",");
      
      // ===== SMOOTHED ACCELERATION DATA (m/s²) =====
      Serial.print(data.accel_x_smooth, 4);
      Serial.print(",");
      Serial.print(data.accel_y_smooth, 4);
      Serial.print(",");
      Serial.print(data.accel_z_smooth, 4);
      Serial.print(",");
      
      // ===== SMOOTHED GYROSCOPE DATA (rad/s) =====
      Serial.print(data.gyro_x_smooth, 4);
      Serial.print(",");
      Serial.print(data.gyro_y_smooth, 4);
      Serial.print(",");
      Serial.print(data.gyro_z_smooth, 4);
      Serial.print(",");
      
      // ===== TEMPERATURE (°C) =====
      Serial.println(data.temperature, 2);
    }
  }
}

// ============================================================================
// SETUP FUNCTION
// ============================================================================

void setup() {
  // Initialize serial communication
  Serial.begin(115200);
  delay(2000);  // Wait for serial to stabilize
  
  // Print startup banner
  Serial.println();
  Serial.println("========================================");
  Serial.println("  Smart Motor Health Diagnostics");
  Serial.println("  Level 1: FreeRTOS Implementation");
  Serial.println("========================================");
  Serial.println();
  Serial.println("Team: Sofia Makowska, Jeje Dennis,");
  Serial.println("      Aniket Fasate, Madison O'Neil");
  Serial.println();
  Serial.println("========================================");
  Serial.println();
  
  // ========================================
  // Initialize I2C Mutex
  // ========================================
  Serial.println("Initializing FreeRTOS resources...");
  i2cMutex = xSemaphoreCreateMutex();
  if (i2cMutex == NULL) {
    Serial.println("ERROR: Failed to create I2C mutex!");
    Serial.println("System halted.");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("✓ I2C Mutex created");
  
  // ========================================
  // Initialize MPU6050 Sensor
  // ========================================
  Serial.println();
  Serial.println("Initializing MPU6050 sensor...");
  
  if (!mpu.begin()) {
    Serial.println();
    Serial.println("========================================");
    Serial.println("ERROR: Failed to find MPU6050 chip");
    Serial.println("========================================");
    Serial.println();
    Serial.println("Please check wiring:");
    Serial.println("  VCC  -> 3.3V (NOT 5V!)");
    Serial.println("  GND  -> GND");
    Serial.println("  SDA  -> GPIO 21 (ESP32)");
    Serial.println("  SCL  -> GPIO 22 (ESP32)");
    Serial.println();
    Serial.println("System halted.");
    Serial.println("========================================");
    while (1) {
      delay(1000);
    }
  }
  Serial.println("✓ MPU6050 Found!");
  
  // ========================================
  // Configure Accelerometer
  // ========================================
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("✓ Accelerometer range: ");
  switch (mpu.getAccelerometerRange()) {
    case MPU6050_RANGE_2_G:
      Serial.println("±2G");
      break;
    case MPU6050_RANGE_4_G:
      Serial.println("±4G");
      break;
    case MPU6050_RANGE_8_G:
      Serial.println("±8G");
      break;
    case MPU6050_RANGE_16_G:
      Serial.println("±16G");
      break;
  }
  
  // ========================================
  // Configure Gyroscope
  // ========================================
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("✓ Gyro range: ");
  switch (mpu.getGyroRange()) {
    case MPU6050_RANGE_250_DEG:
      Serial.println("±250 deg/s");
      break;
    case MPU6050_RANGE_500_DEG:
      Serial.println("±500 deg/s");
      break;
    case MPU6050_RANGE_1000_DEG:
      Serial.println("±1000 deg/s");
      break;
    case MPU6050_RANGE_2000_DEG:
      Serial.println("±2000 deg/s");
      break;
  }
  
  // ========================================
  // Configure Digital Low Pass Filter
  // ========================================
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
  Serial.print("✓ Filter bandwidth: ");
  switch (mpu.getFilterBandwidth()) {
    case MPU6050_BAND_260_HZ:
      Serial.println("260 Hz");
      break;
    case MPU6050_BAND_184_HZ:
      Serial.println("184 Hz");
      break;
    case MPU6050_BAND_94_HZ:
      Serial.println("94 Hz");
      break;
    case MPU6050_BAND_44_HZ:
      Serial.println("44 Hz");
      break;
    case MPU6050_BAND_21_HZ:
      Serial.println("21 Hz");
      break;
    case MPU6050_BAND_10_HZ:
      Serial.println("10 Hz");
      break;
    case MPU6050_BAND_5_HZ:
      Serial.println("5 Hz");
      break;
  }
  
  // ========================================
  // Create FreeRTOS Queues
  // ========================================
  Serial.println();
  Serial.println("Creating FreeRTOS queues...");
  
  rawDataQueue = xQueueCreate(QUEUE_SIZE_RAW, sizeof(SensorData));
  processedDataQueue = xQueueCreate(QUEUE_SIZE_PROCESSED, sizeof(ProcessedData));
  
  if (rawDataQueue == NULL || processedDataQueue == NULL) {
    Serial.println("ERROR: Failed to create queues!");
    Serial.println("System halted.");
    while (1) {
      delay(1000);
    }
  }
  Serial.print("✓ Raw data queue created (size: ");
  Serial.print(QUEUE_SIZE_RAW);
  Serial.println(")");
  Serial.print("✓ Processed data queue created (size: ");
  Serial.print(QUEUE_SIZE_PROCESSED);
  Serial.println(")");
  
  // ========================================
  // Display System Configuration
  // ========================================
  Serial.println();
  Serial.println("========================================");
  Serial.println("System Configuration:");
  Serial.println("========================================");
  Serial.print("Sampling Rate: ");
  Serial.print(1000 / SAMPLE_RATE_MS);
  Serial.println(" Hz");
  Serial.print("Sample Interval: ");
  Serial.print(SAMPLE_RATE_MS);
  Serial.println(" ms");
  Serial.print("Moving Avg Window: ");
  Serial.print(MOVING_AVG_WINDOW);
  Serial.println(" samples");
  Serial.println();
  Serial.println("Task Distribution:");
  Serial.println("  Core 0: Data Acquisition (Priority 3)");
  Serial.println("  Core 1: Data Processing (Priority 2)");
  Serial.println("  Core 1: Data Output (Priority 1)");
  Serial.println("========================================");
  Serial.println();
  
  // ========================================
  // Create FreeRTOS Tasks
  // ========================================
  Serial.println("Creating FreeRTOS tasks...");
  
  // Task 1: Data Acquisition (Core 0, Highest Priority)
  BaseType_t task1 = xTaskCreatePinnedToCore(
    acquisitionTask,              // Task function
    "Data_Acquisition",           // Task name (for debugging)
    STACK_SIZE_ACQUISITION,       // Stack size (words)
    NULL,                         // Task parameters
    PRIORITY_ACQUISITION,         // Priority (0-24, higher = more priority)
    &acquisitionTaskHandle,       // Task handle
    0                             // Core ID (0 or 1)
  );
  
  if (task1 == pdPASS) {
    Serial.println("✓ Task 1: Data Acquisition created on Core 0");
  } else {
    Serial.println("ERROR: Failed to create Data Acquisition task!");
  }
  
  // Task 2: Data Processing (Core 1, Medium Priority)
  BaseType_t task2 = xTaskCreatePinnedToCore(
    processingTask,               // Task function
    "Data_Processing",            // Task name
    STACK_SIZE_PROCESSING,        // Stack size
    NULL,                         // Task parameters
    PRIORITY_PROCESSING,          // Priority
    &processingTaskHandle,        // Task handle
    1                             // Core ID
  );
  
  if (task2 == pdPASS) {
    Serial.println("✓ Task 2: Data Processing created on Core 1");
  } else {
    Serial.println("ERROR: Failed to create Data Processing task!");
  }
  
  // Task 3: Data Output (Core 1, Low Priority)
  BaseType_t task3 = xTaskCreatePinnedToCore(
    outputTask,                   // Task function
    "Data_Output",                // Task name
    STACK_SIZE_OUTPUT,            // Stack size
    NULL,                         // Task parameters
    PRIORITY_OUTPUT,              // Priority
    &outputTaskHandle,            // Task handle
    1                             // Core ID
  );
  
  if (task3 == pdPASS) {
    Serial.println("✓ Task 3: Data Output created on Core 1");
  } else {
    Serial.println("ERROR: Failed to create Data Output task!");
  }
  
  // ========================================
  // System Ready
  // ========================================
  Serial.println();
  Serial.println("========================================");
  Serial.println("System Ready! Starting data acquisition...");
  Serial.println("========================================");
  Serial.println();
  
  delay(1000);
}

// ============================================================================
// MAIN LOOP FUNCTION
// ============================================================================
/**
 * The loop() function is not used when FreeRTOS tasks are running
 * FreeRTOS scheduler automatically handles all task execution
 * This function becomes the idle task (lowest priority)
 */
void loop() {
  // FreeRTOS tasks handle everything
  // This runs as idle task (only when no other task is ready)
  
  // Optional: Print statistics every 10 seconds
  static unsigned long lastStatsTime = 0;
  unsigned long currentTime = millis();
  
  if (currentTime - lastStatsTime >= 10000) {
    lastStatsTime = currentTime;
    
    Serial.println();
    Serial.println("========================================");
    Serial.println("System Statistics:");
    Serial.println("========================================");
    Serial.print("Total samples: ");
    Serial.println(totalSamples);
    Serial.print("Dropped samples: ");
    Serial.println(droppedSamples);
    if (totalSamples > 0) {
      Serial.print("Drop rate: ");
      Serial.print((float)droppedSamples / totalSamples * 100, 2);
      Serial.println("%");
    }
    Serial.print("Free heap: ");
    Serial.print(ESP.getFreeHeap());
    Serial.println(" bytes");
    
    // Print task stack watermarks (minimum free stack ever reached)
    Serial.println();
    Serial.println("Task Stack Usage (lower = more used):");
    if (acquisitionTaskHandle != NULL) {
      Serial.print("  Acquisition: ");
      Serial.print(uxTaskGetStackHighWaterMark(acquisitionTaskHandle));
      Serial.println(" words free");
    }
    if (processingTaskHandle != NULL) {
      Serial.print("  Processing: ");
      Serial.print(uxTaskGetStackHighWaterMark(processingTaskHandle));
      Serial.println(" words free");
    }
    if (outputTaskHandle != NULL) {
      Serial.print("  Output: ");
      Serial.print(uxTaskGetStackHighWaterMark(outputTaskHandle));
      Serial.println(" words free");
    }
    Serial.println("========================================");
    Serial.println();
  }
  
  // Yield to other tasks
  vTaskDelay(pdMS_TO_TICKS(100));
}