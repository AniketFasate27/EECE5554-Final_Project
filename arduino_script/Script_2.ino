// Smart Motor Health Diagnostics - Level 1
// Optimized MPU6050 code for motor vibration monitoring
// Team: Sofia Makowska, Jeje Dennis, Aniket Fasate, Madison O'Neil

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <Wire.h>

Adafruit_MPU6050 mpu;

// Sampling configuration
const int SAMPLE_RATE_MS = 10;  // 100 Hz sampling rate (adjust as needed)
unsigned long lastSampleTime = 0;

void setup(void) {
  Serial.begin(115200);
  
  // Wait for serial connection (comment out for standalone operation)
  while (!Serial) {
    delay(10);
  }

  Serial.println("=== Smart Motor Health Diagnostics System ===");
  Serial.println("Initializing MPU6050...");

  // Try to initialize!
  if (!mpu.begin()) {
    Serial.println("ERROR: Failed to find MPU6050 chip");
    Serial.println("Check wiring: SDA->GPIO21, SCL->GPIO22 (ESP32)");
    while (1) {
      delay(10);
    }
  }
  Serial.println("MPU6050 Found!");

  // Configure accelerometer range
  // ±8G is good for most motor vibrations
  // Use ±16G if you expect very strong vibrations
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  Serial.print("Accelerometer range: ");
  switch (mpu.getAccelerometerRange()) {
    case MPU6050_RANGE_2_G:
      Serial.println("+-2G");
      break;
    case MPU6050_RANGE_4_G:
      Serial.println("+-4G");
      break;
    case MPU6050_RANGE_8_G:
      Serial.println("+-8G");
      break;
    case MPU6050_RANGE_16_G:
      Serial.println("+-16G");
      break;
  }

  // Configure gyroscope range
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  Serial.print("Gyro range: ");
  switch (mpu.getGyroRange()) {
    case MPU6050_RANGE_250_DEG:
      Serial.println("+- 250 deg/s");
      break;
    case MPU6050_RANGE_500_DEG:
      Serial.println("+- 500 deg/s");
      break;
    case MPU6050_RANGE_1000_DEG:
      Serial.println("+- 1000 deg/s");
      break;
    case MPU6050_RANGE_2000_DEG:
      Serial.println("+- 2000 deg/s");
      break;
  }

  // CRITICAL: Set high bandwidth for motor vibration detection
  // 260 Hz allows detection of vibrations up to ~130 Hz (Nyquist)
  mpu.setFilterBandwidth(MPU6050_BAND_260_HZ);
  Serial.print("Filter bandwidth: ");
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

  Serial.println("\nSystem Ready!");
  Serial.println("Data Format: Timestamp(ms), Ax, Ay, Az, Gx, Gy, Gz, Temp");
  Serial.println("Units: m/s^2 (accel), rad/s (gyro), degC (temp)");
  Serial.println("-----------------------------------------------------------");
  
  delay(1000);
  
  // Print CSV header for Telemetry Viewer
  Serial.println("Time,Accel_X,Accel_Y,Accel_Z,Gyro_X,Gyro_Y,Gyro_Z,Temperature");
}

void loop() {
  unsigned long currentTime = millis();
  
  // Maintain consistent sampling rate
  if (currentTime - lastSampleTime >= SAMPLE_RATE_MS) {
    lastSampleTime = currentTime;
    
    // Get new sensor events with the readings
    sensors_event_t a, g, temp;
    mpu.getEvent(&a, &g, &temp);

    // Print in CSV format for easy parsing and visualization
    // Format: timestamp, ax, ay, az, gx, gy, gz, temp
    Serial.print(currentTime);
    Serial.print(",");
    
    // Acceleration data (m/s^2)
    Serial.print(a.acceleration.x, 4);
    Serial.print(",");
    Serial.print(a.acceleration.y, 4);
    Serial.print(",");
    Serial.print(a.acceleration.z, 4);
    Serial.print(",");
    
    // Gyroscope data (rad/s)
    Serial.print(g.gyro.x, 4);
    Serial.print(",");
    Serial.print(g.gyro.y, 4);
    Serial.print(",");
    Serial.print(g.gyro.z, 4);
    Serial.print(",");
    
    // Temperature (degC)
    Serial.println(temp.temperature, 2);
  }
}