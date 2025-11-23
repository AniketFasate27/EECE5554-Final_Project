# Smart Motor Health Diagnostics System

## 🎯 Project Overview

A real-time motor health diagnostics system for predictive maintenance using ESP32, MPU6050 IMU sensor, and machine learning to detect motor faults with 100% accuracy.

## 🔧 Hardware

- **ESP32** (Dual-core microcontroller)
- **MPU6050** (6-axis IMU sensor)
- **Motor** (Test subject)

## ✨ Features

- Real-time vibration monitoring at 100 Hz
- Dual-core FreeRTOS architecture
- Machine learning fault classification
- Detects: Healthy, Imbalance, Misalignment, Bearing Fault
- 100% test accuracy with Random Forest model

## 📊 Results

- **372 training samples** from sliding window approach
- **144 features** extracted per sample
- **100% accuracy** on 75-sample test set
- All 5 ML models (Random Forest, SVM, Neural Network, etc.) achieved perfect classification

## 🚀 Quick Start

### 1. Hardware Setup
```
MPU6050 → ESP32
VCC     → 3.3V
GND     → GND
SDA     → GPIO 21
SCL     → GPIO 22
```

### 2. Upload Firmware
- Navigate to `arduino_script/DC_MPU/`
- Open `.ino` file in Arduino IDE
- Upload to ESP32

### 3. Collect Data
```bash
cd python_script
python data_saving.py
```
Collect data for each motor condition and save to `Data/motor_data/`:
- `motor_healthy_trial1.csv`
- `motor_imbalance_trial1.csv`
- `motor_misalignment_trial1.csv`
- `motor_bearing_fault_trial1.csv`

### 4. Train Model
```bash
python feature_extraction.py      # Extract 144 features → motor_features.csv
python ml_learning.py             # Train ML models → motor_fault_detector.pkl
```

### 5. Visualize Results
```bash
python comparison_visualization.py      # Comparison dashboard
python validate.py                      # Validate model
```

## 📁 Repository Structure

```
EECE5554-FINAL_PROJECT/
│
├── Analysis/                           # Generated visualizations
│   ├── cmd_feature_extraction_output.png
│   ├── confusion_matrix.png
│   ├── fault_comparison_dashboard.png
│   ├── fault_detection_bearing_fault.png
│   ├── fault_detection_healthy.png
│   ├── fault_detection_imbalance.png
│   ├── fault_detection_misalignment.png
│   ├── feature_importance.png
│   └── Schematic ESP32 and MPU 6050.png
│
├── arduino_script/                     # ESP32 firmware
│   ├── DC_MPU/                         # Basic MPU6050 test
│   └── FREERTOS_MovingAvg/             # Main FreeRTOS implementation
│       └── FREERTOS_MovingAvg.ino      # Dual-core data acquisition
│
├── Data/                               # Raw sensor data
│   └── motor_data/
│       ├── motor_bearing_fault_trial1.csv
│       ├── motor_healthy_trial1.csv
│       ├── motor_healthy_trial2.csv
│       ├── motor_imbalance_trial1.csv
│       ├── motor_imbalance_trial2.csv
│       └── motor_misalignment_trial1.csv
│
├── python_script/                      # ML pipeline
│   ├── motor_data/                     # Symlink to Data/motor_data
│   ├── architecture.py                 # System architecture diagram
│   ├── comparison_visualization.py     # Multi-fault comparison plots
│   ├── data_saving.py                  # Automated CSV data logger
│   ├── feature_extraction.py           # Sliding window feature extraction
│   ├── ml_learning.py                  # ML model training (5 algorithms)
│   ├── realtime_prediction.py          # Real-time inference demo
│   ├── validate.py                     # Model validation script
│   ├── motor_features.csv              # Extracted features (372 samples, 144 features)
│   └── motor_fault_detector.pkl        # Trained Random Forest model
│
├── readme.md                           # This file
└── Screenshot 2025-11-04 232412.png    # Project documentation
```

## 🎓 Project Levels

### Level 1: Data Acquisition ✅
- Configured MPU6050 for motor vibration detection (260 Hz bandwidth)
- Dual-core FreeRTOS implementation:
  - **Core 0**: High-priority data acquisition at 100 Hz
  - **Core 1**: Data processing with moving average filter
- Queue-based inter-task communication (2048 samples buffer)
- Zero dropped samples, stable temperature monitoring

### Level 2: Machine Learning ✅
- Collected **156,000+ sensor readings** across 4 fault types
- Sliding window feature extraction:
  - Window size: 1000 samples (10 seconds)
  - Step size: 500 samples (50% overlap)
  - Result: 372 training samples
- **144 features per sample**:
  - Time-domain: Mean, Std, RMS, Skewness, Kurtosis
  - Frequency-domain: FFT peaks, Spectral centroid
  - Vibration metrics: Combined acceleration/gyroscope magnitudes
- Trained 5 ML algorithms:
  - Random Forest ✅
  - Gradient Boosting ✅
  - SVM (RBF) ✅
  - Neural Network ✅
  - K-Nearest Neighbors ✅
- **Achieved 100% accuracy** with all models

### Level 3: Visualization & Deployment ✅
- Real-time fault detection visualization
- Comparative analysis dashboard (4 faults side-by-side)
- Prediction confidence scoring
- Model serialization for deployment

## 📈 Performance Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 100% |
| **Precision** | 100% (all classes) |
| **Recall** | 100% (all classes) |
| **F1-Score** | 100% (all classes) |
| **Training Samples** | 297 |
| **Test Samples** | 75 |
| **Features** | 144 |
| **Sampling Rate** | 100 Hz |

### Test Set Breakdown:
- Bearing Fault: 8/8 correct ✅
- Healthy: 27/27 correct ✅
- Imbalance: 27/27 correct ✅
- Misalignment: 13/13 correct ✅

## 🛠️ Installation

### Requirements
```bash
# Python packages
pip install pandas numpy scipy scikit-learn matplotlib seaborn joblib

# Arduino libraries
- ESP32 Board Support
- MPU6050 Library (Electronic Cats)
- FreeRTOS (included with ESP32)
```

### Hardware Wiring
```
MPU6050    ESP32
────────   ─────
VCC    →   3.3V
GND    →   GND
SDA    →   GPIO 21
SCL    →   GPIO 22
```

## 💡 Key Technical Details

### FreeRTOS Architecture
- **Core 0** (Priority 2): Sensor reading at precise 10ms intervals
- **Core 1** (Priority 1): Moving average filter (window size: 5)
- **Communication**: Queue-based data passing (2048 samples)

### Feature Extraction
- **Sliding Window**: 1000 samples with 50% overlap
- **Feature Categories**:
  - Statistical (54 features)
  - Frequency analysis (72 features)
  - Vibration magnitude (6 features)
  - Temperature (3 features)
  - Advanced metrics (9 features)

### Machine Learning
- **Best Model**: Random Forest (100 trees, max depth 10)
- **Training Time**: <5 seconds
- **Inference Time**: <10ms per prediction
- **Model Size**: ~2.5 MB (serialized)

## 🎯 Use Cases

- ✅ Predictive maintenance scheduling
- ✅ Early fault detection before catastrophic failure
- ✅ Automated quality control in manufacturing
- ✅ Remote motor health monitoring (IIoT)
- ✅ Research and educational demonstrations

## 🚀 Future Enhancements

- [ ] IR sensor integration for RPM measurement
- [ ] On-device inference with TensorFlow Lite
- [ ] Web dashboard for fleet monitoring
- [ ] Remaining Useful Life (RUL) prediction
- [ ] Multi-motor simultaneous monitoring

## 👥 Team


- **Sofia Makowska**
- **Jeje Dennis**
- **Madison O'Neil**
- **Aniket Fasate**

**Course**: EECE 5554 - Robot Snesing and Navigation
**Semester**: Fall 2024

