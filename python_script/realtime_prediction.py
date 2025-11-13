"""
Real-Time Motor Fault Detection
Monitors serial data and predicts faults in real-time
"""

import serial
import pandas as pd
import numpy as np
import joblib
from collections import deque
import time

class RealTimeMotorMonitor:
    def __init__(self, model_file, serial_port, baud_rate=115200):
        # Load trained model
        model_data = joblib.load(model_file)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        # Serial connection
        self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
        
        # Buffer for storing recent data (100 samples = 1 second at 100Hz)
        self.buffer_size = 100
        self.data_buffer = {
            'Ax_Raw': deque(maxlen=self.buffer_size),
            'Ay_Raw': deque(maxlen=self.buffer_size),
            'Az_Raw': deque(maxlen=self.buffer_size),
            'Gx_Raw': deque(maxlen=self.buffer_size),
            'Gy_Raw': deque(maxlen=self.buffer_size),
            'Gz_Raw': deque(maxlen=self.buffer_size),
        }
        
        self.feature_extractor = MotorFeatureExtractor(sampling_rate=100)
        
    def extract_features_from_buffer(self):
        """Extract features from current buffer"""
        # Convert buffer to DataFrame
        df = pd.DataFrame(self.data_buffer)
        
        if len(df) < self.buffer_size:
            return None  # Not enough data yet
        
        # Extract features
        features = {}
        features.update(self.feature_extractor.extract_time_domain_features(df))
        features.update(self.feature_extractor.extract_frequency_domain_features(df))
        
        return list(features.values())
    
    def monitor(self):
        """Real-time monitoring loop"""
        print("Starting real-time motor monitoring...")
        print("Waiting for data...\n")
        
        header_found = False
        sample_count = 0
        
        while True:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8').strip()
                
                # Skip header and info lines
                if 'Time,Ax_Raw' in line:
                    header_found = True
                    continue
                
                if not header_found or line.startswith('[') or line.startswith('='):
                    continue
                
                # Parse CSV line
                try:
                    values = [float(x) for x in line.split(',')]
                    
                    # Add to buffer
                    self.data_buffer['Ax_Raw'].append(values[1])
                    self.data_buffer['Ay_Raw'].append(values[2])
                    self.data_buffer['Az_Raw'].append(values[3])
                    self.data_buffer['Gx_Raw'].append(values[4])
                    self.data_buffer['Gy_Raw'].append(values[5])
                    self.data_buffer['Gz_Raw'].append(values[6])
                    
                    sample_count += 1
                    
                    # Predict every 50 samples (0.5 seconds)
                    if sample_count % 50 == 0 and sample_count >= self.buffer_size:
                        features = self.extract_features_from_buffer()
                        
                        if features is not None:
                            # Scale and predict
                            features_scaled = self.scaler.transform([features])
                            prediction = self.model.predict(features_scaled)[0]
                            proba = self.model.predict_proba(features_scaled)[0]
                            
                            fault_type = self.label_encoder.inverse_transform([prediction])[0]
                            confidence = proba[prediction] * 100
                            
                            # Display result
                            print(f"\n{'='*50}")
                            print(f"Prediction: {fault_type.upper()}")
                            print(f"Confidence: {confidence:.1f}%")
                            print(f"{'='*50}")
                            
                            # Alert if fault detected
                            if fault_type != 'healthy' and confidence > 80:
                                print("⚠️  FAULT DETECTED! ⚠️")
                
                except Exception as e:
                    pass  # Skip malformed lines


# ============================================================================
# RUN REAL-TIME MONITOR
# ============================================================================

if __name__ == "__main__":
    monitor = RealTimeMotorMonitor(
        model_file='motor_fault_detector.pkl',
        serial_port='COM6',  # Change to your port
        baud_rate=115200
    )
    
    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\nMonitoring stopped")