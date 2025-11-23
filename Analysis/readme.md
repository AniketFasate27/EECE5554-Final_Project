Data Collection (data_saving.py)
The data_saving.py Python script records raw sensor data from the controller, including three axes of accelerometer and three axes of gyroscope readings, and stores this data in CSV format. Initially, it gathers baseline (healthy system) data for approximately 30 minutes. Next, it collects data during induced fault trials—such as imbalance, misalignment, and bearing faults—each for another 30 minutes to build datasets suitable for training machine learning algorithms.

Feature Extraction (feature_extraction.py)
This script processes the gathered CSV files from different fault scenarios to extract relevant features required for machine learning analysis. Feature extraction transforms raw sensor data into informative statistical and signal-based descriptors suitable for use in predictive models.

Machine Learning Model Training (ml_learning.py)
The ml_learning.py script uses the extracted features to train a Random Forest algorithm. Training is performed on labeled datasets, including healthy and various fault conditions, to enable robust prediction capabilities for motor health diagnostics.

Real-Time Fault Prediction (realtime_predictions.py)
This module reads live data from the serial port in real time, applies the trained machine learning model, and automatically predicts motor faults as they occur. It is deployed for online monitoring and immediate fault detection based on sensor input.




============================================================
Logging stopped by user (Ctrl+C)
============================================================
Total samples logged: 34592
Errors encountered: 0
Success rate: 100.0%
Data saved to: motor_data/motor_data_20251123_012508.csv
============================================================

motor_data/motor_imbalance_trial1.csv
============================================================
Logging stopped by user (Ctrl+C)
============================================================
Total samples logged: 37922
Errors encountered: 0
Success rate: 100.0%
Data saved to: motor_data/motor_data_20251123_015533.csv
============================================================

motor_data/motor_imbalance_trial2.csv'
============================================================
Logging stopped by user (Ctrl+C)
============================================================
Total samples logged: 30927
Errors encountered: 0
Success rate: 100.0%
Data saved to: motor_data/motor_data_20251123_020251.csv
============================================================



motor_data/motor_misalignment_trial1.csv
============================================================
Logging stopped by user (Ctrl+C)
============================================================
Total samples logged: 33092
Errors encountered: 0
Success rate: 100.0%
Data saved to: motor_data/motor_data_20251123_020859.csv
============================================================




motor_data/motor_bearing_fault_trial1.csv
============================================================
Logging stopped by user (Ctrl+C)
============================================================
Total samples logged: 19347
Errors encountered: 0
Success rate: 100.0%
Data saved to: motor_data/motor_data_20251123_021529.csv
============================================================




