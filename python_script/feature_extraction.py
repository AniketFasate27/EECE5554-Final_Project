# """
# Motor Fault Detection - Feature Extraction
# Extracts time-domain and frequency-domain features from IMU data
# """

# import pandas as pd
# import numpy as np
# from scipy import stats, signal
# from scipy.fft import fft, fftfreq
# import warnings
# warnings.filterwarnings('ignore')

# class MotorFeatureExtractor:
#     def __init__(self, sampling_rate=100):  # Hz
#         self.sampling_rate = sampling_rate
        
#     def extract_time_domain_features(self, data):
#         """Extract statistical features from time-domain signals"""
#         features = {}
        
#         # For each axis (Ax, Ay, Az, Gx, Gy, Gz)
#         for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']:
#             if col in data.columns:
#                 signal_data = data[col].values
                
#                 # Basic statistics
#                 features[f'{col}_mean'] = np.mean(signal_data)
#                 features[f'{col}_std'] = np.std(signal_data)
#                 features[f'{col}_var'] = np.var(signal_data)
#                 features[f'{col}_rms'] = np.sqrt(np.mean(signal_data**2))
#                 features[f'{col}_peak'] = np.max(np.abs(signal_data))
#                 features[f'{col}_peak2peak'] = np.ptp(signal_data)
                
#                 # Shape indicators
#                 features[f'{col}_skewness'] = stats.skew(signal_data)
#                 features[f'{col}_kurtosis'] = stats.kurtosis(signal_data)
                
#                 # Crest factor (peak / RMS) - sensitive to impacts
#                 rms = features[f'{col}_rms']
#                 if rms > 0:
#                     features[f'{col}_crest_factor'] = features[f'{col}_peak'] / rms
#                 else:
#                     features[f'{col}_crest_factor'] = 0
                
#                 # Impulse factor - bearing fault indicator
#                 mean_abs = np.mean(np.abs(signal_data))
#                 if mean_abs > 0:
#                     features[f'{col}_impulse_factor'] = features[f'{col}_peak'] / mean_abs
#                 else:
#                     features[f'{col}_impulse_factor'] = 0
                
#         return features
    
#     def extract_frequency_domain_features(self, data):
#         """Extract features from frequency spectrum (FFT)"""
#         features = {}
        
#         for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']:
#             if col in data.columns:
#                 signal_data = data[col].values
                
#                 # Compute FFT
#                 fft_values = np.abs(fft(signal_data))
#                 freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
                
#                 # Only positive frequencies
#                 positive_freq_idx = freqs > 0
#                 fft_values = fft_values[positive_freq_idx]
#                 freqs = freqs[positive_freq_idx]
                
#                 # Spectral features
#                 features[f'{col}_spectral_mean'] = np.mean(fft_values)
#                 features[f'{col}_spectral_std'] = np.std(fft_values)
#                 features[f'{col}_spectral_peak'] = np.max(fft_values)
                
#                 # Dominant frequency (highest peak)
#                 features[f'{col}_dominant_freq'] = freqs[np.argmax(fft_values)]
                
#                 # Spectral centroid (center of mass of spectrum)
#                 features[f'{col}_spectral_centroid'] = np.sum(freqs * fft_values) / np.sum(fft_values)
                
#                 # Spectral entropy (measure of randomness)
#                 psd = fft_values / np.sum(fft_values)
#                 features[f'{col}_spectral_entropy'] = -np.sum(psd * np.log2(psd + 1e-10))
                
#                 # Energy in specific frequency bands (imbalance, misalignment, etc.)
#                 # Low frequency (0-10 Hz) - imbalance
#                 low_freq_mask = (freqs >= 0) & (freqs < 10)
#                 features[f'{col}_energy_0_10Hz'] = np.sum(fft_values[low_freq_mask]**2)
                
#                 # Mid frequency (10-30 Hz) - misalignment
#                 mid_freq_mask = (freqs >= 10) & (freqs < 30)
#                 features[f'{col}_energy_10_30Hz'] = np.sum(fft_values[mid_freq_mask]**2)
                
#                 # High frequency (30+ Hz) - bearing defects
#                 high_freq_mask = freqs >= 30
#                 features[f'{col}_energy_30plus_Hz'] = np.sum(fft_values[high_freq_mask]**2)
                
#         return features
    
#     def extract_vibration_features(self, data):
#         """Extract overall vibration characteristics"""
#         features = {}
        
#         # Total vibration magnitude (from smoothed data if available)
#         if all(col in data.columns for col in ['Ax_Smooth', 'Ay_Smooth', 'Az_Smooth']):
#             vibration_mag = np.sqrt(
#                 data['Ax_Smooth']**2 + 
#                 data['Ay_Smooth']**2 + 
#                 data['Az_Smooth']**2
#             )
            
#             features['vibration_rms'] = np.sqrt(np.mean(vibration_mag**2))
#             features['vibration_peak'] = np.max(vibration_mag)
#             features['vibration_mean'] = np.mean(vibration_mag)
#             features['vibration_std'] = np.std(vibration_mag)
        
#         # Temperature features
#         if 'Temp' in data.columns:
#             features['temperature_mean'] = np.mean(data['Temp'])
#             features['temperature_std'] = np.std(data['Temp'])
#             features['temperature_max'] = np.max(data['Temp'])
        
#         return features
    
#     def extract_all_features(self, csv_file):
#         """Extract all features from a CSV file"""
#         # Read CSV
#         data = pd.read_csv(csv_file)
        
#         # Combine all features
#         features = {}
#         features.update(self.extract_time_domain_features(data))
#         features.update(self.extract_frequency_domain_features(data))
#         features.update(self.extract_vibration_features(data))
        
#         return features
    
#     def process_dataset(self, csv_files, labels):
#         """Process multiple CSV files and create feature matrix"""
#         all_features = []
#         all_labels = []
        
#         for csv_file, label in zip(csv_files, labels):
#             print(f"Processing: {csv_file} (Label: {label})")
#             features = self.extract_all_features(csv_file)
#             all_features.append(features)
#             all_labels.append(label)
        
#         # Convert to DataFrame
#         feature_df = pd.DataFrame(all_features)
#         feature_df['label'] = all_labels
        
#         return feature_df
    

#     # In feature_extraction.py, add windowing:
#     def extract_features_windowed(csv_file, label, window_size=1000, overlap=500):
#         """Extract features from sliding windows"""
#         data = pd.read_csv(csv_file)
#         features_list = []
        
#         # Create windows
#         for i in range(0, len(data) - window_size, overlap):
#             window = data[i:i+window_size]
#             features = extract_features_from_window(window)  # Your existing feature code
#             features['label'] = label
#             features_list.append(features)
        
#         return features_list


# # ============================================================================
# # USAGE EXAMPLE
# # ============================================================================

# if __name__ == "__main__":
#     # Initialize extractor
#     extractor = MotorFeatureExtractor(sampling_rate=100)  # Match your sampling rate
    
#     # List your CSV files and labels
#     csv_files = [
#         'motor_data/motor_healthy_trial1.csv',
#         'motor_data/motor_healthy_trial2.csv',
#         'motor_data/motor_imbalance_trial1.csv',
#         'motor_data/motor_imbalance_trial2.csv',
#         'motor_data/motor_misalignment_trial1.csv',
#         'motor_data/motor_bearing_fault_trial1.csv',
#     ]
    
#     labels = [
#         'healthy',
#         'healthy',
#         'imbalance',
#         'imbalance',
#         'misalignment',
#         'bearing_fault'
#     ]
    
#     # Extract features from all files
#     feature_dataset = extractor.process_dataset(csv_files, labels)
    
#     # Save feature dataset
#     feature_dataset.to_csv('motor_features.csv', index=False)
#     print("\nFeature extraction complete!")
#     print(f"Total features extracted: {len(feature_dataset.columns) - 1}")
#     print(f"Feature dataset shape: {feature_dataset.shape}")
#     print("\nFirst few rows:")
#     print(feature_dataset.head())



"""
Motor Fault Detection - Feature Extraction with Sliding Windows
Extracts features from CSV files using sliding window approach
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.fft import fft
import warnings
warnings.filterwarnings('ignore')

def calculate_fft_features(signal, sample_rate=100):
    """Calculate FFT-based features"""
    n = len(signal)
    fft_vals = np.abs(fft(signal))
    fft_freq = np.fft.fftfreq(n, 1/sample_rate)
    
    # Only positive frequencies
    positive_freq_idx = np.where(fft_freq > 0)
    fft_vals = fft_vals[positive_freq_idx]
    fft_freq = fft_freq[positive_freq_idx]
    
    # Find dominant frequencies
    top_indices = np.argsort(fft_vals)[-5:][::-1]
    
    features = {}
    for i, idx in enumerate(top_indices):
        features[f'fft_peak{i+1}_freq'] = fft_freq[idx]
        features[f'fft_peak{i+1}_mag'] = fft_vals[idx]
    
    # Spectral features
    features['fft_mean'] = np.mean(fft_vals)
    features['fft_std'] = np.std(fft_vals)
    features['fft_max'] = np.max(fft_vals)
    features['spectral_centroid'] = np.sum(fft_freq * fft_vals) / np.sum(fft_vals)
    
    return features

def extract_features_from_window(window_data):
    """Extract all features from a single window"""
    features = {}
    
    # Get sensor columns
    sensor_cols = ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']
    
    # Time-domain features for each sensor
    for col in sensor_cols:
        if col in window_data.columns:
            signal = window_data[col].values
            
            # Basic statistics
            features[f'{col}_mean'] = np.mean(signal)
            features[f'{col}_std'] = np.std(signal)
            features[f'{col}_var'] = np.var(signal)
            features[f'{col}_min'] = np.min(signal)
            features[f'{col}_max'] = np.max(signal)
            features[f'{col}_range'] = np.ptp(signal)
            features[f'{col}_rms'] = np.sqrt(np.mean(signal**2))
            features[f'{col}_skew'] = stats.skew(signal)
            features[f'{col}_kurtosis'] = stats.kurtosis(signal)
            
            # FFT features
            fft_features = calculate_fft_features(signal)
            for key, value in fft_features.items():
                features[f'{col}_{key}'] = value
    
    # Combined vibration magnitude
    if all(col in window_data.columns for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw']):
        accel_magnitude = np.sqrt(
            window_data['Ax_Raw']**2 + 
            window_data['Ay_Raw']**2 + 
            window_data['Az_Raw']**2
        )
        features['accel_magnitude_mean'] = np.mean(accel_magnitude)
        features['accel_magnitude_std'] = np.std(accel_magnitude)
        features['accel_magnitude_max'] = np.max(accel_magnitude)
    
    if all(col in window_data.columns for col in ['Gx_Raw', 'Gy_Raw', 'Gz_Raw']):
        gyro_magnitude = np.sqrt(
            window_data['Gx_Raw']**2 + 
            window_data['Gy_Raw']**2 + 
            window_data['Gz_Raw']**2
        )
        features['gyro_magnitude_mean'] = np.mean(gyro_magnitude)
        features['gyro_magnitude_std'] = np.std(gyro_magnitude)
        features['gyro_magnitude_max'] = np.max(gyro_magnitude)
    
    # Temperature features
    if 'temperature' in window_data.columns:
        features['temperature_mean'] = np.mean(window_data['temperature'])
        features['temperature_std'] = np.std(window_data['temperature'])
        features['temperature_max'] = np.max(window_data['temperature'])
    
    return features

def process_csv_with_windows(csv_file, label, window_size=1000, step_size=500):
    """
    Process CSV file with sliding windows
    
    Parameters:
    - window_size: Number of samples per window (default 1000 = 10 seconds at 100Hz)
    - step_size: Number of samples to slide (default 500 = 50% overlap)
    """
    print(f"Processing: {csv_file} (Label: {label})")
    
    # Read CSV
    data = pd.read_csv(csv_file)
    
    # Calculate number of windows
    num_windows = (len(data) - window_size) // step_size + 1
    print(f"  Total samples: {len(data)}")
    print(f"  Creating {num_windows} windows (size={window_size}, step={step_size})")
    
    features_list = []
    
    # Extract features from each window
    for i in range(0, len(data) - window_size + 1, step_size):
        window = data[i:i+window_size]
        
        # Extract features
        features = extract_features_from_window(window)
        features['label'] = label
        features['window_start'] = i
        
        features_list.append(features)
    
    return pd.DataFrame(features_list)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOTOR FAULT DETECTION - FEATURE EXTRACTION")
    print("Using Sliding Window Approach")
    print("="*60 + "\n")
    
    # Define your motor data files
    data_files = [
        ('motor_data/motor_healthy_trial1.csv', 'healthy'),
        ('motor_data/motor_healthy_trial2.csv', 'healthy'),
        ('motor_data/motor_imbalance_trial1.csv', 'imbalance'),
        ('motor_data/motor_imbalance_trial2.csv', 'imbalance'),
        ('motor_data/motor_misalignment_trial1.csv', 'misalignment'),
        ('motor_data/motor_bearing_fault_trial1.csv', 'bearing_fault'),
    ]
    
    # Window parameters
    WINDOW_SIZE = 1000   # 10 seconds at 100Hz
    STEP_SIZE = 500      # 50% overlap (5 seconds)
    
    print(f"Window Configuration:")
    print(f"  Window size: {WINDOW_SIZE} samples (10 seconds)")
    print(f"  Step size: {STEP_SIZE} samples (5 seconds)")
    print(f"  Overlap: 50%\n")
    
    all_features = []
    
    # Process each file
    for csv_file, label in data_files:
        try:
            features_df = process_csv_with_windows(csv_file, label, WINDOW_SIZE, STEP_SIZE)
            all_features.append(features_df)
        except Exception as e:
            print(f"  ERROR processing {csv_file}: {e}")
            continue
    
    # Combine all features
    if all_features:
        final_features = pd.concat(all_features, ignore_index=True)
        
        # Remove the window_start column (not needed for ML)
        if 'window_start' in final_features.columns:
            final_features = final_features.drop('window_start', axis=1)
        
        # Save to CSV
        output_file = 'motor_features.csv'
        final_features.to_csv(output_file, index=False)
        
        print("\n" + "="*60)
        print("Feature extraction complete!")
        print("="*60)
        print(f"Total windows/samples extracted: {len(final_features)}")
        print(f"Total features per sample: {len(final_features.columns) - 1}")
        print(f"Feature dataset shape: {final_features.shape}")
        print(f"Output saved to: {output_file}")
        
        # Show class distribution
        print(f"\nClass distribution:")
        print(final_features['label'].value_counts())
        
        print("\nFirst few rows:")
        print(final_features.head())
        print("="*60 + "\n")
    else:
        print("\nERROR: No features extracted!")