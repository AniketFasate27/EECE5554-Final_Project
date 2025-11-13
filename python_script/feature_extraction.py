"""
Motor Fault Detection - Feature Extraction
Extracts time-domain and frequency-domain features from IMU data
"""

import pandas as pd
import numpy as np
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

class MotorFeatureExtractor:
    def __init__(self, sampling_rate=100):  # Hz
        self.sampling_rate = sampling_rate
        
    def extract_time_domain_features(self, data):
        """Extract statistical features from time-domain signals"""
        features = {}
        
        # For each axis (Ax, Ay, Az, Gx, Gy, Gz)
        for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']:
            if col in data.columns:
                signal_data = data[col].values
                
                # Basic statistics
                features[f'{col}_mean'] = np.mean(signal_data)
                features[f'{col}_std'] = np.std(signal_data)
                features[f'{col}_var'] = np.var(signal_data)
                features[f'{col}_rms'] = np.sqrt(np.mean(signal_data**2))
                features[f'{col}_peak'] = np.max(np.abs(signal_data))
                features[f'{col}_peak2peak'] = np.ptp(signal_data)
                
                # Shape indicators
                features[f'{col}_skewness'] = stats.skew(signal_data)
                features[f'{col}_kurtosis'] = stats.kurtosis(signal_data)
                
                # Crest factor (peak / RMS) - sensitive to impacts
                rms = features[f'{col}_rms']
                if rms > 0:
                    features[f'{col}_crest_factor'] = features[f'{col}_peak'] / rms
                else:
                    features[f'{col}_crest_factor'] = 0
                
                # Impulse factor - bearing fault indicator
                mean_abs = np.mean(np.abs(signal_data))
                if mean_abs > 0:
                    features[f'{col}_impulse_factor'] = features[f'{col}_peak'] / mean_abs
                else:
                    features[f'{col}_impulse_factor'] = 0
                
        return features
    
    def extract_frequency_domain_features(self, data):
        """Extract features from frequency spectrum (FFT)"""
        features = {}
        
        for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']:
            if col in data.columns:
                signal_data = data[col].values
                
                # Compute FFT
                fft_values = np.abs(fft(signal_data))
                freqs = fftfreq(len(signal_data), 1/self.sampling_rate)
                
                # Only positive frequencies
                positive_freq_idx = freqs > 0
                fft_values = fft_values[positive_freq_idx]
                freqs = freqs[positive_freq_idx]
                
                # Spectral features
                features[f'{col}_spectral_mean'] = np.mean(fft_values)
                features[f'{col}_spectral_std'] = np.std(fft_values)
                features[f'{col}_spectral_peak'] = np.max(fft_values)
                
                # Dominant frequency (highest peak)
                features[f'{col}_dominant_freq'] = freqs[np.argmax(fft_values)]
                
                # Spectral centroid (center of mass of spectrum)
                features[f'{col}_spectral_centroid'] = np.sum(freqs * fft_values) / np.sum(fft_values)
                
                # Spectral entropy (measure of randomness)
                psd = fft_values / np.sum(fft_values)
                features[f'{col}_spectral_entropy'] = -np.sum(psd * np.log2(psd + 1e-10))
                
                # Energy in specific frequency bands (imbalance, misalignment, etc.)
                # Low frequency (0-10 Hz) - imbalance
                low_freq_mask = (freqs >= 0) & (freqs < 10)
                features[f'{col}_energy_0_10Hz'] = np.sum(fft_values[low_freq_mask]**2)
                
                # Mid frequency (10-30 Hz) - misalignment
                mid_freq_mask = (freqs >= 10) & (freqs < 30)
                features[f'{col}_energy_10_30Hz'] = np.sum(fft_values[mid_freq_mask]**2)
                
                # High frequency (30+ Hz) - bearing defects
                high_freq_mask = freqs >= 30
                features[f'{col}_energy_30plus_Hz'] = np.sum(fft_values[high_freq_mask]**2)
                
        return features
    
    def extract_vibration_features(self, data):
        """Extract overall vibration characteristics"""
        features = {}
        
        # Total vibration magnitude (from smoothed data if available)
        if all(col in data.columns for col in ['Ax_Smooth', 'Ay_Smooth', 'Az_Smooth']):
            vibration_mag = np.sqrt(
                data['Ax_Smooth']**2 + 
                data['Ay_Smooth']**2 + 
                data['Az_Smooth']**2
            )
            
            features['vibration_rms'] = np.sqrt(np.mean(vibration_mag**2))
            features['vibration_peak'] = np.max(vibration_mag)
            features['vibration_mean'] = np.mean(vibration_mag)
            features['vibration_std'] = np.std(vibration_mag)
        
        # Temperature features
        if 'Temp' in data.columns:
            features['temperature_mean'] = np.mean(data['Temp'])
            features['temperature_std'] = np.std(data['Temp'])
            features['temperature_max'] = np.max(data['Temp'])
        
        return features
    
    def extract_all_features(self, csv_file):
        """Extract all features from a CSV file"""
        # Read CSV
        data = pd.read_csv(csv_file)
        
        # Combine all features
        features = {}
        features.update(self.extract_time_domain_features(data))
        features.update(self.extract_frequency_domain_features(data))
        features.update(self.extract_vibration_features(data))
        
        return features
    
    def process_dataset(self, csv_files, labels):
        """Process multiple CSV files and create feature matrix"""
        all_features = []
        all_labels = []
        
        for csv_file, label in zip(csv_files, labels):
            print(f"Processing: {csv_file} (Label: {label})")
            features = self.extract_all_features(csv_file)
            all_features.append(features)
            all_labels.append(label)
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(all_features)
        feature_df['label'] = all_labels
        
        return feature_df


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Initialize extractor
    extractor = MotorFeatureExtractor(sampling_rate=100)  # Match your sampling rate
    
    # List your CSV files and labels
    csv_files = [
        'motor_data/motor_healthy_trial1.csv',
        'motor_data/motor_healthy_trial2.csv',
        'motor_data/motor_imbalance_trial1.csv',
        'motor_data/motor_imbalance_trial2.csv',
        'motor_data/motor_misalignment_trial1.csv',
        'motor_data/motor_bearing_fault_trial1.csv',
    ]
    
    labels = [
        'healthy',
        'healthy',
        'imbalance',
        'imbalance',
        'misalignment',
        'bearing_fault'
    ]
    
    # Extract features from all files
    feature_dataset = extractor.process_dataset(csv_files, labels)
    
    # Save feature dataset
    feature_dataset.to_csv('motor_features.csv', index=False)
    print("\nFeature extraction complete!")
    print(f"Total features extracted: {len(feature_dataset.columns) - 1}")
    print(f"Feature dataset shape: {feature_dataset.shape}")
    print("\nFirst few rows:")
    print(feature_dataset.head())