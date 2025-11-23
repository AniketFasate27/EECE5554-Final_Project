"""
Real-Time Fault Detection Visualization
Shows live predictions with confidence scores and sensor data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.fft import fft
from scipy import stats

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)

class FaultDetectionVisualizer:
    def __init__(self, model_path='motor_fault_detector.pkl', feature_csv='motor_features.csv'):
        """Load trained model and get feature names"""
        print("Loading trained model...")
        model_data = joblib.load(model_path)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        
        # Load feature dataset to get exact feature names and order
        features_df = pd.read_csv(feature_csv)
        self.feature_names = [col for col in features_df.columns if col != 'label']
        
        print(f"Model loaded: {model_data['model_name']}")
        print(f"Classes: {self.label_encoder.classes_}")
        print(f"Expected features: {len(self.feature_names)}\n")
    
    def extract_features_from_window(self, window_data):
        """Extract features from a data window (same as training)"""
        features = {}
        sensor_cols = ['Ax_Raw', 'Ay_Raw', 'Az_Raw', 'Gx_Raw', 'Gy_Raw', 'Gz_Raw']
        
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
                fft_vals = np.abs(fft(signal))
                fft_freq = np.fft.fftfreq(len(signal), 1/100)
                positive_idx = np.where(fft_freq > 0)
                fft_vals = fft_vals[positive_idx]
                fft_freq = fft_freq[positive_idx]
                
                top_indices = np.argsort(fft_vals)[-5:][::-1]
                for i, idx in enumerate(top_indices):
                    features[f'{col}_fft_peak{i+1}_freq'] = fft_freq[idx]
                    features[f'{col}_fft_peak{i+1}_mag'] = fft_vals[idx]
                
                features[f'{col}_fft_mean'] = np.mean(fft_vals)
                features[f'{col}_fft_std'] = np.std(fft_vals)
                features[f'{col}_fft_max'] = np.max(fft_vals)
                features[f'{col}_spectral_centroid'] = np.sum(fft_freq * fft_vals) / np.sum(fft_vals)
        
        # Combined magnitudes
        if all(col in window_data.columns for col in ['Ax_Raw', 'Ay_Raw', 'Az_Raw']):
            accel_mag = np.sqrt(window_data['Ax_Raw']**2 + window_data['Ay_Raw']**2 + window_data['Az_Raw']**2)
            features['accel_magnitude_mean'] = np.mean(accel_mag)
            features['accel_magnitude_std'] = np.std(accel_mag)
            features['accel_magnitude_max'] = np.max(accel_mag)
        
        if all(col in window_data.columns for col in ['Gx_Raw', 'Gy_Raw', 'Gz_Raw']):
            gyro_mag = np.sqrt(window_data['Gx_Raw']**2 + window_data['Gy_Raw']**2 + window_data['Gz_Raw']**2)
            features['gyro_magnitude_mean'] = np.mean(gyro_mag)
            features['gyro_magnitude_std'] = np.std(gyro_mag)
            features['gyro_magnitude_max'] = np.max(gyro_mag)
        
        # Temperature - use Temp column
        if 'Temp' in window_data.columns:
            features['temperature_mean'] = np.mean(window_data['Temp'])
            features['temperature_std'] = np.std(window_data['Temp'])
            features['temperature_max'] = np.max(window_data['Temp'])
        
        return features
    
    def predict_fault(self, features_dict):
        """Predict fault from features"""
        # Create feature array in the EXACT order the model expects
        features_array = []
        for feature_name in self.feature_names:
            if feature_name in features_dict:
                features_array.append(features_dict[feature_name])
            else:
                # If feature is missing, use 0 as default
                features_array.append(0.0)
                print(f"Warning: Missing feature '{feature_name}', using 0.0")
        
        features_array = np.array([features_array])
        
        # Scale features
        features_scaled = self.scaler.transform(features_array)
        
        # Predict
        prediction = self.model.predict(features_scaled)[0]
        prediction_proba = self.model.predict_proba(features_scaled)[0]
        
        # Get label and confidence
        fault_type = self.label_encoder.inverse_transform([prediction])[0]
        confidence = prediction_proba[prediction] * 100
        
        # Get all class probabilities
        all_probs = {self.label_encoder.classes_[i]: prediction_proba[i] * 100 
                     for i in range(len(self.label_encoder.classes_))}
        
        return fault_type, confidence, all_probs
    
    def visualize_fault_detection(self, csv_file, true_label, window_size=1000):
        """Create comprehensive fault detection visualization"""
        print(f"\nAnalyzing: {csv_file}")
        print(f"True Label: {true_label}")
        
        try:
            # Load data
            data = pd.read_csv(csv_file)
            
            # Take middle window for analysis
            start_idx = len(data) // 2 - window_size // 2
            window = data.iloc[start_idx:start_idx + window_size]
            
            # Extract features and predict
            features = self.extract_features_from_window(window)
            fault_type, confidence, all_probs = self.predict_fault(features)
            
            print(f"Predicted: {fault_type} (Confidence: {confidence:.2f}%)")
            
            # Create visualization
            fig = plt.figure(figsize=(18, 12))
            gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
            
            # Title
            fig.suptitle(f'Motor Fault Detection Analysis\nTrue Label: {true_label.upper()} | Predicted: {fault_type.upper()} ({confidence:.1f}% confidence)', 
                         fontsize=16, fontweight='bold', y=0.98)
            
            # 1. Accelerometer time series
            ax1 = fig.add_subplot(gs[0, :])
            time = np.arange(len(window)) / 100  # Convert to seconds
            ax1.plot(time, window['Ax_Raw'], label='X-axis', alpha=0.7, linewidth=1)
            ax1.plot(time, window['Ay_Raw'], label='Y-axis', alpha=0.7, linewidth=1)
            ax1.plot(time, window['Az_Raw'], label='Z-axis', alpha=0.7, linewidth=1)
            ax1.set_xlabel('Time (seconds)', fontsize=11)
            ax1.set_ylabel('Acceleration (g)', fontsize=11)
            ax1.set_title('Accelerometer Time Series', fontsize=12, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # 2. Gyroscope time series
            ax2 = fig.add_subplot(gs[1, :])
            ax2.plot(time, window['Gx_Raw'], label='X-axis', alpha=0.7, linewidth=1)
            ax2.plot(time, window['Gy_Raw'], label='Y-axis', alpha=0.7, linewidth=1)
            ax2.plot(time, window['Gz_Raw'], label='Z-axis', alpha=0.7, linewidth=1)
            ax2.set_xlabel('Time (seconds)', fontsize=11)
            ax2.set_ylabel('Angular Velocity (°/s)', fontsize=11)
            ax2.set_title('Gyroscope Time Series', fontsize=12, fontweight='bold')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            # 3. FFT of Accelerometer X-axis
            ax3 = fig.add_subplot(gs[2, 0])
            fft_vals = np.abs(fft(window['Ax_Raw'].values))
            fft_freq = np.fft.fftfreq(len(window), 1/100)
            positive_idx = np.where(fft_freq > 0)
            ax3.plot(fft_freq[positive_idx], fft_vals[positive_idx], linewidth=1)
            ax3.set_xlabel('Frequency (Hz)', fontsize=10)
            ax3.set_ylabel('Magnitude', fontsize=10)
            ax3.set_title('FFT - Accelerometer X', fontsize=11, fontweight='bold')
            ax3.set_xlim(0, 50)
            ax3.grid(True, alpha=0.3)
            
            # 4. Vibration Magnitude
            ax4 = fig.add_subplot(gs[2, 1])
            vibration = np.sqrt(window['Ax_Raw']**2 + window['Ay_Raw']**2 + window['Az_Raw']**2)
            ax4.plot(time, vibration, color='red', linewidth=1)
            ax4.set_xlabel('Time (seconds)', fontsize=10)
            ax4.set_ylabel('Magnitude (g)', fontsize=10)
            ax4.set_title('Total Vibration Magnitude', fontsize=11, fontweight='bold')
            ax4.grid(True, alpha=0.3)
            
            # 5. Temperature
            ax5 = fig.add_subplot(gs[2, 2])
            ax5.plot(time, window['Temp'], color='orange', linewidth=1)
            ax5.set_xlabel('Time (seconds)', fontsize=10)
            ax5.set_ylabel('Temperature (°C)', fontsize=10)
            ax5.set_title('Motor Temperature', fontsize=11, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # 6. Prediction Confidence Bar Chart
            ax6 = fig.add_subplot(gs[3, :2])
            classes = list(all_probs.keys())
            probs = list(all_probs.values())
            colors = ['green' if c == fault_type else 'lightblue' for c in classes]
            bars = ax6.barh(classes, probs, color=colors, edgecolor='black', linewidth=1.5)
            ax6.set_xlabel('Confidence (%)', fontsize=11)
            ax6.set_title('Classification Probabilities', fontsize=12, fontweight='bold')
            ax6.set_xlim(0, 100)
            
            # Add percentage labels on bars
            for bar, prob in zip(bars, probs):
                ax6.text(prob + 1, bar.get_y() + bar.get_height()/2, 
                        f'{prob:.1f}%', va='center', fontsize=10, fontweight='bold')
            ax6.grid(True, alpha=0.3, axis='x')
            
            # 7. Prediction Result Box
            ax7 = fig.add_subplot(gs[3, 2])
            ax7.axis('off')
            
            # Determine if correct
            is_correct = fault_type.lower() == true_label.lower()
            result_color = 'green' if is_correct else 'red'
            result_text = 'CORRECT' if is_correct else 'INCORRECT'
            
            # Create result box
            box_text = f"""
PREDICTION RESULT

Detected Fault:
{fault_type.upper()}

Confidence:
{confidence:.2f}%

Status:
{result_text}
            """
            
            ax7.text(0.5, 0.5, box_text, 
                    ha='center', va='center',
                    fontsize=13, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=1', facecolor=result_color, alpha=0.3, edgecolor=result_color, linewidth=3))
            
            # IMPORTANT: Save BEFORE showing
            output_filename = f"fault_detection_{true_label}.png"
            plt.savefig(output_filename, dpi=300, bbox_inches='tight')
            print(f"✓ Visualization saved as: {output_filename}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        except Exception as e:
            print(f"✗ Error processing {csv_file}: {e}")
            import traceback
            traceback.print_exc()


# ============================================================================
# MAIN VISUALIZATION SCRIPT
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOTOR FAULT DETECTION - VISUALIZATION")
    print("="*60 + "\n")
    
    # Initialize visualizer
    visualizer = FaultDetectionVisualizer('motor_fault_detector.pkl', 'motor_features.csv')
    
    # Test files to visualize
    test_cases = [
        ('motor_data/motor_healthy_trial1.csv', 'healthy'),
        ('motor_data/motor_imbalance_trial1.csv', 'imbalance'),
        ('motor_data/motor_misalignment_trial1.csv', 'misalignment'),
        ('motor_data/motor_bearing_fault_trial1.csv', 'bearing_fault'),
    ]
    
    # Generate visualizations for each test case
    print("Generating fault detection visualizations...")
    for csv_file, true_label in test_cases:
        visualizer.visualize_fault_detection(csv_file, true_label)
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print("\nGenerated files:")
    print("  ✓ fault_detection_healthy.png")
    print("  ✓ fault_detection_imbalance.png")
    print("  ✓ fault_detection_misalignment.png")
    print("  ✓ fault_detection_bearing_fault.png")
    print("\nUse these in your presentation to show fault detection!")
    print("="*60 + "\n")