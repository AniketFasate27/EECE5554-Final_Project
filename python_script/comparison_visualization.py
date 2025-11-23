"""
Side-by-side comparison of all fault types
Perfect for presentation slides
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft

sns.set_style("whitegrid")

def create_comparison_dashboard():
    """Create 4-panel comparison of all fault types"""
    
    test_files = {
        'Healthy': 'motor_data/motor_healthy_trial1.csv',
        'Imbalance': 'motor_data/motor_imbalance_trial1.csv',
        'Misalignment': 'motor_data/motor_misalignment_trial1.csv',
        'Bearing Fault': 'motor_data/motor_bearing_fault_trial1.csv'
    }
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('Motor Fault Detection - Comparative Analysis', fontsize=18, fontweight='bold', y=0.995)
    
    colors = {'Healthy': 'green', 'Imbalance': 'orange', 'Misalignment': 'red', 'Bearing Fault': 'purple'}
    
    for row, (label, csv_file) in enumerate(test_files.items()):
        print(f"Processing {label}...")
        
        # Load data
        data = pd.read_csv(csv_file)
        
        window_size = 1000
        start_idx = len(data) // 2 - window_size // 2
        window = data.iloc[start_idx:start_idx + window_size]
        time = np.arange(len(window)) / 100
        
        color = colors[label]
        
        # Column 1: Accelerometer
        ax = axes[row, 0]
        vibration = np.sqrt(window['Ax_Raw']**2 + window['Ay_Raw']**2 + window['Az_Raw']**2)
        ax.plot(time, vibration, color=color, linewidth=1)
        ax.set_ylabel(f'{label}\nVibration (g)', fontsize=10, fontweight='bold')
        if row == 0:
            ax.set_title('Vibration Magnitude', fontsize=12, fontweight='bold')
        if row == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Column 2: FFT
        ax = axes[row, 1]
        fft_vals = np.abs(fft(window['Ax_Raw'].values))
        fft_freq = np.fft.fftfreq(len(window), 1/100)
        positive_idx = np.where(fft_freq > 0)
        ax.plot(fft_freq[positive_idx], fft_vals[positive_idx], color=color, linewidth=1)
        ax.set_xlim(0, 50)
        if row == 0:
            ax.set_title('Frequency Spectrum', fontsize=12, fontweight='bold')
        if row == 3:
            ax.set_xlabel('Frequency (Hz)', fontsize=10)
        ax.set_ylabel('Magnitude', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Column 3: Gyroscope
        ax = axes[row, 2]
        gyro_mag = np.sqrt(window['Gx_Raw']**2 + window['Gy_Raw']**2 + window['Gz_Raw']**2)
        ax.plot(time, gyro_mag, color=color, linewidth=1)
        if row == 0:
            ax.set_title('Angular Velocity', fontsize=12, fontweight='bold')
        if row == 3:
            ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('°/s', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Column 4: Statistics Box
        ax = axes[row, 3]
        ax.axis('off')
        
        # Calculate peak frequency
        peak_freq = fft_freq[positive_idx][np.argmax(fft_vals[positive_idx])]
        
        stats_text = f"""
{label.upper()}

Vibration RMS:
{np.sqrt(np.mean(vibration**2)):.4f} g

Std Dev:
{np.std(vibration):.4f} g

Peak Freq:
{peak_freq:.2f} Hz

Temperature:
{np.mean(window['Temp']):.2f} °C
        """
        
        ax.text(0.5, 0.5, stats_text,
               ha='center', va='center', fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', facecolor=color, alpha=0.2, edgecolor=color, linewidth=2))
    
    plt.tight_layout()
    plt.savefig('fault_comparison_dashboard.png', dpi=300, bbox_inches='tight')
    print("\nComparison dashboard saved as: fault_comparison_dashboard.png")
    plt.show()

if __name__ == "__main__":
    print("\n" + "="*60)
    print("CREATING COMPARISON DASHBOARD")
    print("="*60 + "\n")
    create_comparison_dashboard()
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)