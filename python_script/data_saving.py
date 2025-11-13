"""
Real-Time Data Logger for Smart Motor Health Diagnostics
Reads serial data from ESP32 and saves to CSV file
"""

import serial
import csv
import datetime
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================
SERIAL_PORT = 'COM6'           # Change to your ESP32 port (COM3, COM4, etc.)
BAUD_RATE = 115200             # Must match ESP32 baud rate
OUTPUT_FOLDER = 'motor_data'   # Folder to save CSV files

# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{OUTPUT_FOLDER}/motor_data_{timestamp}.csv"
    
    print("=" * 60)
    print("Smart Motor Health Diagnostics - Data Logger")
    print("=" * 60)
    print(f"Serial Port: {SERIAL_PORT}")
    print(f"Baud Rate: {BAUD_RATE}")
    print(f"Output File: {filename}")
    print("=" * 60)
    print("Press Ctrl+C to stop logging")
    print("=" * 60)
    print()
    
    try:
        # Open serial connection
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print("✓ Serial connection established")
        
        # Create output folder if it doesn't exist
        import os
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Open CSV file for writing
        csv_file = open(filename, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        
        print("✓ CSV file created")
        print()
        print("Logging data... (Press Ctrl+C to stop)")
        print()
        
        line_count = 0
        header_found = False
        
        while True:
            # Read line from serial
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                
                # Skip empty lines
                if not line:
                    continue
                
                # Print debug/status messages (lines starting with [ or =)
                if line.startswith('[') or line.startswith('='):
                    print(f"[INFO] {line}")
                    continue
                
                # Check if this is the CSV header
                if 'Time,Ax_Raw' in line:
                    header_found = True
                    csv_writer.writerow(line.split(','))
                    print(f"✓ CSV Header: {line}")
                    continue
                
                # Save data lines (only after header is found)
                if header_found and ',' in line:
                    try:
                        # Split CSV line and write to file
                        data = line.split(',')
                        csv_writer.writerow(data)
                        csv_file.flush()  # Ensure data is written immediately
                        
                        line_count += 1
                        
                        # Print progress every 100 samples
                        if line_count % 100 == 0:
                            print(f"Logged {line_count} samples... (Latest: {data[0]} ms)")
                    
                    except Exception as e:
                        print(f"[WARNING] Error parsing line: {line}")
                        print(f"[WARNING] Error: {e}")
    
    except serial.SerialException as e:
        print(f"\n[ERROR] Serial connection failed: {e}")
        print(f"[ERROR] Make sure:")
        print(f"  1. ESP32 is connected to {SERIAL_PORT}")
        print(f"  2. Arduino IDE Serial Monitor is CLOSED")
        print(f"  3. No other program is using the port")
        sys.exit(1)
    
    except KeyboardInterrupt:
        print()
        print("=" * 60)
        print("Logging stopped by user (Ctrl+C)")
        print("=" * 60)
        print(f"Total samples logged: {line_count}")
        print(f"Data saved to: {filename}")
        print("=" * 60)
    
    finally:
        # Clean up
        if 'ser' in locals():
            ser.close()
        if 'csv_file' in locals():
            csv_file.close()
        print("\n✓ Files closed successfully")

if __name__ == "__main__":
    main()