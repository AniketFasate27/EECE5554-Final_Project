"""
Real-Time Data Logger for Smart Motor Health Diagnostics
Reads serial data from ESP32 and saves to CSV file
FIXED: Handles serial errors and queue overflow warnings
"""

import serial
import csv
import datetime
import sys
import time

# ============================================================================
# CONFIGURATION
# ============================================================================
SERIAL_PORT = 'COM6'           # Change to your ESP32 port
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
        time.sleep(2)  # Wait for connection to stabilize
        ser.reset_input_buffer()  # Clear any old data
        print("✓ Serial connection established")
        
        # Create output folder if it doesn't exist
        import os
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
        
        # Open CSV file for writing
        csv_file = open(filename, 'w', newline='', encoding='utf-8')
        csv_writer = csv.writer(csv_file)
        
        print("✓ CSV file created")
        print()
        print("Logging data... (Press Ctrl+C to stop)")
        print()
        
        line_count = 0
        error_count = 0
        header_found = False
        last_print_time = time.time()
        
        while True:
            try:
                # Read line from serial
                if ser.in_waiting > 0:
                    # Try to decode line, handle errors gracefully
                    try:
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                    except UnicodeDecodeError:
                        error_count += 1
                        if error_count % 10 == 0:
                            print(f"[WARNING] {error_count} decode errors (continuing...)")
                        continue
                    
                    # Skip empty lines
                    if not line:
                        continue
                    
                    # Print debug/status messages (lines starting with [ or =)
                    if line.startswith('[') or line.startswith('='):
                        # Only print important messages, not every queue warning
                        if '[ACQ]' in line or '[PROC] Data Processing' in line or '[OUT] Data Output' in line:
                            print(f"[INFO] {line}")
                        continue
                    
                    # Check if this is the CSV header
                    if 'Time,Ax_Raw' in line or 'Time,Accel_X' in line:
                        header_found = True
                        csv_writer.writerow(line.split(','))
                        csv_file.flush()
                        print(f"✓ CSV Header: {line}")
                        continue
                    
                    # Save data lines (only after header is found)
                    if header_found and ',' in line:
                        try:
                            # Validate line has correct number of columns
                            data = line.split(',')
                            
                            # Should have 14 columns
                            if len(data) == 14:
                                csv_writer.writerow(data)
                                
                                line_count += 1
                                
                                # Print progress every 100 samples (but max once per second)
                                current_time = time.time()
                                if line_count % 100 == 0 and (current_time - last_print_time) >= 1.0:
                                    elapsed_sec = int(data[0]) / 1000
                                    print(f"Logged {line_count} samples... ({elapsed_sec:.1f}s elapsed)")
                                    last_print_time = current_time
                                    csv_file.flush()  # Flush every 100 samples
                            else:
                                # Skip malformed lines
                                pass
                        
                        except Exception as e:
                            # Silently skip bad lines
                            error_count += 1
                            if error_count % 50 == 0:
                                print(f"[WARNING] {error_count} parsing errors")
            
            except KeyboardInterrupt:
                raise  # Re-raise to exit cleanly
            
            except Exception as e:
                # Catch any other errors and continue
                error_count += 1
                if error_count % 50 == 0:
                    print(f"[WARNING] General error: {e}")
                continue
    
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
        print(f"Errors encountered: {error_count}")
        print(f"Success rate: {100 * line_count / (line_count + error_count):.1f}%")
        print(f"Data saved to: {filename}")
        print("=" * 60)
    
    finally:
        # Clean up
        if 'ser' in locals() and ser.is_open:
            ser.close()
        if 'csv_file' in locals():
            csv_file.close()
        print("\n✓ Files closed successfully")

if __name__ == "__main__":
    main()