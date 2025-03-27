import subprocess
import re
import csv
import time
from datetime import datetime

# Reader settings
READER_IP = "169.254.1.1"
PORT = 14150
CSV_FILE_BASE = "rfid_data"

def get_user_input():
    """Get the total runtime from the user in seconds."""
    while True:
        user_input = input("Enter total runtime in seconds (e.g., 10): ").strip()
        try:
            value = float(user_input)
            if value <= 0:
                print("Please enter a positive number.")
                continue
            return value
        except ValueError:
            print("Please enter a valid number.")

def run_inventory():
    # Get total runtime from user
    total_time = get_user_input()
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"
    
    # Interval per antenna (in seconds)
    interval_per_antenna = 0.25  # 0.25 seconds per antenna as specified
    
    # Base command (without antenna specification)
    base_cmd = [
        "sllurp", "inventory", READER_IP,
        "-p", str(PORT), "-X", "0", "-t", str(interval_per_antenna),
        "--impinj-reports"
    ]
    
    # Antennas to cycle through (Impinj R420 has 4 ports: 1, 2, 3, 4)
    antennas = [1, 2, 3, 4]
    total_rows = 0  # Count total rows written to CSV
    all_reads = []  # Accumulate all reads here
    
    # Start time for tracking total duration
    start_time = time.time()
    
    print("Reading...")
    
    # Run inventory for the specified total time
    while (time.time() - start_time) < total_time:
        for antenna in antennas:
            # Check if total time has been exceeded
            if (time.time() - start_time) >= total_time:
                break
            
            # Construct command for the current antenna
            cmd = base_cmd.copy()
            cmd.extend(["-a", str(antenna)])
            
            try:
                # Run the sllurp command
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=interval_per_antenna + 4.75  # 5 seconds timeout
                )
                output = result.stdout
                
                # Parse tags from output
                all_tags = []
                tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
                for match in tag_pattern.finditer(output):
                    tag_str = match.group(1)
                    tag_dict = eval(tag_str, {"__builtins__": {}}, {})
                    if isinstance(tag_dict, list):
                        all_tags.extend(tag_dict)
                
                # Process each tag and accumulate reads
                run_timestamp = datetime.now().isoformat()
                for tag in all_tags:
                    epc = tag.get('EPC', 'Unknown')
                    if isinstance(epc, bytes):
                        epc = epc.hex()
                    rssi = tag.get('ImpinjPeakRSSI', 'Unknown')
                    if rssi != 'Unknown':
                        rssi = f"{rssi / 100:.2f}"
                    phase_angle = tag.get('ImpinjRFPhaseAngle', 'Unknown')
                    if phase_angle != 'Unknown':
                        phase_angle = f"{phase_angle / 100:.2f}"
                    frequency = tag.get('ChannelIndex', 'Unknown')
                    doppler_frequency = tag.get('ImpinjRFDopplerFrequency', 'Unknown')
                    if doppler_frequency != 'Unknown':
                        doppler_frequency = f"{doppler_frequency / 100:.2f}"
                    timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                    read_count = tag.get('TagSeenCount', 0)
                    
                    for _ in range(read_count):
                        all_reads.append({
                            'RunTimestamp': run_timestamp,
                            'Timestamp': timestamp,
                            'EPC': epc,
                            'Antenna': str(antenna),
                            'RSSI': rssi,
                            'PhaseAngle': phase_angle,
                            'Frequency': frequency,
                            'DopplerFrequency': doppler_frequency
                        })
                        total_rows += 1
            
            except subprocess.TimeoutExpired as e:
                output = e.stdout if e.stdout else b""
                output = output.decode('utf-8') if isinstance(output, bytes) else output
                
                all_tags = []
                tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
                for match in tag_pattern.finditer(output):
                    tag_str = match.group(1)
                    tag_dict = eval(tag_str, {"__builtins__": {}}, {})
                    if isinstance(tag_dict, list):
                        all_tags.extend(tag_dict)
                
                run_timestamp = datetime.now().isoformat()
                for tag in all_tags:
                    epc = tag.get('EPC', 'Unknown')
                    if isinstance(epc, bytes):
                        epc = epc.hex()
                    rssi = tag.get('ImpinjPeakRSSI', 'Unknown')
                    if rssi != 'Unknown':
                        rssi = f"{rssi / 100:.2f}"
                    phase_angle = tag.get('ImpinjRFPhaseAngle', 'Unknown')
                    if phase_angle != 'Unknown':
                        phase_angle = f"{phase_angle / 100:.2f}"
                    frequency = tag.get('ChannelIndex', 'Unknown')
                    doppler_frequency = tag.get('ImpinjRFDopplerFrequency', 'Unknown')
                    if doppler_frequency != 'Unknown':
                        doppler_frequency = f"{doppler_frequency / 100:.2f}"
                    timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                    read_count = tag.get('TagSeenCount', 0)
                    
                    for _ in range(read_count):
                        all_reads.append({
                            'RunTimestamp': run_timestamp,
                            'Timestamp': timestamp,
                            'EPC': epc,
                            'Antenna': str(antenna),
                            'RSSI': rssi,
                            'PhaseAngle': phase_angle,
                            'Frequency': frequency,
                            'DopplerFrequency': doppler_frequency
                        })
                        total_rows += 1
            
            except subprocess.CalledProcessError as e:
                print(f"Failed to run inventory: {e}\n{e.stderr}")
                return
            
            time.sleep(0.1)  # Small delay to avoid overwhelming the reader
    
    # Write all accumulated reads to CSV after collection is complete
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
            'PhaseAngle', 'Frequency', 'DopplerFrequency'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for read in all_reads:
            writer.writerow(read)
    
    print(f"Success: {total_rows} reads written to {csv_file}")

if __name__ == "__main__":
    run_inventory()