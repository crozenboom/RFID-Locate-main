import subprocess
import re
import csv
import time
from datetime import datetime
import socket

# Reader settings
READER_IP = "169.254.1.1"
PORT = 14150
CSV_FILE_BASE = "rfid_data"

def check_reader_connection(max_retries=3, retry_delay=2, timeout=10):
    """Check if the reader is connected by attempting a socket connection to the LLRP port."""
    for attempt in range(1, max_retries + 1):
        print(f"Attempting to connect to reader at {READER_IP}:{PORT} (Attempt {attempt}/{max_retries})...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((READER_IP, PORT))
            print("Connection successful: Reader at", READER_IP, "is reachable on port", PORT)
            return True
        except socket.timeout:
            print(f"Connection attempt timed out after {timeout} seconds.")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        except socket.error as e:
            print(f"Connection attempt failed. Details: {e}")
            if attempt < max_retries:
                print(f"Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
        finally:
            sock.close()
    
    print(f"Error: Failed to connect to reader at {READER_IP}:{PORT} after {max_retries} attempts.")
    return False        

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

def get_epc_filter():
    """Get the EPC(s) to filter for from the user."""
    while True:
        user_input = input("Enter EPC(s) to filter (comma-separated, or 'all' for no filter): ").strip().lower()
        if user_input == 'all':
            return None  # No filter, include all EPCs
        if not user_input:
            print("Please enter at least one EPC or 'all'.")
            continue
        # Split the input by commas and strip whitespace
        epcs = [epc.strip() for epc in user_input.split(',')]
        # Validate EPC format (hexadecimal string)
        for epc in epcs:
            if not re.match(r'^[0-9a-fA-F]+$', epc):
                print(f"Invalid EPC format: {epc}. EPCs must be hexadecimal strings (e.g., 30340212e43c789b0223addd).")
                break
        else:
            return epcs  # Return the list of EPCs if all are valid

def run_inventory():
    # Get total runtime from user
    total_time = get_user_input()
    
    # Get EPC filter from user
    epc_filter = get_epc_filter()
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"
    
    # Interval per antenna (in seconds)
    interval_per_antenna = 0.25  # 0.25 seconds per antenna
    
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
        cycle_start_time = time.time()  # Track start time of each cycle
        
        for antenna in antennas:
            # Check if total time has been exceeded
            if (time.time() - start_time) >= total_time:
                break
            
            # Construct command for the current antenna
            cmd = base_cmd.copy()
            cmd.extend(["-a", str(antenna)])
            
            try:
                # Run the sllurp command with increased timeout
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=interval_per_antenna + 0.75  # Increased to 1 second
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
                    
                    # Apply EPC filter
                    if epc_filter and epc != 'Unknown' and epc not in epc_filter:
                        continue  # Skip this tag if it doesn't match the filter
                    
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
                    
                    # Apply EPC filter
                    if epc_filter and epc != 'Unknown' and epc not in epc_filter:
                        continue  # Skip this tag if it doesn't match the filter
                    
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
            
            time.sleep(0.1)  # Add small delay to avoid overwhelming the reader
        
        # Ensure the cycle takes exactly 1 second
        cycle_time = time.time() - cycle_start_time
        if cycle_time < 1.0:
            time.sleep(1.0 - cycle_time)  # Sleep for the remaining time to make the cycle exactly 1 second
    
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