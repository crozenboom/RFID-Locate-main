import subprocess
import re
import csv
from datetime import datetime, time
import sys
import socket

# Reader settings (matching CLI)
READER_IP = "192.168.0.219"
PORT = 5084
CSV_FILE_BASE = "rfid_data"  # Base name for the CSV file

def get_user_input(prompt, default=None, type_cast=str, validator=None):
    """Helper function to get and validate user input."""
    while True:
        user_input = input(f"{prompt} [{'Enter to skip' if default is None else default}]: ").strip()
        if not user_input:  # If user presses Enter, use default
            return default
        try:
            value = type_cast(user_input)
            if validator and not validator(value):
                print(f"Invalid input: {value}. Please try again.")
                continue
            return value
        except ValueError:
            print(f"Invalid input: {user_input}. Please enter a valid {type_cast.__name__}.")

def check_reader_connection(max_retries=3, retry_delay=2, timeout=5):
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

def run_inventory():
    # Check reader connection before proceeding
    if not check_reader_connection():
        print("Exiting due to connection failure.")
        sys.exit(1)

    # Get total runtime from user
    total_time = get_user_input("Total inventory duration (seconds)", 10, float, lambda x: x > 0)
    
    # Get EPC filter from user (comma-separated list or press Enter to skip)
    def validate_epc_list(epc_str):
        if not epc_str:
            return True  # Allow empty input (will be handled by default)
        epcs = [epc.strip() for epc in epc_str.split(',')]
        for epc in epcs:
            if not re.match(r'^[0-9a-fA-F]+$', epc):
                print(f"Invalid EPC format: {epc}. EPCs must be hexadecimal strings (e.g., 30340212e43c789b0223addd).")
                return False
        return True

    epc_input = get_user_input("EPC(s) to filter", None, str, validate_epc_list)
    epc_filter = None if epc_input is None else [epc.strip() for epc in epc_input.split(',')]
    
    # Generate timestamp for the filename (same as RunTimestamp but filesystem-friendly)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"  # e.g., rfid_data_2025-03-21T14-20-00.csv
    
    # Interval per antenna (in seconds)
    interval_per_antenna = 29  # 29 seconds
    
    # Base command (without antenna specification)
    base_cmd = [
        "sllurp", "inventory", READER_IP,
        "-p", str(PORT), "-X", "0", "-t", str(interval_per_antenna), 
        "--impinj-reports"
    ]
    
    # Antennas to cycle through (Impinj R420 has 4 ports)
    antennas = [1, 2, 3, 4]
    total_rows = 0  # Count total rows written to CSV
    time_elapsed = 0  # Track total time elapsed
    all_reads = []  # Accumulate all reads here
    
    print("Reading...", end="", flush=True)
    try:
        while time_elapsed < total_time:
            for antenna in antennas:
                if time_elapsed >= total_time:
                    break
                
                # Construct command for the current antenna
                cmd = base_cmd.copy()
                cmd.extend(["-a", str(antenna)])
                
                print(f"\rReading... (Antenna {antenna})", end="", flush=True)
                
                try:
                    # Run CLI command with timeout (0.25s inventory)
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=interval_per_antenna + 1
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
                    
                    # Log number of tags seen for this antenna
                    tags_seen = len(all_tags)
                    reads_for_antenna = sum(tag.get('TagSeenCount', 0) for tag in all_tags)
                    print(f"\rReading... (Antenna {antenna}: {tags_seen} tags, {reads_for_antenna} reads)", end="", flush=True)
                    
                    # Process each tag and accumulate reads
                    run_timestamp = datetime.now().isoformat()  # Timestamp of this run
                    for tag in all_tags:
                        epc = tag.get('EPC', 'Unknown')
                        if isinstance(epc, bytes):  # Convert bytes to hex string
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
                        
                        # Add each read to the list
                        for _ in range(read_count):
                            all_reads.append({
                                'RunTimestamp': run_timestamp,
                                'Timestamp': timestamp,
                                'EPC': epc,
                                'AntennaPort': str(antenna),  # Use the active antenna port
                                'RSSI': rssi,
                                'PhaseAngle': phase_angle,
                                'Frequency': frequency,
                                'DopplerFrequency': doppler_frequency
                            })
                            total_rows += 1

                except subprocess.TimeoutExpired as e:
                    output = e.stdout
                    if output is None:
                        output = ""
                    else:
                        output = output.decode('utf-8')  # Convert bytes to string
                    
                    # Parse tags from output
                    all_tags = []
                    tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
                    for match in tag_pattern.finditer(output):
                        tag_str = match.group(1)
                        tag_dict = eval(tag_str, {"__builtins__": {}}, {})
                        if isinstance(tag_dict, list):
                            all_tags.extend(tag_dict)

                    # Log number of tags seen for this antenna
                    tags_seen = len(all_tags)
                    reads_for_antenna = sum(tag.get('TagSeenCount', 0) for tag in all_tags)
                    print(f"\rReading... (Antenna {antenna}: {tags_seen} tags, {reads_for_antenna} reads)", end="", flush=True)
                    
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
                                'AntennaPort': str(antenna),  # Use the active antenna port
                                'RSSI': rssi,
                                'PhaseAngle': phase_angle,
                                'Frequency': frequency,
                                'DopplerFrequency': doppler_frequency
                            })
                            total_rows += 1
                    
                except subprocess.CalledProcessError as e:
                    print("\r" + " " * 40 + "\r", end="", flush=True)
                    print(f"Failed to run inventory on antenna {antenna}: {e}\n{e.stderr}")
                    return
                
                time_elapsed += interval_per_antenna  # Increment time after each interval
        
        # Clear "Reading..." message
        print("\r" + " " * 40 + "\r", end="", flush=True)
        
        # Write all accumulated reads to CSV at the end
        with open(csv_file, 'w', newline='') as csvfile:
            fieldnames = [
                'RunTimestamp', 'Timestamp', 'EPC', 'AntennaPort', 'RSSI',
                'PhaseAngle', 'Frequency', 'DopplerFrequency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for read in all_reads:
                writer.writerow(read)
        
        print(f"Inventory completed successfully. {total_rows} reads saved to {csv_file}")
    
    except Exception as e:
        print("\r" + " " * 40 + "\r", end="", flush=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    run_inventory()