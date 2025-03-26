import subprocess
import re
import csv
from datetime import datetime
import sys
import socket
import time

# Reader settings (matching CLI)
READER_IP = "169.254.1.1"
PORT = 14150
CSV_FILE_BASE = "rfid_data"  # Base name for the CSV file

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

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"  # e.g., rfid_data_2025-03-26T12-32-12.csv
    
    # Command to run (matching the manual command, with added -P 30)
    cmd = [
        "sllurp", "inventory", READER_IP,
        "-p", str(PORT), "-a", "0", "-t", "5",  # All antennas, 5 seconds
        "-X", "0", "-P", "30", "--impinj-reports"
    ]
    
    total_rows = 0  # Count total rows written to CSV
    all_reads = []  # Accumulate all reads here
    
    # Print the command for debugging
    print(f"Running sllurp inventory command: {' '.join(cmd)}")
    
    try:
        # Run the sllurp inventory command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=15  # 5s inventory + 10s buffer
        )
        output = result.stdout
        
    except subprocess.TimeoutExpired as e:
        # Decode the bytes-like partial output to a string
        output = e.stdout.decode('utf-8') if e.stdout is not None else ""
        print(f"Command timed out after {e.timeout} seconds. Processing partial output...")
    
    except subprocess.CalledProcessError as e:
        print(f"Failed to run inventory: {e}\n{e.stderr}")
        return
    
    except Exception as e:
        print(f"Error: {e}")
        return
    
    # Parse tags from output (whether from successful run or partial output)
    all_tags = []
    tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
    for match in tag_pattern.finditer(output):
        tag_str = match.group(1)
        tag_dict = eval(tag_str, {"__builtins__": {}}, {})
        if isinstance(tag_dict, list):
            all_tags.extend(tag_dict)
    
    # Log number of tags seen
    tags_seen = len(all_tags)
    reads_total = sum(tag.get('TagSeenCount', 0) for tag in all_tags)
    print(f"Detected {tags_seen} tags with {reads_total} reads")
    
    # If no tags detected, log the raw output for debugging
    if tags_seen == 0:
        print(f"No tags detected. Raw output:\n{output if output else 'No output received from sllurp.'}")
        return
    
    # Process each tag and accumulate reads
    run_timestamp = datetime.now().isoformat()  # Timestamp of this run
    for tag in all_tags:
        epc = tag.get('EPC', 'Unknown')
        if isinstance(epc, bytes):  # Convert bytes to hex string
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
        
        # Add each read to the list
        for _ in range(read_count):
            all_reads.append({
                'RunTimestamp': run_timestamp,
                'Timestamp': timestamp,
                'EPC': epc,
                'Antenna': "0",  # All antennas, so we use 0
                'RSSI': rssi,
                'PhaseAngle': phase_angle,
                'Frequency': frequency,
                'DopplerFrequency': doppler_frequency
            })
            total_rows += 1
    
    # Write all accumulated reads to CSV
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = [
            'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
            'PhaseAngle', 'Frequency', 'DopplerFrequency'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for read in all_reads:
            writer.writerow(read)
    
    print(f"Inventory completed successfully. {total_rows} reads saved to {csv_file}")

if __name__ == "__main__":
    run_inventory()