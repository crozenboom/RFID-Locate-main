import csv
import subprocess
import time
from datetime import datetime
import ast
import sys
import logging
import socket

# Configure logging (console with DEBUG level)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)

# Suppress sllurp.verb.inventory INFO logs to reduce verbose output
logging.getLogger('sllurp.verb.inventory').setLevel(logging.WARNING)

# Configuration
IP_ADDRESS = "192.168.0.219"  # Reader IP address
ANTENNAS = "1,2,3,4"  # Antennas to use
POWER = "90"  # Transmit power
SESSION = "0"  # Session 0 for single tag test
SEARCH_MODE = "2"  # SingleTag mode (as per your example)
MODE_IDENTIFIER = "0"  # High rate, low sensitivity
INTERVAL = 2  # Default cycle duration in seconds
MAX_RETRIES = 3  # Number of retries per cycle
TIMESTAMP_FIELD = 'LastSeenTimestampUTC'  # Fallback timestamp field

# Base sllurp command
COMMAND = [
    "sllurp", "inventory", IP_ADDRESS,
    "-a", ANTENNAS,
    "-X", POWER,
    "-s", SESSION,
    "--impinj-search-mode", SEARCH_MODE,
    "--mode-identifier", MODE_IDENTIFIER,
    "--impinj-extended-configuration",
    "--impinj-reports"
]

def init_csv(csv_file):
    """Initialize CSV file with headers."""
    with open(csv_file, 'w', newline='') as csv_file_handle:
        headers = ['timestamp', 'antenna', 'rssi', 'epc', 'phase_angle', 'channel_index', 'doppler_frequency']
        csv_writer = csv.writer(csv_file_handle)
        csv_writer.writerow(headers)
    logging.info(f"Initialized CSV: {csv_file}")

def parse_output(output, cycle_start_time, cycle_duration):
    """Parse sllurp inventory output and expand TagSeenCount into individual reads."""
    tags = []
    logging.debug(f"Raw output: {output}")
    start = output.find("[{")
    end = output.rfind("}]") + 2
    if start == -1 or end == -1:
        logging.warning("No tag data found in output")
        return tags
    
    tag_list_str = output[start:end]
    try:
        tag_list = ast.literal_eval(tag_list_str)
        for tag in tag_list:
            seen_count = tag.get('TagSeenCount', 1)
            # Distribute timestamps linearly within the cycle duration
            timestamp_interval = cycle_duration / seen_count if seen_count > 1 else 0
            for i in range(seen_count):
                read_timestamp = cycle_start_time + (i * timestamp_interval)
                tags.append({
                    'epc': tag.get('EPC-96', b'').decode('utf-8', errors='ignore'),
                    'antenna': tag.get('AntennaID', 0),
                    'rssi': tag.get('PeakRSSI', 0),
                    'phase_angle': tag.get('ImpinjRFPhaseAngle', None),
                    'channel_index': tag.get('ChannelIndex', None),
                    'doppler_frequency': tag.get('ImpinjRFDopplerFrequency', None),
                    'timestamp': read_timestamp
                })
        antenna_ids = set(tag['antenna'] for tag in tags)
        logging.info(f"Parsed {len(tags)} individual reads (antennas: {antenna_ids})")
    except (SyntaxError, ValueError) as e:
        logging.error(f"Failed to parse output: {e}")
    
    return tags

def run_inventory(cycle_duration):
    """Run the sllurp inventory command for one cycle and return parsed tag data."""
    # Check LLRP port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex((IP_ADDRESS, 5084))
        sock.close()
        if result != 0:
            logging.error("LLRP port 5084 not reachable")
            time.sleep(2)
            return []
    except Exception as e:
        logging.error(f"Socket check failed: {e}")
        time.sleep(2)
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Running command (attempt {attempt+1}): {' '.join(COMMAND)}")
            process = subprocess.Popen(
                COMMAND + ["-t", str(cycle_duration)],  # Add duration argument
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            cycle_start_time = time.time()
            stdout, stderr = process.communicate(timeout=cycle_duration + 2)  # Allow extra time for process to finish
            
            if process.returncode != 0 and stderr:
                logging.error(f"Command error: {stderr}")
                continue
            
            tags = parse_output(stdout, cycle_start_time, cycle_duration)
            if not tags:
                logging.info("No tags detected in this cycle")
            return tags
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            logging.warning("Process killed after timeout")
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    logging.error("All attempts failed")
    return []

def log_tags(tags, csv_file, epc_filter=None):
    """Log tag data to CSV, filtering by EPC if specified."""
    with open(csv_file, 'a', newline='') as csv_file_handle:
        csv_writer = csv.writer(csv_file_handle)
        filtered_count = 0
        for tag in tags:
            if epc_filter is None or tag['epc'] == epc_filter:
                timestamp = datetime.fromtimestamp(tag['timestamp']).isoformat()
                csv_writer.writerow([
                    timestamp,
                    tag['antenna'],
                    tag['rssi'],
                    tag['epc'],
                    tag['phase_angle'],
                    tag['channel_index'],
                    tag['doppler_frequency']
                ])
                filtered_count += 1
        if filtered_count > 0:
            logging.info(f"Logged {filtered_count} tags to CSV (EPC filter: {epc_filter or 'None'})")
        else:
            logging.info(f"No tags logged to CSV (EPC filter: {epc_filter or 'None'})")

def main():
    """Main function to handle user input and run inventory cycles."""
    # Generate unique CSV filename based on current timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"rfid_tags_{timestamp_str}.csv"
    
    # Get cycle duration
    print("Enter cycle duration in seconds (default is 2):")
    cycle_input = input().strip()
    try:
        cycle_duration = float(cycle_input) if cycle_input else INTERVAL
        if cycle_duration <= 0:
            raise ValueError("Cycle duration must be positive")
    except ValueError:
        print("Invalid input. Using default cycle duration of 2 seconds.")
        logging.error("Invalid cycle duration input; using default")
        cycle_duration = INTERVAL
    
    # Get total runtime
    print("Enter total runtime in seconds (or press Enter to run indefinitely):")
    runtime_input = input().strip()
    start_time = time.time()
    if runtime_input:
        try:
            runtime = float(runtime_input)
            if runtime <= 0:
                print("Runtime must be positive.")
                logging.error("Invalid runtime: must be positive")
                return
            end_time = start_time + runtime
            print(f"Running for {runtime} seconds with {cycle_duration}-second cycles...")
            logging.info(f"Starting inventory for {runtime} seconds")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter.")
            logging.error("Invalid runtime input")
            return
    else:
        end_time = None
        print("Running indefinitely (press Ctrl+C to stop)...")
        logging.info("Starting inventory indefinitely")
    
    # Prompt for EPC filter
    print("Enter EPC to filter (24-character hex, e.g., 303435fc8803d056d15b963f, or press Enter for no filter):")
    epc_filter = input().strip()
    if epc_filter:
        if len(epc_filter) == 24 and all(c in '0123456789abcdefABCDEF' for c in epc_filter):
            logging.info(f"Applying EPC filter: {epc_filter}")
        else:
            logging.warning(f"Invalid EPC filter '{epc_filter}' (must be 24-character hex). No filter applied.")
            epc_filter = None
    else:
        logging.info("No EPC filter applied")
        epc_filter = None
    
    init_csv(csv_file)
    
    cycle_count = 0
    try:
        while True:
            cycle_count += 1
            logging.info(f"Starting cycle {cycle_count}")
            if end_time and time.time() >= end_time:
                logging.info("Runtime expired")
                break
            
            tags = run_inventory(cycle_duration)
            if tags:
                log_tags(tags, csv_file, epc_filter)
            
            # No sleep needed; cycle_duration controls the command runtime
            elapsed = time.time() - start_time
            if end_time:
                remaining = end_time - time.time()
                logging.debug(f"Elapsed: {elapsed:.2f}s, Remaining: {remaining:.2f}s")
    
    except KeyboardInterrupt:
        print("\nStopping inventory...")
        logging.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Unexpected error: {e}")
    finally:
        print("Inventory stopped.")
        logging.info(f"Inventory stopped. Total cycles: {cycle_count}")

if __name__ == '__main__':
    main()