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
ANTENNAS = "1"  # Antennas to use
POWER = "90"  # Transmit power
SESSION = "2"           # 0 - for single tag test
                        # 1 - for one tag at a time
                        # 2 - for tag dense inventory
                        # 3 - for many new tags
SEARCH_MODE = "1"  # Single or Dual
MODE_IDENTIFIER = "2"   # 0 - high rate, low sensitivity
                        # 1 - moderate rate, good sensitivity
                        # 2 - low rate, high sensitivity
                        # 3 - very low rate, very high sensitivity
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

def parse_output(output):
    """Parse sllurp inventory output to extract tag data."""
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
            tags.append({
                'epc': tag.get('EPC-96', b'').decode('utf-8', errors='ignore'),
                'antenna': tag.get('AntennaID', 0),
                'rssi': tag.get('PeakRSSI', 0),
                'phase_angle': tag.get('ImpinjRFPhaseAngle', None),
                'channel_index': tag.get('ChannelIndex', None),
                'doppler_frequency': tag.get('ImpinjRFDopplerFrequency', None),
                'last_seen_timestamp': tag.get(TIMESTAMP_FIELD, 0)
            })
        antenna_ids = set(tag['antenna'] for tag in tags)
        if len(antenna_ids) < 4:
            logging.warning(f"Incomplete antenna cycle: {antenna_ids}")
        logging.info(f"Parsed {len(tags)} tags")
    except (SyntaxError, ValueError) as e:
        logging.error(f"Failed to parse output: {e}")
    
    return tags

def run_inventory():
    """Run the sllurp inventory command and return parsed tag data."""
    # Check LLRP port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('192.168.0.219', 5084))
        sock.close()
        if result != 0:
            logging.error("LLRP port 5084 not reachable")
            time.sleep(2)  # Wait before retrying to allow reader recovery
            return []
    except Exception as e:
        logging.error(f"Socket check failed: {e}")
        time.sleep(2)  # Wait before retrying
        return []
    
    for attempt in range(MAX_RETRIES):
        try:
            logging.info(f"Running command (attempt {attempt+1}): {' '.join(COMMAND)}")
            process = subprocess.Popen(
                COMMAND,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            time.sleep(INTERVAL)
            
            process.terminate()
            try:
                stdout, stderr = process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                stdout, stderr = process.communicate()
                logging.warning("Process killed after failing to terminate")
            
            if process.returncode != 0 and stderr:
                logging.error(f"Command error: {stderr}")
                continue
            
            tags = parse_output(stdout)
            if not tags:
                logging.info("No tags detected in this cycle")
            return tags
        except Exception as e:
            logging.error(f"Attempt {attempt+1} failed: {e}")
            time.sleep(1)
    logging.error("All attempts failed")
    return []

def log_tags(tags, csv_file, epc_filter=None):
    """Log tag data to CSV, filtering by EPC if specified."""
    #antenna_ids = set(tag['antenna'] for tag in tags)
    #if len(antenna_ids) < 4:
    #    logging.warning(f"Skipping incomplete cycle with antennas: {antenna_ids}")
    #    return
    with open(csv_file, 'a', newline='') as csv_file_handle:
        csv_writer = csv.writer(csv_file_handle)
        filtered_count = 0
        for tag in tags:
            if epc_filter is None or tag['epc'] == epc_filter:
                timestamp_us = tag.get('last_seen_timestamp', 0)
                timestamp = datetime.fromtimestamp(timestamp_us / 1e6).isoformat() if timestamp_us else datetime.now().isoformat()
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
    """Main function to handle user input and run inventory."""
    # Generate unique CSV filename based on current timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"rfid_tags_{timestamp_str}.csv"
    
    print("Enter runtime in seconds (or press Enter to run indefinitely):")
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
            print(f"Running for {runtime} seconds...")
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
        # Validate EPC: 24-character hexadecimal
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
            
            cycle_start = time.time()
            tags = run_inventory()
            if tags:
                log_tags(tags, csv_file, epc_filter)
            
            elapsed = time.time() - cycle_start
            sleep_time = max(0, INTERVAL - elapsed)
            time.sleep(sleep_time)
    
    except KeyboardInterrupt:
        print("\nStopping inventory...")
        logging.info("Stopped by user (Ctrl+C)")
    except Exception as e:
        print(f"Error: {e}")
        logging.error(f"Unexpected error: {e}")
    finally:
        print("Inventory stopped.")
        logging.info("Inventory stopped")

if __name__ == '__main__':
    main()