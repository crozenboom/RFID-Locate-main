import csv
import subprocess
import time
from datetime import datetime, timezone
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
COMMAND = ["sllurp", "inventory", "192.168.0.219", "-a", "0", "-X", "2000", "--impinj-reports"]
INTERVAL = 2.0  # Time between CSV writes
MAX_RETRIES = 3
TIMESTAMP_FIELD = 'LastSeenTimestampUTC'  # Options: 'LastSeenTimestampUTC', 'LastSeenTimestampUptime'

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

def log_tags(tags, csv_file, epc_filter=None):
    """Log tag data to CSV, filtering by EPC if specified."""
    antenna_ids = set(tag['antenna'] for tag in tags)
    if len(antenna_ids) < 4:
        logging.warning(f"Skipping incomplete cycle with antennas: {antenna_ids}")
        return
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

def run_inventory(csv_file, runtime, epc_filter):
    """Run continuous sllurp inventory, logging tags to CSV."""
    # Check LLRP port
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('192.168.0.219', 5084))
        sock.close()
        if result != 0:
            logging.error("LLRP port 5084 not reachable")
            return
    except Exception as e:
        logging.error(f"Socket check failed: {e}")
        return
    
    start_time = time.time()
    end_time = start_time + runtime if runtime else None
    next_log_time = start_time + INTERVAL
    tag_buffer = []

    try:
        logging.info(f"Starting continuous inventory: {' '.join(COMMAND)}")
        process = subprocess.Popen(
            COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1  # Line-buffered
        )
        
        while True:
            # Check runtime
            current_time = time.time()
            if end_time and current_time >= end_time:
                logging.info("Runtime expired")
                break
            
            # Read output line-by-line
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                logging.error("Sllurp process ended unexpectedly")
                break
            
            if line:
                tags = parse_output(line)
                if tags:
                    tag_buffer.extend(tags)
            
            # Log tags to CSV at intervals
            if current_time >= next_log_time:
                if tag_buffer:
                    log_tags(tag_buffer, csv_file, epc_filter)
                    tag_buffer = []  # Clear buffer after logging
                next_log_time += INTERVAL
            
            # Check for errors
            if process.poll() is not None:
                stderr = process.stderr.read()
                if stderr:
                    logging.error(f"Command error: {stderr}")
                break
        
        # Polite stop
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=5)
            if stdout:
                tags = parse_output(stdout)
                if tags:
                    log_tags(tags, csv_file, epc_filter)
            if stderr:
                logging.error(f"Command error: {stderr}")
        except subprocess.TimeoutExpired:
            process.kill()
            logging.warning("Process killed after failing to terminate")
    
    except KeyboardInterrupt:
        logging.info("Stopped by user (Ctrl+C)")
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            logging.warning("Process killed after failing to terminate")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        process.kill()

def main():
    """Main function to handle user input and run inventory."""
    # Generate unique CSV filename based on current timestamp
    timestamp_str = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"rfid_tags_{timestamp_str}.csv"
    
    print("Enter runtime in seconds (or press Enter to run indefinitely):")
    runtime_input = input().strip()
    
    runtime = None
    if runtime_input:
        try:
            runtime = float(runtime_input)
            if runtime <= 0:
                print("Runtime must be positive.")
                logging.error("Invalid runtime: must be positive")
                return
            print(f"Running for {runtime} seconds...")
            logging.info(f"Starting inventory for {runtime} seconds")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter.")
            logging.error("Invalid runtime input")
            return
    else:
        print("Running indefinitely (press Ctrl+C to stop)...")
        logging.info("Starting inventory indefinitely")
    
    # Prompt for EPC filter
    print("Enter EPC to filter (24-character hex, e.g., 303435fc8803d056d15b963f, or press Enter for no filter):")
    epc_filter = input().strip()
    if epc_filter:
        # Validate EPC: 24-character hexadecimal
        if len(epc_filter) == 24 and all(c in '0123456789abcdefABCDEF' for c in epc_filter):
            logging.info 