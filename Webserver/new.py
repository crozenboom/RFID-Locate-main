import sllurp.llrp as llrp
from sllurp.llrp import LLRPReaderClient
import csv
import time
from datetime import datetime
import socket
import sys
import re
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Reader settings
READER_IP = "192.168.0.219"
PORT = 5084
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
        epcs = [epc.strip() for epc in user_input.split(',')]
        for epc in epcs:
            if not re.match(r'^[0-9a-fA-F]+$', epc):
                print(f"Invalid EPC format: {epc}. EPCs must be hexadecimal strings (e.g., 30340212e43c789b0223addd).")
                break
        else:
            return epcs

class TagReportHandler:
    def __init__(self, epc_filter, all_reads, total_rows):
        self.epc_filter = epc_filter
        self.all_reads = all_reads
        self.total_rows = total_rows

    def handle_tag_report(self, reader, tag_reports):
        """Handle tag reports received from the reader."""
        logger.debug(f"Received {len(tag_reports)} tag reports: {tag_reports}")
        run_timestamp = datetime.now().isoformat()
        for tag in tag_reports:
            epc = tag.get('EPC', b'').hex() if isinstance(tag.get('EPC'), bytes) else tag.get('EPC', 'Unknown')
            
            # Apply EPC filter
            if self.epc_filter and epc != 'Unknown' and epc not in self.epc_filter:
                continue

            antenna = tag.get('AntennaID', 'Unknown')
            rssi = tag.get('PeakRSSI', 'Unknown')
            if rssi != 'Unknown':
                rssi = f"{rssi:.2f}" if isinstance(rssi, (int, float)) else rssi
            phase_angle = tag.get('ImpinjRFPhaseAngle', 'Unknown')
            if phase_angle != 'Unknown':
                phase_angle = f"{phase_angle:.2f}" if isinstance(phase_angle, (int, float)) else phase_angle
            frequency = tag.get('ChannelIndex', 'Unknown')
            doppler_frequency = tag.get('RFDopplerFrequency', 'Unknown')
            if doppler_frequency != 'Unknown':
                doppler_frequency = f"{doppler_frequency:.2f}" if isinstance(doppler_frequency, (int, float)) else doppler_frequency
            timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
            read_count = tag.get('TagSeenCount', 1)

            for _ in range(read_count):
                self.all_reads.append({
                    'RunTimestamp': run_timestamp,
                    'Timestamp': timestamp,
                    'EPC': epc,
                    'Antenna': str(antenna),
                    'RSSI': rssi,
                    'PhaseAngle': phase_angle,
                    'Frequency': frequency,
                    'DopplerFrequency': doppler_frequency
                })
                self.total_rows[0] += 1

def run_inventory():
    # Check reader connection
    if not check_reader_connection():
        print("Exiting due to connection failure.")
        sys.exit(1)

    # Get total runtime from user
    total_time = get_user_input()
    
    # Get EPC filter from user
    epc_filter = get_epc_filter()
    
    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"
    
    # Antennas to use (Impinj R420 has 4 ports: 1, 2, 3, 4)
    antennas = [1, 2, 3, 4]
    total_rows = [0]  # Use a list to allow modification in the handler
    all_reads = []  # Accumulate all reads here
    
    # Create a tag report handler
    handler = TagReportHandler(epc_filter, all_reads, total_rows)
    
    # Connect to the reader, specifying all antennas
    print("Connecting to reader...")
    reader = LLRPReaderClient(READER_IP, PORT, antennas)
    reader.add_tag_report_callback(handler.handle_tag_report)
    
    try:
        reader.connect()
    except Exception as e:
        print(f"Failed to connect to reader: {e}")
        sys.exit(1)
    
    # Start time for tracking total duration
    start_time = time.time()
    
    print("Reading...")
    
    # Run inventory for the specified total time
    try:
        reader.start_inventory()
        time.sleep(total_time)  # Run inventory for the full duration
        reader.stop_inventory()
    
    except Exception as e:
        print(f"Error during inventory: {e}")
    
    finally:
        # Cleanly disconnect from the reader
        print("Disconnecting from reader...")
        reader.disconnect()
    
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
    
    print(f"Success: {total_rows[0]} reads written to {csv_file}")

if __name__ == "__main__":
    run_inventory()