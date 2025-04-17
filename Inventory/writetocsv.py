from flask import Flask, request # type: ignore
import urllib.parse
import csv
from datetime import datetime
import threading
import time
import os
import signal
import sys

app = Flask(__name__)

# Global list to store all tag reads during the run
all_tag_reads = []

def signal_handler(sig, frame):
    """Handle Ctrl+C to save data before exiting."""
    print("\nInterrupted! Saving data to CSV...")
    write_csv(epc_filter)
    sys.exit(0)

@app.route('/rfid', methods=['POST'])
def receive_data():
    # Get raw data as text
    raw_data = request.get_data(as_text=True)
    
    # Parse URL-encoded data into a dictionary
    parsed_data = urllib.parse.parse_qs(raw_data)
    
    # Extract reader metadata, stripping quotes
    reader_name = parsed_data.get('reader_name', [''])[0].strip('"')
    mac_address = parsed_data.get('mac_address', [''])[0].strip('"')
    
    # Extract field names and values
    field_names = parsed_data.get('field_names', [''])[0].split(',')
    field_values = parsed_data.get('field_values', [''])[0].split('\n')
    
    # Process each tag read
    for value_line in field_values:
        if value_line.strip():  # Skip empty lines
            values = [v.strip('"') for v in value_line.split(',')]  # Strip quotes from each value
            tag_read = dict(zip(field_names, values))
            # Add reader metadata and timestamp to the tag read
            tag_read['reader_name'] = reader_name
            tag_read['mac_address'] = mac_address
            tag_read['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            all_tag_reads.append(tag_read)
    
    # Print parsed data to console
    print(f"Reader: {reader_name}, MAC: {mac_address}")
    for value_line in field_values:
        if value_line.strip():
            values = [v.strip('"') for v in value_line.split(',')]
            tag_read = dict(zip(field_names, values))
            print(f"Tag Read: {tag_read}")
    
    return "OK", 200

def write_csv(epc_filter=None):
    if not all_tag_reads:
        print("No tag reads collected, skipping CSV creation.")
        return
    
    # Filter tag reads by EPC if a filter is provided
    filtered_reads = all_tag_reads
    if epc_filter:
        epc_filter = epc_filter.lower()
        filtered_reads = [
            read for read in all_tag_reads
            if 'epc' in read and epc_filter in read['epc'].lower()
        ]
    
    if not filtered_reads:
        print("No tag reads match the EPC filter, skipping CSV creation.")
        return
    
    # Generate single CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"rfid_data_{timestamp}.csv"
    
    # Get all possible field names from filtered data
    field_names = set()
    for read in filtered_reads:
        field_names.update(read.keys())
    field_names = sorted(field_names)  # Sort for consistent column order
    
    # Write to CSV
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for read in filtered_reads:
            writer.writerow(read)
    
    print(f"Data written to {csv_filename}")

def shutdown_server(duration, epc_filter):
    """Stop the Flask server after the specified duration (in seconds)."""
    time.sleep(duration)
    print(f"Run duration of {duration} seconds completed. Stopping server...")
    write_csv(epc_filter)
    # Graceful shutdown
    os._exit(0)

if __name__ == '__main__':
    # Register Ctrl+C handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Prompt user for run duration
    while True:
        duration_input = input("Enter run duration in seconds (leave blank to run indefinitely): ").strip()
        if not duration_input:
            run_duration = None
            break
        try:
            run_duration = int(duration_input)
            if run_duration > 0:
                break
            print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number or leave blank.")
    
    # Prompt user for EPC filter
    epc_filter = input("Enter EPC filter (leave blank for no filter): ").strip()
    if not epc_filter:
        epc_filter = None
    
    # Start the shutdown timer only if a duration was provided
    if run_duration is not None:
        threading.Thread(target=shutdown_server, args=(run_duration, epc_filter), daemon=True).start()
    
    # Run the Flask app
    app.run(host="0.0.0.0", port=5050)