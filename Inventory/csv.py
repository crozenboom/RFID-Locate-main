from flask import Flask, request # type: ignore
import urllib.parse
import csv
from datetime import datetime
import threading
import time
import os

app = Flask(__name__)

# Global list to store all tag reads during the run
all_tag_reads = []
# Duration to run the server (in seconds)
RUN_DURATION = 60  # Set your desired duration here

@app.route('/rfid', methods=['POST'])
def receive_data():
    # Get raw data as text
    raw_data = request.get_data(as_text=True)
    
    # Parse URL-encoded data into a dictionary
    parsed_data = urllib.parse.parse_qs(raw_data)
    
    # Extract reader metadata
    reader_name = parsed_data.get('reader_name', [''])[0]
    mac_address = parsed_data.get('mac_address', [''])[0]
    
    # Extract field names and values
    field_names = parsed_data.get('field_names', [''])[0].split(',')
    field_values = parsed_data.get('field_values', [''])[0].split('\n')
    
    # Process each tag read
    for value_line in field_values:
        if value_line.strip():  # Skip empty lines
            values = value_line.split(',')
            tag_read = dict(zip(field_names, values))
            # Add reader metadata to the tag read
            tag_read['reader_name'] = reader_name
            tag_read['mac_address'] = mac_address
            all_tag_reads.append(tag_read)
    
    # Print parsed data to console
    print(f"Reader: {reader_name}, MAC: {mac_address}")
    for value_line in field_values:
        if value_line.strip():
            values = value_line.split(',')
            tag_read = dict(zip(field_names, values))
            print(f"Tag Read: {tag_read}")
    
    return "OK", 200

def write_csv():
    if not all_tag_reads:
        print("No tag reads collected, skipping CSV creation.")
        return
    
    # Generate single CSV file with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"rfid_data_{timestamp}.csv"
    
    # Get all possible field names from collected data
    field_names = set()
    for read in all_tag_reads:
        field_names.update(read.keys())
    field_names = sorted(field_names)  # Sort for consistent column order
    
    # Write to CSV
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=field_names)
        writer.writeheader()
        for read in all_tag_reads:
            writer.writerow(read)
    
    print(f"Data written to {csv_filename}")

def shutdown_server():
    """Stop the Flask server after RUN_DURATION seconds."""
    time.sleep(RUN_DURATION)
    print(f"Run duration of {RUN_DURATION} seconds completed. Stopping server...")
    write_csv()
    # Graceful shutdown (works in most environments)
    os._exit(0)

if __name__ == '__main__':
    # Start the shutdown timer in a separate thread
    threading.Thread(target=shutdown_server, daemon=True).start()
    # Run the Flask app
    app.run(host="0.0.0.0", port=5050)