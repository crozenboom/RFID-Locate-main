from flask import Flask, request # type: ignore
import urllib.parse
import csv # type: ignore
import os
from datetime import datetime

app = Flask(__name__)

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
    tag_reads = []
    for value_line in field_values:
        if value_line.strip():  # Skip empty lines
            values = value_line.split(',')
            tag_read = dict(zip(field_names, values))
            tag_reads.append(tag_read)
    
    # Generate CSV file with timestamp in filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"rfid_data_{timestamp}.csv"
    
    # Write to CSV
    with open(csv_filename, mode='w', newline='') as csv_file:
        # Define CSV columns: reader metadata + dynamic field names
        csv_columns = ['reader_name', 'mac_address'] + field_names
        writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
        
        # Write header
        writer.writeheader()
        
        # Write tag reads
        for read in tag_reads:
            # Combine reader metadata with tag read data
            row = {'reader_name': reader_name, 'mac_address': mac_address}
            row.update(read)
            writer.writerow(row)
    
    # Print parsed data to console (as before)
    print(f"Reader: {reader_name}, MAC: {mac_address}")
    for read in tag_reads:
        print(f"Tag Read: {read}")
    
    # Confirm CSV was written
    print(f"Data written to {csv_filename}")
    
    return "OK", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)