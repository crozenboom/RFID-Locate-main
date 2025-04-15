from flask import Flask, request
import urllib.parse

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
    
    # Print or process the parsed data
    print(f"Reader: {reader_name}, MAC: {mac_address}")
    for read in tag_reads:
        print(f"Tag Read: {read}")
    
    return "OK", 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5050)