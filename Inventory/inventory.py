import subprocess
import sys
import time
import ast
import csv
import re

# Configuration
HOST = '192.168.0.219'
TIME = 20  # User-defined time in seconds
SLLURP_COMMAND = ['sllurp', 'inventory', HOST, '-t', str(TIME), '-a', '0', '--impinj-reports']
CSV_FILE = 'rfid_inventory.csv'
COMMAND_TIMEOUT = TIME + 2  # Timeout slightly longer than TIME to allow command completion

def parse_sllurp_output(output):
    """Parse the sllurp command output to extract tag data."""
    tags = []
    # Look for the 'saw tag(s):' section
    match = re.search(r"saw tag\(s\): (\[.*?\])", output, re.DOTALL)
    if match:
        try:
            # Safely evaluate the tag list string as a Python object
            tag_list_str = match.group(1)
            tag_list = ast.literal_eval(tag_list_str)
            for tag in tag_list:
                # Get the number of times the tag was seen
                tag_seen_count = tag.get('TagSeenCount', 1)
                # Create a row for each tag read
                for _ in range(tag_seen_count):
                    tag_data = {
                        'AntennaID': tag.get('AntennaID', 0),
                        'EPC': tag.get('EPC', b'').decode('utf-8') if isinstance(tag.get('EPC'), bytes) else tag.get('EPC', ''),
                        'FirstSeen': tag.get('FirstSeenTimestampUTC', 0),
                        'LastSeen': tag.get('LastSeenTimestampUTC', 0),
                        'ImpinjPeakRSSI': tag.get('ImpinjPeakRSSI', 0),
                        'Phase Angle': tag.get('ImpinjRFPhaseAngle', 0),
                        'Doppler Frequency': tag.get('ImpinjRFDopplerFrequency', 0),
                    }
                    tags.append(tag_data)
        except (SyntaxError, ValueError) as e:
            print(f"Error parsing tag data: {e}")
    return tags

def write_to_csv(tags):
    """Write tag data to CSV file."""
    if not tags:
        print("No tag data to write to CSV.")
        return

    # Define the exact field order for the CSV
    fieldnames = ['AntennaID', 'EPC', 'FirstSeen', 'LastSeen', 'ImpinjPeakRSSI', 'Phase Angle', 'Doppler Frequency']
    
    with open(CSV_FILE, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if f.tell() == 0:  # Write header only if file is empty
            writer.writeheader()
        writer.writerows(tags)
    print(f"Data saved to {CSV_FILE}")

def main():
    # Run sllurp command
    print(f"Running command: {' '.join(SLLURP_COMMAND)}")
    try:
        # Run the command and capture output
        process = subprocess.Popen(
            SLLURP_COMMAND,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # Wait for the command to complete or timeout
        start_time = time.time()
        output_lines = []
        while process.poll() is None and time.time() - start_time < COMMAND_TIMEOUT:
            line = process.stdout.readline()
            if line:
                print(line.strip())
                output_lines.append(line)
            time.sleep(0.1)

        # Terminate the process if it's still running
        if process.poll() is None:
            print("Command timeout reached. Terminating...")
            process.terminate()
            process.wait(timeout=2)  # Wait for termination

        # Capture any remaining output
        remaining_output, _ = process.communicate()
        if remaining_output:
            output_lines.append(remaining_output)
            print(remaining_output.strip())

        # Combine all output
        full_output = ''.join(output_lines)

        # Parse output and save to CSV
        tags = parse_sllurp_output(full_output)
        write_to_csv(tags)

    except subprocess.SubprocessError as e:
        print(f"Error running sllurp command: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()