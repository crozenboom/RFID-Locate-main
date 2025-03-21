import subprocess
import re
import csv
from datetime import datetime
import sys

# Reader settings (defaults)
READER_IP = "169.254.1.1"
DEFAULT_PORT = 14150
CSV_FILE = "rfid_data.csv"  # Output CSV file name

def get_user_input(prompt, default=None, type_cast=str, validator=None):
    """Helper function to get and validate user input."""
    while True:
        user_input = input(f"{prompt} [{'Enter to skip' if default is None else default}]: ").strip()
        if not user_input:  # If user presses Enter, use default
            return default
        try:
            value = type_cast(user_input)
            if validator and not validator(value):
                print(f"Invalid input: {user_input}. Please try again.")
                continue
            return value
        except ValueError:
            print(f"Invalid input: {user_input}. Please enter a valid {type_cast.__name__}.")

def validate_comma_separated_integers(value):
    """Validate comma-separated integers (e.g., '1,2,3,4')."""
    try:
        return all(int(x) >= 0 for x in value.split(','))
    except ValueError:
        return False

def validate_search_mode(value):
    """Validate Impinj search mode (1, 2, or 0 to skip)."""
    return value in [0, 1, 2]

def configure_inventory_options():
    """Prompt user for sllurp inventory options and return the command list."""
    print("\nConfiguring sllurp inventory options...\n")

    # Port (-p, --port)
    port = get_user_input("Port number", DEFAULT_PORT, int, lambda x: x > 0)

    # Time (-t, --time)
    time = get_user_input("Inventory duration (seconds)", 10.0, float, lambda x: x > 0)

    # Report every N tags (-n, --report-every-n-tags)
    report_n = get_user_input("Report every N tags (0 to disable)", 0, int, lambda x: x >= 0)

    # Antennas (-a, --antennas)
    antennas = get_user_input("Antennas (comma-separated, e.g., '1,2,3,4', 0=all)", "1", str, validate_comma_separated_integers)

    # Transmit power (-X, --tx-power)
    tx_power = get_user_input("Transmit power (0=max power)", 0, int, lambda x: x >= 0)

    # Tari (-T, --tari)
    tari = get_user_input("Tari value (0=auto)", 0, int, lambda x: x >= 0)

    # Session (-s, --session)
    session = get_user_input("Gen2 session", 2, int, lambda x: x >= 0)

    # Mode identifier (--mode-identifier)
    mode_id = get_user_input("ModeIdentifier value (Enter to skip)", 0, int)

    # Tag population (-P, --tag-population)
    tag_population = get_user_input("Tag Population value", 4, int, lambda x: x > 0)

    # Reconnect (-r, --reconnect)
    reconnect = get_user_input("Reconnect on connection failure? (y/n)", "n", str, lambda x: x.lower() in ['y', 'n'])
    reconnect = reconnect.lower() == 'y'

    # Reconnect retries (--reconnect-retries)
    if reconnect:
        reconnect_retries = get_user_input("Max reconnect attempts (-1 for unlimited)", 0, int)
    else:
        reconnect_retries = None

    # Tag filter mask (--tag-filter-mask)
    tag_filter = get_user_input("Tag filter mask (EPC prefix, Enter to skip)", None, str)
    tag_filters = [tag_filter] if tag_filter else []

    # Keepalive interval (--keepalive-interval)
    keepalive = get_user_input("Keepalive interval (ms, 0 to disable)", 0, int, lambda x: x >= 0)

    # Impinj extended configuration (--impinj-extended-configuration)
    impinj_ext = get_user_input("Get Impinj extended configuration? (y/n)", "n", str, lambda x: x.lower() in ['y', 'n'])
    impinj_ext = impinj_ext.lower() == 'y'

    # Impinj search mode (--impinj-search-mode)
    impinj_search = get_user_input("Impinj search mode (1=single, 2=dual, 0 to skip)", 0, int, validate_search_mode)

    # Impinj reports (--impinj-reports)
    impinj_reports = get_user_input("Enable Impinj reports (Phase angle, RSSI, Doppler)? (y/n)", "y", str, lambda x: x.lower() in ['y', 'n'])
    impinj_reports = impinj_reports.lower() == 'y'

    # Frequencies (-f, --frequencies)
    frequencies = get_user_input("Frequencies (comma-separated indexes, e.g., '1,2,3', 0=all)", "1", str, validate_comma_separated_integers)

    # Hoptable ID (--hoptable-id)
    hoptable_id = get_user_input("HopTableID (1 by default)", 1, int, lambda x: x > 0)

    # Construct the sllurp command
    cmd = ["sllurp", "inventory", READER_IP, "-p", str(port), "-t", str(time)]

    if report_n > 0:
        cmd.extend(["-n", str(report_n)])
    if antennas:
        cmd.extend(["-a", antennas])
    if tx_power > 0:
        cmd.extend(["-X", str(tx_power)])
    if tari > 0:
        cmd.extend(["-T", str(tari)])
    if session != 2:  # Only add if different from default
        cmd.extend(["-s", str(session)])
    if mode_id > 0:
        cmd.extend(["--mode-identifier", str(mode_id)])
    if tag_population != 4:  # Only add if different from default
        cmd.extend(["-P", str(tag_population)])
    if reconnect:
        cmd.append("-r")
        if reconnect_retries is not None and reconnect_retries != 0:
            cmd.extend(["--reconnect-retries", str(reconnect_retries)])
    for filter in tag_filters:
        cmd.extend(["--tag-filter-mask", filter])
    if keepalive > 0:
        cmd.extend(["--keepalive-interval", str(keepalive)])
    if impinj_ext:
        cmd.append("--impinj-extended-configuration")
    if impinj_search > 0:
        cmd.extend(["--impinj-search-mode", str(impinj_search)])
    if impinj_reports:
        cmd.append("--impinj-reports")
    if frequencies:
        cmd.extend(["-f", frequencies])
    if hoptable_id != 1:  # Only add if different from default
        cmd.extend(["--hoptable-id", str(hoptable_id)])

    return cmd

def run_inventory():
    """Run sllurp CLI inventory, parse tag data, and save to CSV with each read as a separate row."""
    # Get user-configured sllurp command
    cmd = configure_inventory_options()

    try:
        # Display "Reading..." message
        print("Reading...", end="", flush=True)

        # Run CLI command with timeout (user-specified time + 10s buffer)
        inventory_time = float([cmd[i + 1] for i, x in enumerate(cmd) if x == "-t"][0])
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=inventory_time + 10  # Dynamic timeout based on -t
        )
        output = result.stdout
        
        # Clear "Reading..." message
        print("\r" + " " * 10 + "\r", end="", flush=True)

        # Parse tags from output
        all_tags = []
        tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
        for match in tag_pattern.finditer(output):
            tag_str = match.group(1)
            tag_dict = eval(tag_str, {"__builtins__": {}}, {})
            if isinstance(tag_dict, list):
                all_tags.extend(tag_dict)
        
        # Save results to CSV
        with open(CSV_FILE, 'a', newline='') as csvfile:
            fieldnames = [
                'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
                'PhaseAngle', 'Frequency', 'DopplerFrequency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header if file is empty/new
            if csvfile.tell() == 0:
                writer.writeheader()
            
            # Process each tag
            run_timestamp = datetime.now().isoformat()  # Timestamp of this run
            total_rows = 0  # Count total rows written to CSV
            
            for tag in all_tags:
                epc = tag.get('EPC', 'Unknown')
                if isinstance(epc, bytes):  # Convert bytes to hex string
                    epc = epc.hex()
                antenna = tag.get('AntennaID', 'Unknown')
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
                
                # Write a row for each individual read
                for _ in range(read_count):
                    writer.writerow({
                        'RunTimestamp': run_timestamp,
                        'Timestamp': timestamp,
                        'EPC': epc,
                        'Antenna': antenna,
                        'RSSI': rssi,
                        'PhaseAngle': phase_angle,
                        'Frequency': frequency,
                        'DopplerFrequency': doppler_frequency
                    })
                    total_rows += 1
        
        # Minimal success message
        print(f"Inventory completed successfully. {total_rows} reads saved to {CSV_FILE}")
        
    except subprocess.TimeoutExpired as e:
        # Clear "Reading..." message
        print("\r" + " " * 10 + "\r", end="", flush=True)

        output = e.stdout
        if output is not None:
            output = output.decode('utf-8')  # Convert bytes to string
        else:
            output = ""
        
        # Parse tags from output
        all_tags = []
        tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
        for match in tag_pattern.finditer(output):
            tag_str = match.group(1)
            tag_dict = eval(tag_str, {"__builtins__": {}}, {})
            if isinstance(tag_dict, list):
                all_tags.extend(tag_dict)
        
        # Save results to CSV
        with open(CSV_FILE, 'a', newline='') as csvfile:
            fieldnames = [
                'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
                'PhaseAngle', 'Frequency', 'DopplerFrequency'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            if csvfile.tell() == 0:
                writer.writeheader()
            
            run_timestamp = datetime.now().isoformat()
            total_rows = 0
            
            for tag in all_tags:
                epc = tag.get('EPC', 'Unknown')
                if isinstance(epc, bytes):
                    epc = epc.hex()
                antenna = tag.get('AntennaID', 'Unknown')
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
                
                for _ in range(read_count):
                    writer.writerow({
                        'RunTimestamp': run_timestamp,
                        'Timestamp': timestamp,
                        'EPC': epc,
                        'Antenna': antenna,
                        'RSSI': rssi,
                        'PhaseAngle': phase_angle,
                        'Frequency': frequency,
                        'DopplerFrequency': doppler_frequency
                    })
                    total_rows += 1
        
        print(f"Inventory completed successfully. {total_rows} reads saved to {CSV_FILE}")
        
    except subprocess.CalledProcessError as e:
        # Clear "Reading..." message
        print("\r" + " " * 10 + "\r", end="", flush=True)
        print(f"Failed to run inventory: {e}\n{e.stderr}")
    except Exception as e:
        # Clear "Reading..." message
        print("\r" + " " * 10 + "\r", end="", flush=True)
        print(f"Error: {e}")

if __name__ == "__main__":
    run_inventory()