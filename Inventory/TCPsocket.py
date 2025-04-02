import socket
import struct
import time
from datetime import datetime
import csv
import signal
import sys
import queue
import threading

# Reader settings
READER_IP = "192.168.0.219"
LLRP_PORT = 5084
CSV_FILE_BASE = "rfid_data"

def get_user_input(prompt, default=None, type_cast=str, validator=None):
    """Helper function to get and validate user input."""
    while True:
        user_input = input(f"{prompt} [{'Enter to skip' if default is None else default}]: ").strip()
        if not user_input:
            return default
        try:
            value = type_cast(user_input)
            if validator and not validator(value):
                print(f"Invalid input: {value}. Please try again.")
                continue
            return value
        except ValueError:
            print(f"Invalid input: {user_input}. Please enter a valid {type_cast.__name__}.")

def check_reader_connection(max_retries=3, retry_delay=2, timeout=5):
    """Check if the reader is connected by attempting a socket connection to the LLRP port."""
    for attempt in range(1, max_retries + 1):
        print(f"Attempting to connect to reader at {READER_IP}:{LLRP_PORT} (Attempt {attempt}/{max_retries})...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect((READER_IP, LLRP_PORT))
            print("Connection successful: Reader at", READER_IP, "is reachable on port", LLRP_PORT)
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
    
    print(f"Error: Failed to connect to reader at {READER_IP}:{LLRP_PORT} after {max_retries} attempts.")
    return False

def save_to_csv_periodically(data_queue, csv_file, stop_event):
    """Periodically save data from the queue to a CSV file."""
    fieldnames = [
        'RunTimestamp', 'Timestamp', 'EPC', 'AntennaPort', 'RSSI',
        'PhaseAngle', 'Frequency', 'DopplerFrequency'
    ]
    all_reads = []
    
    # Write an empty CSV with headers at the start
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    while not stop_event.is_set():
        # Collect data from the queue
        while not data_queue.empty():
            all_reads.append(data_queue.get())

        # Save to CSV every 5 seconds if there’s new data
        if all_reads:
            with open(csv_file, 'a', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for read in all_reads:
                    writer.writerow(read)
            print(f"\nSaved {len(all_reads)} reads to {csv_file}", flush=True)
            all_reads = []

        time.sleep(5)

def construct_llrp_message(msg_type, msg_id, parameters=b''):
    """Construct an LLRP message in binary format."""
    # LLRP message header: 10 bytes
    # Bits 0-5: Reserved (0)
    # Bits 6-15: Message Type (10 bits)
    # Bits 16-31: Message Length (16 bits)
    # Bits 32-63: Message ID (32 bits)
    msg_length = 10 + len(parameters)  # Header (10 bytes) + parameters
    # Version 1 (001), Reserved (00000), Message Type (10 bits)
    msg_type_field = (1 << 10) | (msg_type & 0x3FF)  # Version 1, message type
    header = struct.pack('>HI', msg_type_field, msg_length) + struct.pack('>I', msg_id)
    return header + parameters

def parse_llrp_message(data):
    """Parse an LLRP message and return the message type, length, ID, and body."""
    if len(data) < 10:
        return None, None, None, None
    msg_type_field, msg_length = struct.unpack('>HI', data[:6])
    msg_id = struct.unpack('>I', data[6:10])[0]
    msg_type = msg_type_field & 0x3FF  # Extract the message type (last 10 bits)
    body = data[10:msg_length]
    return msg_type, msg_length, msg_id, body

def parse_tag_report(body, epc_filter):
    """Parse an RO_ACCESS_REPORT message body and extract tag data."""
    tags = []
    pos = 0
    while pos < len(body):
        if pos + 4 > len(body):
            break
        param_type, param_length = struct.unpack('>HH', body[pos:pos+4])
        param_type &= 0x3FF  # Last 10 bits
        if param_type != 240:  # TagReportData
            pos += param_length
            continue
        param_body = body[pos+4:pos+param_length]
        pos += param_length

        # Parse TagReportData
        tag_data = {}
        tag_pos = 0
        while tag_pos < len(param_body):
            if tag_pos + 4 > len(param_body):
                break
            sub_param_type, sub_param_length = struct.unpack('>HH', param_body[tag_pos:tag_pos+4])
            sub_param_type &= 0x3FF
            sub_param_body = param_body[tag_pos+4:tag_pos+sub_param_length]
            tag_pos += sub_param_length

            if sub_param_type == 128:  # EPCData
                epc_length_bits = struct.unpack('>H', sub_param_body[:2])[0]
                epc_length_bytes = (epc_length_bits + 7) // 8
                epc = sub_param_body[2:2+epc_length_bytes].hex()
                tag_data['EPC'] = epc
            elif sub_param_type == 139:  # AntennaID
                tag_data['AntennaID'] = struct.unpack('>H', sub_param_body)[0]
            elif sub_param_type == 141:  # PeakRSSI
                tag_data['PeakRSSI'] = struct.unpack('>h', sub_param_body)[0] / 100.0
            elif sub_param_type == 142:  # ChannelIndex
                tag_data['ChannelIndex'] = struct.unpack('>H', sub_param_body)[0]
            elif sub_param_type == 145:  # LastSeenTimestampUTC
                tag_data['LastSeenTimestampUTC'] = struct.unpack('>Q', sub_param_body)[0]
            elif sub_param_type == 146:  # TagSeenCount
                tag_data['TagSeenCount'] = struct.unpack('>H', sub_param_body)[0]
            elif sub_param_type == 1023:  # Custom (Impinj extensions)
                vendor_id, subtype = struct.unpack('>IB', sub_param_body[:5])
                if vendor_id == 25882:  # Impinj
                    if subtype == 2:  # ImpinjTagReportContentSelector
                        sub_pos = 5
                        while sub_pos < len(sub_param_body):
                            if sub_pos + 4 > len(sub_param_body):
                                break
                            custom_type, custom_length = struct.unpack('>HH', sub_param_body[sub_pos:sub_pos+4])
                            custom_type &= 0x3FF
                            custom_body = sub_param_body[sub_pos+4:sub_pos+custom_length]
                            sub_pos += custom_length
                            if custom_type == 1:  # ImpinjRFPhaseAngle
                                tag_data['ImpinjRFPhaseAngle'] = struct.unpack('>H', custom_body)[0] / 100.0
                            elif custom_type == 2:  # ImpinjRFDopplerFrequency
                                tag_data['ImpinjRFDopplerFrequency'] = struct.unpack('>h', custom_body)[0] / 100.0

        # Apply EPC filter
        epc = tag_data.get('EPC', 'Unknown')
        if epc_filter and epc != 'Unknown' and epc not in epc_filter:
            continue

        tags.append(tag_data)

    return tags

def run_inventory():
    # Check reader connection before proceeding
    if not check_reader_connection():
        print("Exiting due to connection failure.")
        sys.exit(1)

    # Get total runtime from user
    total_time = get_user_input("Total inventory duration (seconds)", 10, float, lambda x: x > 0)

    # Get EPC filter from user
    def validate_epc_list(epc_str):
        if not epc_str:
            return True
        epcs = [epc.strip() for epc in epc_str.split(',')]
        for epc in epcs:
            if not re.match(r'^[0-9a-fA-F]+$', epc):
                print(f"Invalid EPC format: {epc}. EPCs must be hexadecimal strings.")
                return False
        return True

    epc_input = get_user_input("EPC(s) to filter", None, str, validate_epc_list)
    epc_filter = None if epc_input is None else [epc.strip() for epc in epc_input.split(',')]

    # Generate timestamp for the filename
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    csv_file = f"{CSV_FILE_BASE}_{timestamp}.csv"

    # Set up threading components
    data_queue = queue.Queue()
    stop_event = threading.Event()

    # Start a thread to save data periodically
    save_thread = threading.Thread(
        target=save_to_csv_periodically,
        args=(data_queue, csv_file, stop_event),
        daemon=True
    )
    save_thread.start()

    # Connect to the reader
    print(f"Connecting to reader at {READER_IP}:{LLRP_PORT}...")
    with socket.create_connection((READER_IP, LLRP_PORT)) as s:
        s.settimeout(5)  # Set a timeout for receiving data

        # Step 1: Reset the reader configuration (SET_READER_CONFIG with ResetToFactoryDefault)
        set_reader_config = construct_llrp_message(
            msg_type=20,  # SET_READER_CONFIG
            msg_id=1,
            parameters=(
                b'\x00\x14\x00\x06' +  # SET_READER_CONFIG parameter (type 20, length 6)
                b'\x01' +  # ResetToFactoryDefault = True
                b'\x00'  # Reserved
            )
        )
        s.sendall(set_reader_config)
        response = s.recv(4096)
        msg_type, _, _, _ = parse_llrp_message(response)
        if msg_type != 110:  # SET_READER_CONFIG_RESPONSE
            print("Failed to reset reader configuration")
            sys.exit(1)

        # Step 2: Enable Impinj Extensions (CUSTOM_MESSAGE)
        impinj_enable_extensions = construct_llrp_message(
            msg_type=60,  # CUSTOM_MESSAGE
            msg_id=2,
            parameters=(
                b'\x00\x3C\x00\x0E' +  # VendorDescriptor (type 60, length 14)
                struct.pack('>I', 25882) +  # VendorID (Impinj = 25882)
                b'\x00\x00\x00\x00' +  # Reserved
                b'\x03\xFF\x00\x09' +  # Custom (type 1023, length 9)
                struct.pack('>I', 25882) +  # VendorID
                b'\x15' +  # Subtype 21 (IMPINJ_ENABLE_EXTENSIONS)
                b''  # No additional data
            )
        )
        s.sendall(impinj_enable_extensions)
        response = s.recv(4096)
        msg_type, _, _, _ = parse_llrp_message(response)
        if msg_type != 60:  # CUSTOM_MESSAGE_RESPONSE
            print("Failed to enable Impinj extensions")
            sys.exit(1)

        # Step 3: Add an RO_SPEC (ADD_ROSPEC)
        # RO_SPEC structure:
        # - ROSpecID: 1
        # - Priority: 0
        # - CurrentState: Disabled
        # - ROBoundarySpec: Immediate start, no stop trigger
        # - AISpec: Antennas 1, 2, 3, 4; Duration trigger (total_time)
        # - ROReportSpec: Report every tag
        ro_spec = (
            b'\x00\x8E\x00\x4A' +  # ROSpec (type 142, length 74)
            struct.pack('>I', 1) +  # ROSpecID
            b'\x00' +  # Priority
            b'\x00' +  # CurrentState (Disabled)
            b'\x00\x8F\x00\x0A' +  # ROBoundarySpec (type 143, length 10)
            b'\x00\x90\x00\x04' +  # ROSpecStartTrigger (type 144, length 4)
            b'\x00' +  # Immediate
            b'\x00\x00' +  # Reserved
            b'\x00\x91\x00\x06' +  # ROSpecStopTrigger (type 145, length 6)
            b'\x00' +  # Null
            b'\x00\x00\x00\x00' +  # Duration (0 = no stop)
            b'\x00\x92\x00\x1C' +  # AISpec (type 146, length 28)
            b'\x00\x04' +  # AntennaCount
            struct.pack('>HHHH', 1, 2, 3, 4) +  # AntennaIDs
            b'\x00\x93\x00\x08' +  # AISpecStopTrigger (type 147, length 8)
            b'\x01' +  # Duration
            b'\x00' +  # Reserved
            struct.pack('>I', int(total_time * 1000)) +  # Duration in milliseconds
            b'\x00\x94\x00\x06' +  # InventoryParameterSpec (type 148, length 6)
            struct.pack('>H', 1) +  # InventoryParameterSpecID
            b'\x01' +  # ProtocolID (EPCGlobalClass1Gen2)
            b'\x00\x00' +  # Reserved
            b'\x00\x95\x00\x12' +  # ROReportSpec (type 149, length 18)
            b'\x01' +  # ROReportTrigger (Upon_N_Tags_Or_End_Of_AISpec)
            struct.pack('>H', 1) +  # N (1 tag)
            b'\x00\x96\x00\x0C' +  # TagReportContentSelector (type 150, length 12)
            b'\x00\x00' +  # Enable bits (set below)
            b'\x00\x00' +  # Enable bits (set below)
            b'\x00\x00' +  # Enable bits (set below)
            b'\x00\x00' +  # Enable bits (set below)
            b'\x00\x00'  # Reserved
        )
        # Enable specific fields in TagReportContentSelector
        # Enable: AntennaID, ChannelIndex, PeakRSSI, LastSeenTimestamp, TagSeenCount
        ro_spec = ro_spec[:77] + b'\x00\x1C' + ro_spec[79:]  # Set bits 2, 3, 4, 6, 7
        add_ro_spec = construct_llrp_message(
            msg_type=30,  # ADD_ROSPEC
            msg_id=3,
            parameters=ro_spec
        )
        s.sendall(add_ro_spec)
        response = s.recv(4096)
        msg_type, _, _, _ = parse_llrp_message(response)
        if msg_type != 120:  # ADD_ROSPEC_RESPONSE
            print("Failed to add RO_SPEC")
            sys.exit(1)

        # Step 4: Enable the RO_SPEC (ENABLE_ROSPEC)
        enable_ro_spec = construct_llrp_message(
            msg_type=32,  # ENABLE_ROSPEC
            msg_id=4,
            parameters=(
                b'\x00\xA0\x00\x06' +  # ROSpecID (type 160, length 6)
                struct.pack('>I', 1)  # ROSpecID
            )
        )
        s.sendall(enable_ro_spec)
        response = s.recv(4096)
        msg_type, _, _, _ = parse_llrp_message(response)
        if msg_type != 122:  # ENABLE_ROSPEC_RESPONSE
            print("Failed to enable RO_SPEC")
            sys.exit(1)

        # Step 5: Listen for RO_ACCESS_REPORT messages
        print(f"Running inventory for {total_time} seconds. Press Ctrl+C to stop early.", flush=True)

        # Handle Ctrl+C to stop gracefully
        def signal_handler(sig, frame):
            print("\nStopping inventory...", flush=True)
            stop_event.set()
            # Send STOP_ROSPEC
            stop_ro_spec = construct_llrp_message(
                msg_type=34,  # STOP_ROSPEC
                msg_id=5,
                parameters=(
                    b'\x00\xA0\x00\x06' +  # ROSpecID (type 160, length 6)
                    struct.pack('>I', 1)  # ROSpecID
                )
            )
            s.sendall(stop_ro_spec)
            response = s.recv(4096)
            # Save any remaining data
            all_reads = []
            while not data_queue.empty():
                all_reads.append(data_queue.get())
            if all_reads:
                with open(csv_file, 'a', newline='') as csvfile:
                    fieldnames = [
                        'RunTimestamp', 'Timestamp', 'EPC', 'AntennaPort', 'RSSI',
                        'PhaseAngle', 'Frequency', 'DopplerFrequency'
                    ]
                    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                    for read in all_reads:
                        writer.writerow(read)
                print(f"Final save: {len(all_reads)} reads to {csv_file}", flush=True)
            print("Inventory stopped.", flush=True)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)

        # Run for the specified total time
        start_time = time.time()
        buffer = b''
        try:
            while time.time() - start_time < total_time:
                try:
                    data = s.recv(4096)
                    if not data:
                        break
                    buffer += data

                    # Process complete messages from the buffer
                    while len(buffer) >= 10:  # Minimum length for an LLRP message header
                        msg_type, msg_length, msg_id, body = parse_llrp_message(buffer)
                        if msg_type is None or msg_length > len(buffer):
                            break  # Incomplete message, wait for more data

                        if msg_type == 62:  # RO_ACCESS_REPORT
                            tags = parse_tag_report(body, epc_filter)
                            run_timestamp = datetime.now().isoformat()
                            for tag in tags:
                                epc = tag.get('EPC', 'Unknown')
                                antenna = tag.get('AntennaID', 'Unknown')
                                rssi = tag.get('PeakRSSI', 'Unknown')
                                phase_angle = tag.get('ImpinjRFPhaseAngle', 'Unknown')
                                frequency = tag.get('ChannelIndex', 'Unknown')
                                doppler_frequency = tag.get('ImpinjRFDopplerFrequency', 'Unknown')
                                timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                                read_count = tag.get('TagSeenCount', 1)

                                for _ in range(read_count):
                                    read_data = {
                                        'RunTimestamp': run_timestamp,
                                        'Timestamp': timestamp,
                                        'EPC': epc,
                                        'AntennaPort': str(antenna),
                                        'RSSI': rssi,
                                        'PhaseAngle': phase_angle,
                                        'Frequency': frequency,
                                        'DopplerFrequency': doppler_frequency
                                    }
                                    data_queue.put(read_data)

                        buffer = buffer[msg_length:]  # Remove the processed message

                except socket.timeout:
                    continue

        except KeyboardInterrupt:
            signal_handler(None, None)

        # Step 6: Stop the inventory after the runtime expires
        print("\nRuntime completed. Stopping inventory...", flush=True)
        stop_event.set()
        stop_ro_spec = construct_llrp_message(
            msg_type=34,  # STOP_ROSPEC
            msg_id=5,
            parameters=(
                b'\x00\xA0\x00\x06' +  # ROSpecID (type 160, length 6)
                struct.pack('>I', 1)  # ROSpecID
            )
        )
        s.sendall(stop_ro_spec)
        response = s.recv(4096)

        # Save any remaining data
        all_reads = []
        while not data_queue.empty():
            all_reads.append(data_queue.get())
        if all_reads:
            with open(csv_file, 'a', newline='') as csvfile:
                fieldnames = [
                    'RunTimestamp', 'Timestamp', 'EPC', 'AntennaPort', 'RSSI',
                    'PhaseAngle', 'Frequency', 'DopplerFrequency'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                for read in all_reads:
                    writer.writerow(read)
            print(f"Final save: {len(all_reads)} reads to {csv_file}", flush=True)

        print("Inventory completed.", flush=True)

if __name__ == "__main__":
    run_inventory()