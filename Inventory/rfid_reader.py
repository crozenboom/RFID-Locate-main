import socket
import struct
import queue
import threading
import csv
import time
import datetime
import argparse
import signal
import sys
from typing import Dict, List, Optional

# LLRP message types
LLRP_MSG_CLOSE_CONNECTION = 14
LLRP_MSG_GET_READER_CAPABILITIES = 103
LLRP_MSG_SET_READER_CONFIG = 101
LLRP_MSG_DELETE_ALL_ROSPECS = 22
LLRP_MSG_ADD_ROSPEC = 20
LLRP_MSG_ENABLE_ROSPEC = 24
LLRP_MSG_RO_ACCESS_REPORT = 201

# Global variables
tag_queue = queue.Queue()
running = threading.Event()
reader_socket = None

def create_llrp_message(msg_type: int, data: bytes) -> bytes:
    """Create an LLRP message with header and data."""
    version = 1
    msg_id = int(time.time() * 1000) % 2**32
    length = 10 + len(data)  # Header (10 bytes) + data
    header = struct.pack('>HHI', (version << 10) | msg_type, length, msg_id)
    print(f"{datetime.datetime.now()}: Creating LLRP message: type={msg_type}, length={length}, id={msg_id}")
    return header + data

def parse_llrp_response(data: bytes) -> Dict:
    """Parse an LLRP response message for status."""
    print(f"{datetime.datetime.now()}: Parsing LLRP response: {len(data)} bytes, raw={data.hex()}")
    if len(data) < 10:
        print(f"{datetime.datetime.now()}: Invalid response: too short")
        return {'status': 'Invalid response', 'msg_type': 0, 'error_msg': '', 'capabilities': {}}
    msg_type = struct.unpack('>H', data[0:2])[0] & 0x03FF
    status = 'Unknown'
    error_msg = ''
    capabilities = {}
    pos = 10
    while pos < len(data):
        param_type = struct.unpack('>H', data[pos:pos+2])[0]
        param_length = struct.unpack('>H', data[pos+2:pos+4])[0]
        if param_type == 287:  # LLRPStatus
            status_code = struct.unpack('>H', data[pos+4:pos+6])[0]
            status = 'Success' if status_code == 0 else f'Error {status_code}'
            if param_length > 6:
                error_len = struct.unpack('>H', data[pos+6:pos+8])[0]
                if error_len > 0:
                    error_msg = data[pos+8:pos+8+error_len].decode('utf-8', errors='ignore')
            print(f"{datetime.datetime.now()}: Found LLRPStatus: {status}, error_msg={error_msg}")
        elif param_type == 143:  # GeneralDeviceCapabilities
            if param_length > 12:
                max_antennas = struct.unpack('>H', data[pos+12:pos+14])[0]
                capabilities['max_antennas'] = max_antennas
                device_name_len = struct.unpack('>H', data[pos+4:pos+6])[0]
                if device_name_len > 0 and pos+6+device_name_len <= len(data):
                    device_name = data[pos+6:pos+6+device_name_len].decode('utf-8', errors='ignore')
                    capabilities['device_name'] = device_name
                    print(f"{datetime.datetime.now()}: Found GeneralDeviceCapabilities: max_antennas={max_antennas}, device_name={device_name}")
        elif param_type == 246:  # Vendor-specific parameter
            param_data = data[pos+4:pos+param_length]
            print(f"{datetime.datetime.now()}: Vendor-specific parameter type 246, length={param_length}, data={param_data.hex()}")
            error_msg += f"Vendor-specific error (type 246): {param_data.hex()}"
            if param_length > 4:
                try:
                    error_code = struct.unpack('>I', param_data[0:4])[0]
                    print(f"{datetime.datetime.now()}: Possible vendor error code: {error_code}")
                    error_msg += f", possible error code: {error_code}"
                except:
                    pass
                try:
                    error_str = param_data[4:].decode('utf-8', errors='ignore')
                    if error_str:
                        print(f"{datetime.datetime.now()}: Possible vendor error string: {error_str}")
                        error_msg += f", possible error string: {error_str}"
                except:
                    pass
        else:
            print(f"{datetime.datetime.now()}: Unknown parameter type {param_type}, length={param_length}")
        pos += param_length
    return {'msg_type': msg_type, 'status': status, 'error_msg': error_msg, 'capabilities': capabilities}

def parse_ro_access_report(data: bytes) -> List[Dict]:
    """Parse an RO_ACCESS_REPORT message into a list of tag reads."""
    print(f"{datetime.datetime.now()}: Parsing RO_ACCESS_REPORT: {len(data)} bytes")
    reads = []
    pos = 0
    msg_length = len(data)

    while pos < msg_length:
        param_type = struct.unpack('>H', data[pos:pos+2])[0]
        param_length = struct.unpack('>H', data[pos+2:pos+4])[0]
        
        if param_type == 1024:  # TagReportData
            param_data = data[pos+4:pos+param_length]
            param_pos = 0
            print(f"{datetime.datetime.now()}: Found TagReportData, length={param_length}")

            tag_read = {
                'timestamp': None,
                'antenna': None,
                'rssi': None,
                'epc': None,
                'phase_angle': None,
                'channel_index': None,
                'doppler_frequency': None
            }
            while param_pos < len(param_data):
                sub_type = struct.unpack('>H', param_data[param_pos:param_pos+2])[0]
                sub_length = struct.unpack('>H', param_data[param_pos+2:param_pos+4])[0]
                
                if sub_type == 1025:  # EPCData
                    epc_length = struct.unpack('>H', param_data[param_pos+4:param_pos+6])[0]
                    tag_read['epc'] = param_data[param_pos+6:param_pos+6+epc_length].hex()
                    print(f"{datetime.datetime.now()}: Parsed EPC: {tag_read['epc']}")
                elif sub_type == 1026:  # AntennaID
                    tag_read['antenna'] = struct.unpack('>H', param_data[param_pos+4:param_pos+6])[0]
                    print(f"{datetime.datetime.now()}: Parsed Antenna: {tag_read['antenna']}")
                elif sub_type == 1027:  # PeakRSSI
                    tag_read['rssi'] = struct.unpack('>h', param_data[param_pos+4:param_pos+6])[0]
                    print(f"{datetime.datetime.now()}: Parsed RSSI: {tag_read['rssi']}")
                elif sub_type == 1028:  # FirstSeenTimestampUTC
                    timestamp_us = struct.unpack('>Q', param_data[param_pos+4:param_pos+12])[0]
                    tag_read['timestamp'] = datetime.datetime.utcfromtimestamp(timestamp_us / 1_000_000).isoformat()
                    print(f"{datetime.datetime.now()}: Parsed Timestamp: {tag_read['timestamp']}")
                elif sub_type == 1030:  # PhaseAngle
                    tag_read['phase_angle'] = struct.unpack('>H', param_data[param_pos+4:param_pos+6])[0]
                    print(f"{datetime.datetime.now()}: Parsed PhaseAngle: {tag_read['phase_angle']}")
                elif sub_type == 1031:  # ChannelIndex
                    tag_read['channel_index'] = struct.unpack('>H', param_data[param_pos+4:param_pos+6])[0]
                    print(f"{datetime.datetime.now()}: Parsed ChannelIndex: {tag_read['channel_index']}")
                elif sub_type == 1032:  # DopplerFrequency
                    tag_read['doppler_frequency'] = struct.unpack('>h', param_data[param_pos+4:param_pos+6])[0]
                    print(f"{datetime.datetime.now()}: Parsed DopplerFrequency: {tag_read['doppler_frequency']}")
                
                param_pos += sub_length
            
            if tag_read['epc']:
                reads.append(tag_read)
                print(f"{datetime.datetime.now()}: Added tag read to list: {tag_read}")
        
        pos += param_length
    
    print(f"{datetime.datetime.now()}: Parsed {len(reads)} tag reads")
    return reads

def configure_reader(reader_ip: str, reader_port: int) -> Optional[socket.socket]:
    """Connect to the reader and configure it with minimal messages."""
    global reader_socket
    max_retries = 2
    for attempt in range(max_retries + 1):
        try:
            print(f"{datetime.datetime.now()}: Attempting to connect to {reader_ip}:{reader_port} (attempt {attempt + 1}/{max_retries + 1})")
            reader_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reader_socket.settimeout(5.0)
            reader_socket.connect((reader_ip, reader_port))
            print(f"{datetime.datetime.now()}: Connected to reader")
            time.sleep(0.1)

            # CLOSE_CONNECTION to test session management
            print(f"{datetime.datetime.now()}: Sending CLOSE_CONNECTION")
            close_data = b''
            reader_socket.send(create_llrp_message(LLRP_MSG_CLOSE_CONNECTION, close_data))
            try:
                response = reader_socket.recv(4096)
                status = parse_llrp_response(response)
                print(f"{datetime.datetime.now()}: CLOSE_CONNECTION response: {status}")
            except socket.timeout:
                print(f"{datetime.datetime.now()}: No CLOSE_CONNECTION response, continuing")
            time.sleep(0.1)

            # Reconnect
            reader_socket.close()
            reader_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            reader_socket.settimeout(5.0)
            reader_socket.connect((reader_ip, reader_port))
            print(f"{datetime.datetime.now()}: Reconnected to reader")
            time.sleep(0.1)

            # GET_READER_CAPABILITIES
            print(f"{datetime.datetime.now()}: Sending GET_READER_CAPABILITIES")
            capabilities_data = b'\x00\x01'  # RequestedData: All (type 1)
            reader_socket.send(create_llrp_message(LLRP_MSG_GET_READER_CAPABILITIES, capabilities_data))
            response = reader_socket.recv(4096)
            status = parse_llrp_response(response)
            print(f"{datetime.datetime.now()}: GET_READER_CAPABILITIES response: {status}")
            if status['status'] != 'Success' and status['msg_type'] != 113:
                print(f"{datetime.datetime.now()}: GET_READER_CAPABILITIES failed, continuing anyway")
            time.sleep(0.1)

            # DELETE_ALL_ROSPECS
            print(f"{datetime.datetime.now()}: Sending DELETE_ALL_ROSPECS")
            delete_data = b'\x00\x00\x00\x00'  # ROSpecID 0 (delete all)
            reader_socket.send(create_llrp_message(LLRP_MSG_DELETE_ALL_ROSPECS, delete_data))
            response = reader_socket.recv(4096)
            status = parse_llrp_response(response)
            print(f"{datetime.datetime.now()}: DELETE_ALL_ROSPECS response: {status}")
            if status['status'] != 'Success' and status['msg_type'] != 32:
                print(f"{datetime.datetime.now()}: DELETE_ALL_ROSPECS failed, continuing anyway")
            time.sleep(0.1)

            # SET_READER_CONFIG: Minimal reset
            print(f"{datetime.datetime.now()}: Sending SET_READER_CONFIG")
            config_data = (
                b'\x00\x06\x00\x04'  # Reset to factory defaults (type 6, length 4)
            )
            reader_socket.send(create_llrp_message(LLRP_MSG_SET_READER_CONFIG, config_data))
            response = reader_socket.recv(4096)
            status = parse_llrp_response(response)
            print(f"{datetime.datetime.now()}: SET_READER_CONFIG response: {status}")
            if status['status'] != 'Success' and status['msg_type'] != 111:
                print(f"{datetime.datetime.now()}: SET_READER_CONFIG failed, continuing anyway")
            time.sleep(0.1)

            reader_socket.settimeout(1.0)  # Timeout for tag reading
            print(f"{datetime.datetime.now()}: Reader configured successfully")
            return reader_socket
        
        except (ConnectionResetError, BrokenPipeError, socket.timeout) as e:
            print(f"{datetime.datetime.now()}: Configure reader error (attempt {attempt + 1}): {e}")
            if reader_socket:
                reader_socket.close()
                reader_socket = None
            if attempt < max_retries:
                print(f"{datetime.datetime.now()}: Retrying configuration in 1 second...")
                time.sleep(1)
            else:
                print(f"{datetime.datetime.now()}: Max configuration retries reached")
                return None
        except Exception as e:
            print(f"{datetime.datetime.now()}: Configure reader error (attempt {attempt + 1}): {e}")
            if reader_socket:
                reader_socket.close()
                reader_socket = None
            if attempt < max_retries:
                print(f"{datetime.datetime.now()}: Retrying configuration in 1 second...")
                time.sleep(1)
            else:
                print(f"{datetime.datetime.now()}: Max configuration retries reached")
                return None

def reader_thread(reader_ip: str, reader_port: int):
    """Thread to handle LLRP communication and queue tag reads."""
    max_retries = 3
    retry_delay = 5  # seconds
    print(f"{datetime.datetime.now()}: Starting reader thread")
    print(f"{datetime.datetime.now()}: Running event state: {running.is_set()}")

    while running.is_set():
        print(f"{datetime.datetime.now()}: Entering reader thread loop")
        try:
            global reader_socket
            print(f"{datetime.datetime.now()}: Configuring reader")
            reader_socket = configure_reader(reader_ip, reader_port)
            if not reader_socket:
                raise RuntimeError("Failed to configure reader")

            print(f"{datetime.datetime.now()}: Reader connected and configured")

            while running.is_set():
                try:
                    print(f"{datetime.datetime.now()}: Waiting for data from reader")
                    data = reader_socket.recv(4096)
                    if not data:
                        print(f"{datetime.datetime.now()}: Reader closed connection")
                        break
                    
                    print(f"{datetime.datetime.now()}: Received {len(data)} bytes")
                    if len(data) < 10:
                        print(f"{datetime.datetime.now()}: Data too short, skipping")
                        continue
                    msg_type = struct.unpack('>H', data[0:2])[0] & 0x03FF
                    print(f"{datetime.datetime.now()}: Message type: {msg_type}")
                    if msg_type == LLRP_MSG_RO_ACCESS_REPORT:
                        reads = parse_ro_access_report(data[10:])
                        print(f"{datetime.datetime.now()}: Queuing {len(reads)} tag reads")
                        for read in reads:
                            tag_queue.put(read)
                
                except socket.timeout:
                    print(f"{datetime.datetime.now()}: Receive timeout, continuing")
                    continue
        
        except (ConnectionResetError, ConnectionRefusedError) as e:
            print(f"{datetime.datetime.now()}: Connection error: {e}")
            if reader_socket:
                print(f"{datetime.datetime.now()}: Closing reader socket")
                reader_socket.close()
                reader_socket = None
            if max_retries > 0:
                print(f"{datetime.datetime.now()}: Retrying in {retry_delay} seconds... ({max_retries} retries left)")
                time.sleep(retry_delay)
                max_retries -= 1
                continue
            else:
                print(f"{datetime.datetime.now()}: Max retries reached. Stopping reader thread.")
                break
        
        except Exception as e:
            print(f"{datetime.datetime.now()}: Reader thread error: {e}")
            break
        
        finally:
            if reader_socket:
                print(f"{datetime.datetime.now()}: Closing reader socket")
                reader_socket.close()
                reader_socket = None
    
    print(f"{datetime.datetime.now()}: Reader thread stopped")
    running.clear()

def csv_writer_thread(output_file: str):
    """Thread to write tag reads from queue to CSV."""
    print(f"{datetime.datetime.now()}: Starting CSV writer thread, output file: {output_file}")
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['timestamp', 'antenna', 'rssi', 'epc', 'phase_angle', 'channel_index', 'doppler_frequency']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        print(f"{datetime.datetime.now()}: Wrote CSV header")
        
        batch = []
        last_flush = time.time()
        
        while running.is_set() or not tag_queue.empty():
            try:
                print(f"{datetime.datetime.now()}: Checking queue, size={tag_queue.qsize()}")
                read = tag_queue.get(timeout=1.0)
                batch.append(read)
                print(f"{datetime.datetime.now()}: Dequeued tag read: {read}")
                
                if len(batch) >= 100 or time.time() - last_flush >= 1.0:
                    print(f"{datetime.datetime.now()}: Writing batch of {len(batch)} reads to CSV")
                    for read in batch:
                        writer.writerow(read)
                    csvfile.flush()
                    print(f"{datetime.datetime.now()}: Flushed batch to CSV")
                    batch = []
                    last_flush = time.time()
                
                tag_queue.task_done()
            
            except queue.Empty:
                print(f"{datetime.datetime.now()}: Queue empty")
                if batch:
                    print(f"{datetime.datetime.now()}: Writing remaining batch of {len(batch)} reads to CSV")
                    for read in batch:
                        writer.writerow(read)
                    csvfile.flush()
                    print(f"{datetime.datetime.now()}: Flushed remaining batch to CSV")
                    batch = []
                    last_flush = time.time()
    
    print(f"{datetime.datetime.now()}: CSV writer thread stopped")

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down."""
    print(f"{datetime.datetime.now()}: Received SIGINT, stopping reader...")
    running.clear()
    if reader_socket:
        print(f"{datetime.datetime.now()}: Closing reader socket in signal handler")
        reader_socket.close()
    sys.exit(0)

def main():
    print(f"{datetime.datetime.now()}: Starting RFID reader script")
    parser = argparse.ArgumentParser(description="Custom RFID tag reader")
    parser.add_argument('--reader-ip', default='192.168.1.100', help='Reader IP address')
    parser.add_argument('--reader-port', type=int, default=5084, help='Reader port')
    parser.add_argument('--duration', type=int, help='Run duration in seconds')
    args = parser.parse_args()
    print(f"{datetime.datetime.now()}: Arguments: ip={args.reader_ip}, port={args.reader_port}, duration={args.duration}")

    # Set up output file
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'rfid_reads_{timestamp}.csv'
    print(f"{datetime.datetime.now()}: Output file: {output_file}")

    # Set up signal handler
    print(f"{datetime.datetime.now()}: Setting up signal handler")
    signal.signal(signal.SIGINT, signal_handler)

    # Set running event
    print(f"{datetime.datetime.now()}: Setting running event")
    running.set()

    # Start threads
    print(f"{datetime.datetime.now()}: Starting reader thread")
    reader_t = threading.Thread(target=reader_thread, args=(args.reader_ip, args.reader_port))
    print(f"{datetime.datetime.now()}: Starting CSV writer thread")
    writer_t = threading.Thread(target=csv_writer_thread, args=(output_file,))
    
    reader_t.start()
    writer_t.start()
    print(f"{datetime.datetime.now()}: Threads started")

    # Run for specified duration or indefinitely
    if args.duration:
        print(f"{datetime.datetime.now()}: Running for {args.duration} seconds")
        time.sleep(args.duration)
        print(f"{datetime.datetime.now()}: Duration ended, stopping")
        running.clear()
    
    print(f"{datetime.datetime.now()}: Waiting for threads to join")
    reader_t.join()
    writer_t.join()
    print(f"{datetime.datetime.now()}: Script finished")

if __name__ == '__main__':
    main()