import subprocess
import logging
import re
import csv
from datetime import datetime

# Set up minimal logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Reader settings (matching CLI)
READER_IP = "169.254.1.1"
PORT = 14150
CSV_FILE = "rfid_data.csv"  # Output CSV file name

def run_inventory():
    """Run sllurp CLI inventory, parse tag data, and save to CSV."""
    logger.info(f"Connected to R420 Reader at {READER_IP}:{PORT}")
    
    cmd = [
        "sllurp", "inventory", READER_IP,
        "-p", str(PORT), "-a", "1", "-X", "20", "-t", "10",
        "--impinj-reports"
    ]
    
    try:
        # Run CLI command with timeout (10s inventory + 1s buffer)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=11
        )
        output = result.stdout
        
        # Parse tags from output
        all_tags = []
        tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
        for match in tag_pattern.finditer(output):
            tag_str = match.group(1)
            tag_dict = eval(tag_str, {"__builtins__": {}}, {})
            if isinstance(tag_dict, list):
                all_tags.extend(tag_dict)
        
        # Log and save results
        if all_tags:
            logger.info("Success - Reader data collected:")
            # Open CSV file in append mode
            with open(CSV_FILE, 'a', newline='') as csvfile:
                fieldnames = [
                    'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
                    'PhaseAngle', 'Frequency', 'ReadCount', 'DopplerFrequency'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is empty/new
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                # Process each tag
                run_timestamp = datetime.now().isoformat()  # Timestamp of this run
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
                    read_count = tag.get('TagSeenCount', 0)
                    doppler_frequency = tag.get('ImpinjRFDopplerFrequency', 'Unknown')
                    if doppler_frequency != 'Unknown':
                        doppler_frequency = f"{doppler_frequency / 100:.2f}"
                    timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                    
                    # Log to console
                    logger.info(
                        f"  Tag Data:\n"
                        f"    Timestamp: {timestamp}\n"
                        f"    EPC: {epc}\n"
                        f"    Antenna: {antenna}\n"
                        f"    RSSI: {rssi}\n"
                        f"    PhaseAngle: {phase_angle}\n"
                        f"    Frequency: {frequency}\n"
                        f"    ReadCount: {read_count}\n"
                        f"    DopplerFrequency: {doppler_frequency}\n"
                    )
                    
                    # Write to CSV
                    writer.writerow({
                        'RunTimestamp': run_timestamp,
                        'Timestamp': timestamp,
                        'EPC': epc,
                        'Antenna': antenna,
                        'RSSI': rssi,
                        'PhaseAngle': phase_angle,
                        'Frequency': frequency,
                        'ReadCount': read_count,
                        'DopplerFrequency': doppler_frequency
                    })
            logger.info(f"Total # of tags seen: {len(all_tags)}")
            logger.info(f"Data saved to {CSV_FILE}")
        else:
            logger.info("Success - No tag data collected")
        
    except subprocess.TimeoutExpired as e:
        logger.info("Inventory completed after 10 seconds")
        output = e.stdout
        
        # Parse tags from output
        all_tags = []
        tag_pattern = re.compile(r"saw tag\(s\): (\[.*?\])", re.DOTALL)
        for match in tag_pattern.finditer(output):
            tag_str = match.group(1)
            tag_dict = eval(tag_str, {"__builtins__": {}}, {})
            if isinstance(tag_dict, list):
                all_tags.extend(tag_dict)
        
        # Log and save results
        if all_tags:
            logger.info("Success - Reader data collected:")
            # Open CSV file in append mode
            with open(CSV_FILE, 'a', newline='') as csvfile:
                fieldnames = [
                    'RunTimestamp', 'Timestamp', 'EPC', 'Antenna', 'RSSI',
                    'PhaseAngle', 'Frequency', 'ReadCount', 'DopplerFrequency'
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Write header if file is empty/new
                if csvfile.tell() == 0:
                    writer.writeheader()
                
                # Process each tag
                run_timestamp = datetime.now().isoformat()  # Timestamp of this run
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
                    read_count = tag.get('TagSeenCount', 0)
                    doppler_frequency = tag.get('ImpinjRFDopplerFrequency', 'Unknown')
                    if doppler_frequency != 'Unknown':
                        doppler_frequency = f"{doppler_frequency / 100:.2f}"
                    timestamp = tag.get('LastSeenTimestampUTC', 'Unknown')
                    
                    # Log to console
                    logger.info(
                        f"  Tag Data:\n"
                        f"    Timestamp: {timestamp}\n"
                        f"    EPC: {epc}\n"
                        f"    Antenna: {antenna}\n"
                        f"    RSSI: {rssi}\n"
                        f"    PhaseAngle: {phase_angle}\n"
                        f"    Frequency: {frequency}\n"
                        f"    ReadCount: {read_count}\n"
                        f"    DopplerFrequency: {doppler_frequency}\n"
                    )
                    
                    # Write to CSV
                    writer.writerow({
                        'RunTimestamp': run_timestamp,
                        'Timestamp': timestamp,
                        'EPC': epc,
                        'Antenna': antenna,
                        'RSSI': rssi,
                        'PhaseAngle': phase_angle,
                        'Frequency': frequency,
                        'ReadCount': read_count,
                        'DopplerFrequency': doppler_frequency
                    })
            logger.info(f"Total # of tags seen: {len(all_tags)}")
            logger.info(f"Data saved to {CSV_FILE}")
        else:
            logger.info("Success - No tag data collected")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to run inventory: {e}\n{e.stderr}")
    except Exception as e:
        logger.error(f"Error parsing output: {e}")

if __name__ == "__main__":
    run_inventory()