import csv
import logging
import time
from datetime import datetime
from twisted.internet import reactor
from sllurp import llrp
import sys

# Configure logging
logging.getLogger().setLevel(logging.INFO)
sllurp_logger = logging.getLogger('sllurp')
sllurp_logger.setLevel(logging.DEBUG)
sllurp_logger.addHandler(logging.StreamHandler())

# Reader configuration
READER_IP = '192.168.0.219'
READER_PORT = 5084
CSV_FILE = 'rfid_tags.csv'
INVENTORY_INTERVAL = 1.0  # Run inventory every 1 second

# Global variables
factory = None
start_time = None
end_time = None
csv_writer = None
csv_file = None

def init_csv():
    """Initialize CSV file with headers."""
    global csv_file, csv_writer
    csv_file = open(CSV_FILE, 'w', newline='')
    headers = ['timestamp', 'antenna', 'rssi', 'epc', 'phase_angle', 'channel_index', 'doppler_frequency']
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(headers)

def tag_report_callback(tag_report):
    """Callback to process tag reports and log to CSV."""
    tags = tag_report.msgdict.get('RO_ACCESS_REPORT', {}).get('TagReportData', [])
    current_time = datetime.utcnow().isoformat()
    
    for tag in tags:
        epc = tag.get('EPC-96', b'').decode('utf-8', errors='ignore') or tag.get('EPCData', {}).get('EPC', b'').decode('utf-8', errors='ignore')
        antenna_id = tag.get('AntennaID', [0])[0]
        rssi = tag.get('PeakRSSI', [0])[0]
        phase_angle = tag.get('PhaseAngle', [None])[0]
        channel_index = tag.get('ChannelIndex', [None])[0]
        doppler_frequency = tag.get('DopplerFrequency', [None])[0]
        
        logging.info(f"Tag: EPC={epc}, Antenna={antenna_id}, RSSI={rssi}, "
                     f"Phase={phase_angle}, Channel={channel_index}, Doppler={doppler_frequency}")
        csv_writer.writerow([current_time, antenna_id, rssi, epc, phase_angle, channel_index, doppler_frequency])
        csv_file.flush()  # Ensure data is written immediately

def start_inventory():
    """Start a single inventory cycle."""
    global factory
    if factory is None:
        # Configure tag content selector for required fields
        tag_content_selector = {
            'EnableROSpecID': False,
            'EnableSpecIndex': False,
            'EnableInventoryParameterSpecID': False,
            'EnableAntennaID': True,
            'EnableChannelIndex': True,
            'EnablePeakRSSI': True,
            'EnableFirstSeenTimestamp': False,
            'EnableLastSeenTimestamp': False,
            'EnableTagSeenCount': False,
            'EnableAccessSpecID': False,
            'EnablePhaseAngle': True,
            'EnableDopplerFrequency': True,
        }
        
        # Create factory with configuration matching command
        factory = llrp.LLRPClientFactory(
            antennas=[0],  # 0 means all antennas
            tx_power=0,    # Max power
            tag_content_selector=tag_content_selector
        )
        factory.addTagReportCallback(tag_report_callback)
        
        # Connect to reader
        reactor.connectTCP(READER_IP, READER_PORT, factory)
    
    # Start inventory
    factory.startInventory()
    
    # Check if runtime has expired
    if end_time and time.time() >= end_time:
        stop_inventory()
    else:
        # Schedule next inventory
        reactor.callLater(INVENTORY_INTERVAL, start_inventory)

def stop_inventory():
    """Stop inventory and clean up."""
    global factory, csv_file
    if factory:
        factory.stopInventory()
        factory.disconnect()
    if csv_file:
        csv_file.close()
    reactor.stop()

def main():
    """Main function to handle user input and start inventory."""
    global start_time, end_time
    print("Enter runtime in seconds (or press Enter to run indefinitely):")
    runtime_input = input().strip()
    
    start_time = time.time()
    if runtime_input:
        try:
            runtime = float(runtime_input)
            if runtime <= 0:
                print("Runtime must be positive.")
                return
            end_time = start_time + runtime
            print(f"Running for {runtime} seconds...")
        except ValueError:
            print("Invalid input. Please enter a number or press Enter.")
            return
    else:
        print("Running indefinitely (press Ctrl+C to stop)...")
    
    init_csv()
    start_inventory()
    reactor.run()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopping inventory...")
        stop_inventory()
    except Exception as e:
        print(f"Error: {e}")
        stop_inventory()