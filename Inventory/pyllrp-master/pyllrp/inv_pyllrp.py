#!/usr/bin/env python
import sys
import time
import signal 
try:
	from . import pyllrp
	from .AutoDetect import AutoDetect
	from .LLRPConnector import LLRPConnector
	from .TagInventory import TagInventory
except Exception as e:
	import pyllrp
	from AutoDetect import AutoDetect
	from LLRPConnector import LLRPConnector
	from TagInventory import TagInventory

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    """Handle Ctrl+C to gracefully shut down."""
    global running
    print('\nShutting down...')
    running = False

def access_report_handler(connector, access_report):
    """Handle tag reports and print all available data."""
    for tag in access_report.getTagData():
        # Extract standard LLRP fields
        epc = HexFormatToStr(tag.get('EPC', b''))
        tid = HexFormatToStr(tag.get('TID', b'')) if 'TID' in tag else "N/A"
        antenna = tag.get('AntennaID', 'N/A')
        rssi = tag.get('PeakRSSI', 'N/A')
        first_seen = tag.get('FirstSeenTimestampUTC', 'N/A')
        last_seen = tag.get('LastSeenTimestampUTC', 'N/A')
        channel = tag.get('ChannelIndex', 'N/A')
        
        # Extract Impinj-specific fields (if available)
        impinj_data = tag.get('Custom', {}).get('Impinj', {})
        phase_angle = impinj_data.get('RFPhaseAngle', 'N/A')
        doppler = impinj_data.get('RFDopplerFrequency', 'N/A')

        print(f"Tag Report:")
        print(f"  EPC: {epc}")
        print(f"  TID: {tid}")
        print(f"  Antenna: {antenna}")
        print(f"  Peak RSSI: {rssi} dBm")
        print(f"  First Seen: {first_seen}")
        print(f"  Last Seen: {last_seen}")
        print(f"  Channel Index (Freq): {channel}")
        print(f"  Phase Angle: {phase_angle}")
        print(f"  Doppler Frequency: {doppler}")
        print("-" * 50)

def main():
    # Reader IP (replace with your reader's IP if different)
    reader_ip = '192.168.0.219'
    rospec_id = 123
    inventory_param_spec_id = 1234

    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)

    # Create and connect to the reader
    print(f"Connecting to reader at {reader_ip}...")
    connector = LLRPConnector()
    try:
        connector.connect(reader_ip)
    except Exception as e:
        print(f"Failed to connect: {e}")
        sys.exit(-1)

    # Reset reader to factory defaults
    #response = connector.transact(SET_READER_CONFIG_Message(ResetToFactoryDefault=True))
    #if not response.success():
     #   print("Failed to reset reader config.")
      #  connector.disconnect()
       # sys.exit(-1)

    # Enable Impinj extensions (if supported)
    #response = connector.transact(IMPINJ_ENABLE_EXTENSIONS_Message(MessageID=1))
    #if not response.success():
     #   print("Warning: Impinj extensions not supported by this reader.")
    #else:
     #   print("Impinj extensions enabled.")

    # Define ROSpec with all data fields enabled
    rospec = ADD_ROSPEC_Message(Parameters=[
        ROSpec_Parameter(
            ROSpecID=rospec_id,
            CurrentState=ROSpecState.Disabled,
            Parameters=[
                ROBoundarySpec_Parameter(
                    Parameters=[
                        ROSpecStartTrigger_Parameter(ROSpecStartTriggerType=ROSpecStartTriggerType.Immediate),
                        ROSpecStopTrigger_Parameter(ROSpecStopTriggerType=ROSpecStopTriggerType.Null),
                    ]
                ),
                AISpec_Parameter(
                    AntennaIDs=[0],  # Use all antennas (0 means all in some readers)
                    Parameters=[
                        AISpecStopTrigger_Parameter(AISpecStopTriggerType=AISpecStopTriggerType.Null),
                        InventoryParameterSpec_Parameter(
                            InventoryParameterSpecID=inventory_param_spec_id,
                            ProtocolID=AirProtocols.EPCGlobalClass1Gen2,
                        ),
                        # Optional: Add TID read operation (requires access command)
                        C1G2Read_Parameter(
                            OpSpecID=1,
                            AntennaID=0,
                            MB=1,  # Memory bank 1 = TID
                            WordPtr=0,
                            WordCount=4,  # Read 4 words (adjust based on tag)
                            AccessPassword=0,
                        ),
                    ]
                ),
                ROReportSpec_Parameter(
                    ROReportTrigger=ROReportTriggerType.Upon_N_Tags_Or_End_Of_AISpec,
                    N=1,  # Report every tag read
                    Parameters=[
                        TagReportContentSelector_Parameter(
                            EnableAntennaID=True,
                            EnableChannelIndex=True,
                            EnablePeakRSSI=True,
                            EnableFirstSeenTimestamp=True,
                            EnableLastSeenTimestamp=True,
                            EnableTagSeenCount=True,
                            EnableROSpecID=True,
                            EnableSpecIndex=True,
                            EnableInventoryParameterSpecID=True,
                            EnableCRC=True,
                            EnablePCBits=True,
                            EnableAccessSpecID=True,
                        ),
                        # Impinj-specific extensions
                        ImpinjTagReportContentSelector_Parameter(
                            EnableRFPhaseAngle=True,
                            EnablePeakRSSI=True,  # Redundant but included for clarity
                            EnableRFDopplerFrequency=True,
                        ),
                    ]
                ),
            ]
        ),
    ])

    # Clean up any existing ROSpec
    connector.transact(DELETE_ROSPEC_Message(ROSpecID=rospec_id))

    # Add and enable the ROSpec
    response = connector.transact(rospec)
    if not response.success():
        print(f"Failed to add ROSpec: {response}")
        connector.disconnect()
        sys.exit(-1)

    response = connector.transact(ENABLE_ROSPEC_Message(ROSpecID=rospec_id))
    if not response.success():
        print(f"Failed to enable ROSpec: {response}")
        connector.disconnect()
        sys.exit(-1)

    # Add handler for tag reports
    connector.addHandler(RO_ACCESS_REPORT_Message, access_report_handler)

    # Start listener thread
    print("Starting listener... (Press Ctrl+C to stop)")
    connector.startListener()

    # Run indefinitely until interrupted
    try:
        while running:
            time.sleep(1)  # Keep main thread alive
    except KeyboardInterrupt:
        pass  # Handled by signal_handler

    # Cleanup
    connector.stopListener()
    connector.transact(DISABLE_ROSPEC_Message(ROSpecID=rospec_id))
    connector.transact(DELETE_ROSPEC_Message(ROSpecID=rospec_id))
    connector.disconnect()
    print("Disconnected from reader.")

if __name__ == '__main__':
    main()