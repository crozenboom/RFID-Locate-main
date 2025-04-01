import pyllrp
from pyllrp import Message, Parameter, LLRPError

# Reader settings
READER_IP = "192.168.0.219"
PORT = 5084  # Default LLRP port for Impinj readers (not 14150)

# Callback function to handle tag reports
def tag_report_callback(message):
    if message.MsgType == 'RO_ACCESS_REPORT':
        for tag_report in message.TagReportData:
            epc = tag_report.EPCData.hex() if tag_report.EPCData else 'Unknown'
            antenna = tag_report.AntennaID if tag_report.AntennaID is not None else 'Unknown'
            print(f"EPC: {epc}, Antenna: {antenna}")

# Connect to the reader
connection = pyllrp.ClientConnection(host=READER_IP, port=PORT)

try:
    # Connect and add a callback for incoming messages
    connection.connect()
    connection.addMessageCallback(tag_report_callback)

    # Reset the reader configuration
    reset_msg = Message('SET_READER_CONFIG', {
        'ResetToFactoryDefault': True,
        'MessageID': 1
    })
    response = connection.transact(reset_msg)
    if response.MsgType != 'SET_READER_CONFIG_RESPONSE' or not response.Success:
        raise LLRPError("Failed to reset reader configuration")

    # Create an RO_SPEC to define the inventory operation
    ro_spec = Parameter('ROSpec', {
        'ROSpecID': 1,
        'Priority': 0,
        'CurrentState': 'Disabled',
        'ROBoundarySpec': {
            'ROSpecStartTrigger': {
                'ROSpecStartTriggerType': 'Immediate'
            },
            'ROSpecStopTrigger': {
                'ROSpecStopTriggerType': 'Null'
            }
        },
        'AISpec': {
            'AntennaIDs': [1, 2, 3, 4],  # Use all antennas
            'AISpecStopTrigger': {
                'AISpecStopTriggerType': 'Duration',
                'DurationTriggerValue': 5000  # 5000 ms (5 seconds)
            },
            'InventoryParameterSpec': {
                'InventoryParameterSpecID': 1,
                'ProtocolID': 'EPCGlobalClass1Gen2'
            }
        },
        'ROReportSpec': {
            'ROReportTrigger': 'Upon_N_Tags_Or_End_Of_AISpec',
            'N': 1,
            'TagReportContentSelector': {
                'EnableROSpecID': False,
                'EnableSpecIndex': False,
                'EnableInventoryParameterSpecID': False,
                'EnableAntennaID': True,
                'EnableChannelIndex': False,
                'EnablePeakRSSI': True,
                'EnableFirstSeenTimestamp': False,
                'EnableLastSeenTimestamp': True,
                'EnableTagSeenCount': True,
                'EnableAccessSpecID': False
            }
        }
    })

    # Add the RO_SPEC to the reader
    add_ro_spec_msg = Message('ADD_ROSPEC', {
        'MessageID': 2,
        'ROSpec': ro_spec
    })
    response = connection.transact(add_ro_spec_msg)
    if response.MsgType != 'ADD_ROSPEC_RESPONSE' or not response.Success:
        raise LLRPError("Failed to add RO_SPEC")

    # Enable the RO_SPEC
    enable_ro_spec_msg = Message('ENABLE_ROSPEC', {
        'MessageID': 3,
        'ROSpecID': 1
    })
    response = connection.transact(enable_ro_spec_msg)
    if response.MsgType != 'ENABLE_ROSPEC_RESPONSE' or not response.Success:
        raise LLRPError("Failed to enable RO_SPEC")

    # Start the inventory by enabling the RO_SPEC (already set to start immediately)
    print("Starting inventory...")
    time.sleep(5)  # Wait for 5 seconds to collect tag reports

    # Stop the inventory
    stop_ro_spec_msg = Message('STOP_ROSPEC', {
        'MessageID': 4,
        'ROSpecID': 1
    })
    response = connection.transact(stop_ro_spec_msg)
    if response.MsgType != 'STOP_ROSPEC_RESPONSE' or not response.Success:
        raise LLRPError("Failed to stop RO_SPEC")

finally:
    # Disconnect from the reader
    print("Disconnecting from reader...")
    connection.disconnect()