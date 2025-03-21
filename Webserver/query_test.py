from rfidpy import ImpinjReader # type: ignore

# Initialize connection to reader
reader = ImpinjReader("169.254.1.1")

# Connect to the reader
reader.connect()

# Start the inventory (with detailed reports, including antenna info)
tags = reader.query_inventory(antenna=True)

# Print out the tags and the antenna data
for tag in tags:
    print(f"Tag EPC: {tag.epc}, Antenna ID: {tag.antenna_id}, RSSI: {tag.rssi}")

# Disconnect from the reader
reader.disconnect()