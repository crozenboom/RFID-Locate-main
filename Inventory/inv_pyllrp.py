import pyllrp

# Connect to the reader
reader = pyllrp.LLRPReader(host="169.254.1.1", port=14150)
reader.connect()

# Start inventory
response = reader.start_inventory()
tags = response.get_tags()

for tag in tags:
    print(f"EPC: {tag.epc}, Antenna: {tag.antenna_id}")

# Stop inventory
reader.stop_inventory()
reader.disconnect()