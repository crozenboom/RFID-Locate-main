# Inventory
This folder is fairly obsolete, as using the edited sllurp log command from the CLI replaces any of the code in this folder's functionality. The scripts in this folder shows the progression of how gathering data was done earlier on in the project.

## Reader_Settings.md
This is an important file for reader configuration. Use this file to set sllurp to appropriate settings in llrp.py

## fetch.py
- Simple Flask server listening on `/rfid` POST endpoint.  
- Prints raw POST request content to console.  
- Responds with HTTP 200 OK.  
- Runs on all network interfaces at port 5050.  

## inv_sllurp.py
- Connects to an RFID reader using `sllurp`.  
- Runs inventory cycles and logs tag data (EPC, RSSI, antenna, phase, channel, doppler, timestamp) into CSV.  
- Performs socket checks to ensure the reader is reachable.  
- Retries failed cycles automatically.  
- Optional EPC filtering for selective logging.  
- Can run for a user-specified duration or indefinitely.  
- Provides detailed logging of each cycle and parsed tag data.  

## inv_sllurp2.0.py
- Runs RFID inventory cycles with `sllurp`.  
- Expands each tag's `TagSeenCount` into individual reads.  
- Evenly distributes timestamps across expanded reads.  
- Logs results (EPC, antenna, RSSI, phase, channel, doppler, timestamp) to CSV.  
- Allows users to configure cycle duration, total runtime, and optional EPC filters.  
- Handles retries and checks reader connectivity.  
- Provides detailed logging throughout execution.  

## inventory_post.py
- Runs `sllurp inventory` command against an RFID reader for a set duration.  
- Parses command output to extract: antenna ID, EPC, timestamps, RSSI, phase angle, Doppler frequency.  
- Expands each tag read according to `TagSeenCount`.  
- Logs structured data into `rfid_inventory.csv` with headers.  
- Manages process timeouts and ensures orderly termination.  
- Saves tag information for later analysis.  

## parse.py
- Flask server listening on `/rfid` POST endpoint.  
- Parses URL-encoded RFID data from incoming requests.  
- Extracts reader metadata (name, MAC address).  
- Processes tag reads by pairing field names with values.  
- Prints parsed tag data to console.  
- Responds with HTTP 200 OK to confirm receipt.  

## rfid_reader.py
- Custom RFID reader client using LLRP protocol over TCP.  
- Configures reader with messages: GET_READER_CAPABILITIES, SET_READER_CONFIG, DELETE_ALL_ROSPECS, etc.  
- Listens for RO_ACCESS_REPORT messages containing tag reads.  
- Parses each tag read for: EPC, timestamp, antenna ID, RSSI, phase angle, channel index, Doppler frequency.  
- Uses a queue for reads, and a separate CSV writer thread to batch-save data.  
- Handles retries, timeouts, and errors gracefully.  
- Supports Ctrl+C for clean shutdown.  
- Produces timestamped CSV files for analysis.  

## writetocsv.py
- Flask-based RFID data collector listening on `/rfid` POST endpoint.  
- Parses reader metadata (name, MAC address) and tag read fields/values.  
- Adds timestamps and stores all reads in memory.  
- Prints all tag reads to console for monitoring.  
- Writes data to CSV when program ends:  
  - On Ctrl+C  
  - After user-defined run duration  
  - When shutdown timer triggers  
- Optional EPC filtering before CSV export.  
- Useful for logging tag reads from multiple readers in a structured, time-limited way.  

