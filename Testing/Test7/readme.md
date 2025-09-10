# Test 7 Documentation for R420 Reader

## OBJECTIVE:
Collect tag reads at each of 16 grid points to gather comprehensive RFID fingerprinting data within a 3x3 subgrid.
Size

## LOCATIONS:
7 ft x 7 ft square grid with reads inside a 3 ft x 3 ft square grid.
Coordinate System
(x, y) coordinates, with (0, 0) at the bottom-left corner.

### Antennas:
1. (0, 7)
2. (7, 7)
3. (7, 0)
4. (0, 0)
- Antennas on the ground

### Tags:
A single RFID tag was placed at each of 16 points on a 3x3 subgrid within the 7x7 ft area, with 1 ft spacing, centered in the grid. 
Names of the CSV correspond to their coordinates

#### Note:
Test7-Mystery.csv was to try to find a stray tag coming from outside the room

## PROCESS
- tag 3 inches above the ground
- 16 grid points in a static position (no motion).
- Python script inv_sllurp.py.
- 50 tag reads per antenna per location.

## Data Fields
Each CSV includes:

- timestamp: ISO-formatted timestamp (timezone-free, derived from LastSeenTimestampUTC).
- antenna: Antenna ID (1–4).
- rssi: Peak RSSI (dBm).
- epc: Electronic Product Code of the tag.
- phase_angle: Impinj RF phase angle (if available).
- channel_index: RF channel index.
- doppler_frequency: Impinj RF Doppler frequency (if available).

## Script Description
The Python script (inv_sllurp.py) is a command-line application that:
- Connects to the Speedway R420 reader at 192.168.0.219:5084 using the sllurp library.
- Runs a continuous inventory for a user-specified duration.
- Supports an optional EPC filter to include only specific tags in the CSV output (exact match, e.g., 303435fc8803d056d15b963f).
- Collects tag reads in real-time, buffering them and writing to CSV every 2 seconds.
- Saves data to a CSV file named Test7-.csv (e.g., Test7-1.csv for (2, 5)).
- Logs tag reads and errors to the console for monitoring.
- Supports manual interruption (Ctrl+C) to stop inventory and save remaining data.

## SAMPLE OUTPUT:

CSV Files: 16 files (Test7-1.csv to Test7-16.csv), one per grid point, each containing at least 50 reads per antenna.
Example (Test7-1.csv for (2, 5)):timestamp,antenna,rssi,epc,phase_angle,channel_index,doppler_frequency
2025-05-06T10:00:05.123456,1,-35,303435fc8803d056d15b963f,3200,14,250
2025-05-06T10:00:05.178901,2,-68,303435fc8803d056d15b963f,2400,17,-1000
2025-05-06T10:00:05.185234,3,-62,303435fc8803d056d15b963f,1100,14,500
2025-05-06T10:00:05.191567,4,-72,303435fc8803d056d15b963f,800,14,200
...


Each file corresponds to a specific grid point (see TAG LOCATIONS for mapping).

Console Output
2025-05-06 10:05:08,123 INFO: Starting cycle 1
2025-05-06 10:05:08,127 INFO: Running command (attempt 1): sllurp inventory 192.168.0.219 -a 0 -X 0 --impinj-reports
2025-05-06 10:05:10,135 DEBUG: Raw output: 2025-05-06 10:05:08,183 sllurp.llrp: INFO: will reset reader state on connect
2025-05-06 10:05:08,183 sllurp.llrp: INFO: will start inventory on connect
2025-05-06 10:05:08,183 sllurp.llrp: INFO: Enabling Impinj extensions
2025-05-06 10:05:08,183 sllurp.llrp: INFO: using antennas: [0]
2025-05-06 10:05:08,183 sllurp.llrp: INFO: transmit power: {0: 0}
2025-05-06 10:05:08,187 sllurp.llrp: INFO: connected to 192.168.0.219 (:5084)
2025-05-06 10:05:08,202 sllurp.llrp: INFO: using reader mode: None
2025-05-06 10:05:08,213 sllurp.llrp: INFO: stopping politely
2025-05-06 10:05:10,135 sllurp.verb.inventory: INFO: saw tag(s): [{'AntennaID': 1, 'ChannelIndex': 41, 'EPC': b'303435fc8803d056d15b963f', 'EPC-96': b'303435fc8803d056d15b963f', 'FirstSeenTimestampUTC': 1745881648793145, 'ImpinjPeakRSSI': -6850, 'ImpinjRFDopplerFrequency': 3984, 'ImpinjRFPhaseAngle': 2624, 'LastSeenTimestampUTC': 1745881649932476, 'PeakRSSI': -68, 'TagSeenCount': 8}, ...]
2025-05-06 10:05:10,135 sllurp.llrp: INFO: starting inventory
2025-05-06 10:05:10,136 INFO: Parsed 4 tags
2025-05-06 10:05:10,136 INFO: Logged 4 tags to CSV (EPC filter: 303435fc8803d056d15b963f)
