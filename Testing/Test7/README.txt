Test 7 Documentation for R420 Reader
OBJECTIVE
Collect tag reads at each of 16 grid points to gather comprehensive RFID fingerprinting data within a 3x3 subgrid.
Size
7 ft x 7 ft square grid with reads inside a 3 ft x 3 ft square grid.
Coordinate System
(x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations

Antenna 1: (0, 7)
Antenna 2: (7, 7)
Antenna 3: (7, 0)
Antenna 4: (0, 0)

TAG LOCATIONS
A single RFID tag was placed at each of 16 points on a 3x3 subgrid within the 7x7 ft area, with 1 ft spacing, centered in the grid. The locations are numbered 1 to 16 as follows:

(2, 5) - Test7-1.csv
(3, 5) - Test7-2.csv
(4, 5) - Test7-3.csv
(5, 5) - Test7-4.csv
(2, 4) - Test7-5.csv
(3, 4) - Test7-6.csv
(4, 4) - Test7-7.csv
(5, 4) - Test7-8.csv
(2, 3) - Test7-9.csv
(3, 3) - Test7-10.csv
(4, 3) - Test7-11.csv
(5, 3) - Test7-12.csv
(2, 2) - Test7-13.csv
(3, 2) - Test7-14.csv
(4, 2) - Test7-15.csv
(5, 2) - Test7-16.csv

PROCESS

A single RFID tag was placed flat, 3 inches above the ground, at each of the 16 grid points in a static position (no motion).
The Impinj Speedway R420 reader was controlled using the sllurp library via the Python script inv_sllurp.py.
The script ran at each location, collecting at least 50 tag reads per antenna per location.
The reader cycled through all four antennas, positioned at the corners of the 7x7 ft grid, pointing toward the center.
The test was repeated for all 16 points, moving the tag to each new location (e.g., from (2, 5) to (3, 5), etc.) and running the script for each point.
Data was saved to individual CSV files named Test7-1.csv to Test7-16.csv, corresponding to the grid points listed above.

Data Fields
Each CSV includes:

timestamp: ISO-formatted timestamp (timezone-free, derived from LastSeenTimestampUTC).
antenna: Antenna ID (1–4).
rssi: Peak RSSI (dBm).
epc: Electronic Product Code of the tag.
phase_angle: Impinj RF phase angle (if available).
channel_index: RF channel index.
doppler_frequency: Impinj RF Doppler frequency (if available).

Software Requirements

Python: 3.6–3.10
sllurp: Install via pip install sllurp
Impinj Speedway R420 RFID Reader: Configured with web interface or script parameters
RFID Tags: UHF RFID tags compatible with the reader (e.g., EPCs 303435fc8803d056d15b963f, 300833b2ddd9014000000000)
Network: Reader must be reachable at IP 192.168.0.219 on port 5084 (LLRP)

Script Description
The Python script (inv_sllurp.py) is a command-line application that:

Connects to the Speedway R420 reader at 192.168.0.219:5084 using the sllurp library.
Runs a continuous inventory for a user-specified duration.
Supports an optional EPC filter to include only specific tags in the CSV output (exact match, e.g., 303435fc8803d056d15b963f).
Collects tag reads in real-time, buffering them and writing to CSV every 2 seconds.
Saves data to a CSV file named Test7-.csv (e.g., Test7-1.csv for (2, 5)).
Logs tag reads and errors to the console for monitoring.
Supports manual interruption (Ctrl+C) to stop inventory and save remaining data.

SETUP

Verify reader status is active and LLRP port 5084 is open: telnet 192.168.0.219 5084


Set Up the Environment

Create and activate a virtual environment:python3 -m venv env
source env/bin/activate


Install sllurp:pip install sllurp



Place the Tag

Position a single UHF RFID tag flat, 3 inches above the ground, at the first grid point (e.g., (2, 5) for Test7-1.csv).
Ensure the tag is static (no motion) and within the read range of the antennas (1–3 ft recommended).

Run the Script

Execute:python -u inv_sllurp.py


Input:
EPC filter: Enter a specific EPC (e.g., 303435fc8803d056d15b963f) or press Enter for all tags.



OUTPUT

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

Notes

Test Duration: Each run collects at least 50 reads per antenna. The script was run 16 times to cover all grid points.
EPC Filter: Requires an exact 24-character hexadecimal match (e.g., 303435fc8803d056d15b963f). Leave blank to include all tags.
Tag Placement: Tags were static, placed flat 3 inches above the ground, consistent with Test 6.
Antenna Configuration: Antennas were placed on the ground at the corners of the 7x7 ft grid, pointing toward the center. Read rates vary by tag position (e.g., (2, 5) is closer to Antenna 1 at (0, 7) than Antenna 3 at (7, 0)).
CSV Naming: Files are explicitly labeled Test7-1.csv to Test7-16.csv, mapped to grid points (2, 5) to (5, 2) for clarity.

LLRP Port Issues

If LLRP port 5084 is not reachable, power cycle the reader (unplug PoE injector, wait 30 seconds, reconnect).
Check for IP conflicts: ping 192.168.0.219

License
This project is intended for research purposes and is not licensed for commercial use. Ensure compliance with Impinj Speedway R420 licensing for production environments.
Contact
For issues, modifications, or questions about the test setup, contact the project maintainer or refer to the comments in inv_sllurp.py for implementation details.
