Test 8 Documentation for R420 Reader
OBJECTIVE
Collect tag reads at each of 16 grid points to gather comprehensive RFID fingerprinting data within a 3x3 subgrid.
Size
15 ft x 15 ft square grid with reads inside a 13 ft x 13 ft square grid.
Coordinate System
(x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations

Antenna 1: (0, 0)
Antenna 2: (0, 15)
Antenna 3: (15, 15)
Antenna 4: (15, 0)

TAG LOCATIONS
Names of the CSV correspond to their coordinates

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

CLI command sllurp log 192.168.0.219 -a 1,2,3,4 -X 0 -t 20 -r -o data.csv --impinj-reports


SETUP

Verify reader status is active and LLRP port 5084 is open: telnet 192.168.0.219 5084


Set Up the Environment

Create and activate a virtual environment:python3 -m venv env
source env/bin/activate


Install sllurp:pip install sllurp


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
