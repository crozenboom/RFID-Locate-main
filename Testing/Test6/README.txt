Test 6 Documentation for R420 Reader
OBJECTIVE
Collect tag reads at each of 36 grid points to gather comprehensive RFID fingerprinting data.

Size: 7 ft x 7 ft square grid.
Coordinate System: (x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations:
Antenna 1: (0, 7)
Antenna 2: (7, 7)
Antenna 3: (7, 0)
Antenna 4: (0, 0)



TAG LOCATIONS
A single RFID tag was placed at each of 36 points on a 6x6 subgrid within the 7x7 ft area, with 1 ft spacing. 
Names of the CSV correspond to their coordinates.


PROCESS

A single RFID tag was placed flat, 3 inches above the ground, at each of the 36 grid points in a static position (no motion).
The Impinj Speedway R420 reader was controlled using the sllurp library via the Python script inv_sllurp.py.
The script ran at each location, collecting at least 50 tag reads per antenna per location.
The reader cycled through all four antennas, positioned at the corners of the 7x7 ft grid, pointing toward the center.
The test was repeated for all 36 points, moving the tag to each new location (e.g., from (1, 6) to (2, 6), etc.) and running the script for each point.
Data was saved to individual CSV files named Test6-1.csv to Test6-36.csv, corresponding to the grid points listed above.

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
Saves data to a CSV file named Test6-<N>.csv (e.g., Test6-1.csv for (1, 6)).
Logs tag reads and errors to the console for monitoring.
Supports manual interruption (Ctrl+C) to stop inventory and save remaining data.

SETUP

Configure the RFID Reader:

Access the web interface (http://192.168.0.219, default credentials: admin/impinj).
Set:
Antennas: All 4 enabled (1–4).
Power: 20 dBm (or set via script with -X 2000).
Sensitivity: -80 dBm.
Search Mode: Dual Target or Max Throughput.
Dwell Time: ~100 ms per antenna (default or adjusted for ~50 reads/antenna).


Verify reader status is active and LLRP port 5084 is open:telnet 192.168.0.219 5084




Set Up the Environment:

Create and activate a virtual environment:python3 -m venv env
source env/bin/activate


Install sllurp:pip install sllurp




Place the Tag:

Position a single UHF RFID tag flat, 3 inches above the ground, at the first grid point (e.g., (1, 6) for Test6-1.csv).
Ensure the tag is static (no motion) and within the read range of the antennas (1–3 ft recommended).


Run the Script:

Execute:python -u inv_sllurp.py


Input:

EPC filter: Enter a specific EPC (e.g., 303435fc8803d056d15b963f) or press Enter for all tags.



OUTPUT

CSV Files: 36 files (Test6-1.csv to Test6-36.csv), one per grid point, each containing at least 50 reads per antenna.

Example (Test6-1.csv for (1, 6)):timestamp,antenna,rssi,epc,phase_angle,channel_index,doppler_frequency
2025-04-28T15:08:05.361755,1,-37,303435fc8803d056d15b963f,3408,14,273
2025-04-28T15:08:05.415440,2,-66,303435fc8803d056d15b963f,2536,17,-1093
2025-04-28T15:08:05.421268,3,-65,303435fc8803d056d15b963f,1184,14,566
2025-04-28T15:08:05.427535,4,-70,303435fc8803d056d15b963f,844,14,273
...


Each file corresponds to a specific grid point (see TAG LOCATIONS for mapping).


Console Output:

2025-04-28 17:05:08,553 INFO: Starting cycle 1
2025-04-28 17:05:08,557 INFO: Running command (attempt 1): sllurp inventory 192.168.0.219 -a 0 -X 0 --impinj-reports
2025-04-28 17:05:10,565 DEBUG: Raw output: 2025-04-28 17:05:08,613 sllurp.llrp: INFO: will reset reader state on connect
2025-04-28 17:05:08,613 sllurp.llrp: INFO: will start inventory on connect
2025-04-28 17:05:08,613 sllurp.llrp: INFO: Enabling Impinj extensions
2025-04-28 17:05:08,613 sllurp.llrp: INFO: using antennas: [0]
2025-04-28 17:05:08,613 sllurp.llrp: INFO: transmit power: {0: 0}
2025-04-28 17:05:08,617 sllurp.llrp: INFO: connected to 192.168.0.219 (:5084)
2025-04-28 17:05:08,632 sllurp.llrp: INFO: using reader mode: None
2025-04-28 17:05:08,643 sllurp.llrp: INFO: stopping politely
2025-04-28 17:05:08,658 sllurp.verb.inventory: INFO: saw tag(s): [{'AntennaID': 3,
  'ChannelIndex': 43,
  'EPC': b'300833b2ddd9014000000000',
  'EPC-96': b'300833b2ddd9014000000000',
  'FirstSeenTimestampUTC': 1745881649350205,
  'ImpinjPeakRSSI': -6800,
  'ImpinjRFDopplerFrequency': 2815,
  'ImpinjRFPhaseAngle': 2592,
  'LastSeenTimestampUTC': 1745881649612720,
  'PeakRSSI': -68,
  'TagSeenCount': 2},
 {'AntennaID': 1,
  'ChannelIndex': 41,
  'EPC': b'303435fc8803d056d15b963f',
  'EPC-96': b'303435fc8803d056d15b963f',
  'FirstSeenTimestampUTC': 1745881648793145,
  'ImpinjPeakRSSI': -6850,
  'ImpinjRFDopplerFrequency': 3984,
  'ImpinjRFPhaseAngle': 2624,
  'LastSeenTimestampUTC': 1745881649932476,
  'PeakRSSI': -68,
  'TagSeenCount': 8},
 {'AntennaID': 2,
  'ChannelIndex': 38,
  'EPC': b'303435fc8803d056d15b963f',
  'EPC-96': b'303435fc8803d056d15b963f',
  'FirstSeenTimestampUTC': 1745881648181617,
  'ImpinjPeakRSSI': -6000,
  'ImpinjRFDopplerFrequency': 2431,
  'ImpinjRFPhaseAngle': 2708,
  'LastSeenTimestampUTC': 1745881649996423,
  'PeakRSSI': -60,
  'TagSeenCount': 12},
 {'AntennaID': 3,
  'ChannelIndex': 38,
  'EPC': b'303435fc8803d056d15b963f',
  'EPC-96': b'303435fc8803d056d15b963f',
  'FirstSeenTimestampUTC': 1745881648234707,
  'ImpinjPeakRSSI': -3800,
  'ImpinjRFDopplerFrequency': 117,
  'ImpinjRFPhaseAngle': 1228,
  'LastSeenTimestampUTC': 1745881650047152,
  'PeakRSSI': -38,
  'TagSeenCount': 18},
 {'AntennaID': 4,
  'ChannelIndex': 41,
  'EPC': b'303435fc8803d056d15b963f',
  'EPC-96': b'303435fc8803d056d15b963f',
  'FirstSeenTimestampUTC': 1745881648756406,
  'ImpinjPeakRSSI': -6300,
  'ImpinjRFDopplerFrequency': -839,
  'ImpinjRFPhaseAngle': 2540,
  'LastSeenTimestampUTC': 1745881649880005,
  'PeakRSSI': -63,
  'TagSeenCount': 10}]
2025-04-28 17:05:08,658 sllurp.llrp: INFO: Impinj search mode? None
2025-04-28 17:05:08,658 sllurp.llrp: INFO: starting inventory

2025-04-28 17:05:10,566 INFO: Parsed 5 tags
2025-04-28 17:05:10,566 INFO: Logged 4 tags to CSV (EPC filter: 303435fc8803d056d15b963f)



Notes

Test Duration: Each run collects at least 50 reads per antenna. The script was run 36 times to cover all grid points.
EPC Filter: Requires an exact 24-character hexadecimal match (e.g., 303435fc8803d056d15b963f). Leave blank to include all tags.
Tag Placement: Tags were static, placed flat 3 inches above the ground. This differs from Test 4’s random motion, potentially affecting read rates due to fixed orientation.
Antenna Configuration: Antennas were placed on the ground at the corners of the 7x7 ft grid, pointing toward the center. Read rates vary by tag position (e.g., (1, 6) is closer to Antenna 1 at (0, 7) than Antenna 3 at (7, 0)).
CSV Naming: Files are explicitly labeled Test6-1.csv to Test6-36.csv, mapped to grid points (1, 6) to (6, 1) for clarity.




LLRP Port Issues:

If LLRP port 5084 not reachable, power cycle the reader (unplug PoE injector, wait 30 seconds, reconnect).
Check for IP conflicts: ping 192.168.0.219





License
This project is intended for research purposes and is not licensed for commercial use. Ensure compliance with Impinj Speedway R420 licensing for production environments.
Contact
For issues, modifications, or questions about the test setup, contact the project maintainer or refer to the comments in inv_sllurp.py for implementation details.
