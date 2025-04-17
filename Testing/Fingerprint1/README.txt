RFID Fingerprinting Test


Overview:

This project conducts an RFID fingerprinting test using an Impinj Speedway RFID reader with Speedway Connect software and a Flask-based Python script (100Test.py) to collect tag reads. The test was performed on a 6 ft by 6 ft grid with four antennas at fixed locations. RFID tags were placed at 16 grid points, and data was collected until each antenna recorded at least 100 tag reads. To ensure reliable detection, tags were spun randomly at each point to create motion, enhancing the antennas' ability to pick up signals.


Grid Layout:

Size: 6 ft x 6 ft square grid.
Coordinate System: (x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations:
Antenna 1: (0, 3)
Antenna 2: (3, 6)
Antenna 3: (6, 3)
Antenna 4: (3, 0)


Tag Locations:

Tags were placed at 16 points on the grid, forming a 4x4 subgrid with 2 ft spacing:
(0, 6)
(2, 6)
(4, 6)
(6, 6)
(0, 4)
(2, 4)
(4, 4)
(6, 4)
(0, 2)
(2, 2)
(4, 2)
(6, 2)
(0, 0)
(2, 0)
(4, 0)
(6, 0)


Data Collection:

Objective: Collect at least 100 tag reads per antenna (ports 1, 2, 3, 4).
Process:
Tags were placed at each of the 16 grid points, one at a time.
At each point, the tag was spun randomly to create motion, improving detection by the antennas.
The RFID reader, configured via Speedway Connect, sent tag data to a Flask server (100Test.py) via HTTP POST requests.
The server ran until each antenna recorded at least 100 reads, then saved the data to a CSV file.
Data collection was performed across all 16 points to ensure comprehensive coverage.


Data Fields: Included antenna_port, epc, first_seen_timestamp, mac_address, peak_rssi, reader_name, tid, timestamp, and user_memory.


Software Requirements:

Python: 3.8 or higher
Flask: Install via pip install flask
Impinj Speedway RFID Reader: Configured with Speedway Connect software
RFID Tags: Compatible with the reader
Network: Reader must be able to send HTTP POST requests to the Flask server (port 5050)


Script Description:

The Python script (100Test.py) is a Flask application that:
Listens for HTTP POST requests from Speedway Connect at http://<server-ip>:5050/rfid.
Parses URL-encoded tag data, removing quotation marks for clean output.
Tracks the number of tag reads per antenna (ports 1, 2, 3, 4) until each has at least 100 reads.
Prompts the user for an optional EPC filter to include only specific tags in the CSV output (supports partial, case-insensitive matches).
Saves all collected data to a single CSV file with filtered results when the 100-read threshold is met or on manual interruption (Ctrl+C).
Prints parsed tag data and antenna read counts to the console for real-time monitoring.


Setup:

-Configure the RFID reader in Speedway Connect:
Protocol: HTTP POST
URL: http://<server-ip>:5050/rfid (e.g., http://192.168.0.216:5050/rfid)
Content Type: Application/x-www-form-urlencoded
Field Names: antenna_port,epc,first_seen_timestamp,mac_address,peak_rssi,reader_name,tid,user_memory
Trigger Mode: Immediate
Reader Mode: AutoSet

-Apply settings and ensure the reader status is green (enabled).
-Place a tag at one of the 16 grid points, spinning it to induce motion. Repeat for each point as needed to accumulate reads.
-Run the Script: python 100Test.py
-Prompt: “Enter EPC filter (leave blank for no filter):”
-Enter an EPC (e.g., 303211) to filter CSV output to tags containing that EPC, or press Enter to include all tags.
-The script runs until each antenna (1, 2, 3, 4) has at least 100 reads, printing tag data and antenna counts to the console.
-Or press Ctrl+C to stop early and save collected data to a CSV.


Output:

A CSV file is created when:
All antennas reach at least 100 reads, or
The user presses Ctrl+C.


The CSV includes fields like antenna_port, epc, timestamp, etc., filtered by the EPC input if provided.
Example CSV (with EPC filter 303211):antenna_port,epc,first_seen_timestamp,mac_address,peak_rssi,reader_name,tid,timestamp
1,303211468E03E29B022504C1,1744748056184199,00:16:25:12:77:b9,-65,Impinj RFID Reader,E2801191200070852BEA030D,2025-04-15 14:12:01

Console output includes real-time tag reads and antenna counts (e.g., Antenna read counts: {'1': 50, '2': 30, '3': 20, '4': 10}).



Notes:

Antenna Requirement: The script requires all four antennas (1, 2, 3, 4) to reach 100 reads before stopping. If fewer antennas are connected, modify antenna_counts in 100Test.py to include only active ports (e.g., {'1': 0, '2': 0} for two antennas).
EPC Filter: Supports partial, case-insensitive matches (e.g., 303211 matches 303211468E03E29B022504C1). Leave blank to include all tags.
Motion: Spinning tags at each grid point ensures reliable detection by varying tag orientation, as RFID signals are sensitive to positioning.
Troubleshooting:
No Data Received:
Verify Speedway Connect’s URL matches the server’s IP (e.g., http://192.168.0.216:5050/rfid).
Ensure the reader is enabled (green status) and tags are in range.
Test the Flask endpoint:curl -X POST -d "test=data" http://<server-ip>:5050/rfid

Expect console output: Received data: test=data.


Network Issues:
Confirm the server’s IP (e.g., 192.168.0.216) using ifconfig (macOS/Linux) or ipconfig (Windows).
Check if port 5050 is open (sudo lsof -i :5050 on macOS) and not blocked by a firewall.


Antenna Reads:
Monitor console output for Antenna read counts. If some antennas aren’t incrementing, ensure they’re connected and tags are in range.





License
This project is intended for research purposes and is not licensed for commercial use. Ensure compliance with Impinj Speedway Connect licensing for production environments.
Contact
For issues, modifications, or questions about the test setup, contact the project maintainer or refer to the comments in 100Test.py for implementation details.
