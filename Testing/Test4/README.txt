Test 4 Documentation for R420 Reader

OBJECTIVE: 
Collect tag reads at each of the 9 grid points for 30 seconds per location.

Size: 6 ft x 6 ft square grid.
Coordinate System: (x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations:
Antenna 1: (0, 3)
Antenna 2: (3, 6)
Antenna 3: (6, 3)
Antenna 4: (3, 0)



TAG LOCATIONS:
Tags were placed and moved at 9 points on the grid, forming a 3x3 subgrid with 2 ft spacing:
Names of the CSV correspond to their coordinates


PROCESS:
A tag was placed at each grid point and moved randomly (e.g., waved or spun) to create motion, improving detection by the antennas.
The RFID reader, configured via Speedway Connect, sent tag data to a Flask server (writetocsv.py) via HTTP POST requests.
The server ran for 30 seconds at each location, saving the collected data to a CSV file.
The test was repeated for all 9 points to gather comprehensive fingerprinting data.


Data Fields: Included antenna_port, epc, first_seen_timestamp, mac_address, peak_rssi, reader_name, tid, and timestamp.

Software
Requirements

Python: 3.8 or higher
Flask: Install via pip install flask
Impinj Speedway RFID Reader: Configured with Speedway Connect software
RFID Tags: Compatible with the reader
Network: Reader must be able to send HTTP POST requests to the Flask server (port 5050)

Script Description
The Python script (writetocsv.py) is a Flask application that:

Listens for HTTP POST requests from Speedway Connect at http://<server-ip>:5050/rfid.
Parses URL-encoded tag data, removing quotation marks for clean output.
Collects tag reads for a fixed 30-second duration per run, prompted by the user.
Supports an optional EPC filter to include only specific tags in the CSV output (partial, case-insensitive matches).
Saves all collected data to a single CSV file (rfid_data_YYYYMMDD_HHMMSS.csv) with filtered results after each 30-second run or on manual interruption (Ctrl+C).
Prints parsed tag data to the console for real-time monitoring.

SETUP:

-Configure the RFID reader in Speedway Connect:
Protocol: HTTP POST
URL: http://<server-ip>:5050/rfid (e.g., http://192.168.0.216:5050/rfid)
Content Type: Application/x-www-form-urlencoded
Field Names: antenna_port,epc,first_seen_timestamp,mac_address,peak_rssi,reader_name,tid,user_memory
Trigger Mode: Immediate
Reader Mode: AutoSet

-Apply settings and ensure the reader status is green (enabled).
Place a tag at one of the 9 grid points (e.g., (1,5)) and move it randomly (e.g., wave or spin) to induce motion.
-Run the Script: python writetocsv.py
-The script runs for 30 seconds, collecting tag reads and printing data to the console.
-Repeat the process for each of the 9 grid points, moving the tag to the next location (e.g., (3,5), (5,5), etc.) and restarting the script.


OUTPUT:

A CSV file (rfid_data_YYYYMMDD_HHMMSS.csv) is created after each 30-second run or on Ctrl+C.
The CSV includes fields like antenna_port, epc, timestamp, etc., filtered by the EPC input if provided.
Example CSV (with EPC filter 303211):antenna_port,epc,first_seen_timestamp,mac_address,peak_rssi,reader_name,tid,timestamp,user_memory
1,303211468E03E29B022504C1,1744748056184199,00:16:25:12:77:b9,-65,Impinj RFID Reader,E2801191200070852BEA030D,2025-04-15 14:12:01,
...


Console output shows real-time tag reads (e.g., Reader: Impinj RFID Reader, MAC: 00:16:25:12:77:b9, Tag Read: {...}).



Notes

Test Duration: Each run lasts 30 seconds, controlled by the script’s timer. Repeat the script for each of the 9 grid points to cover all locations.
EPC Filter: Supports partial, case-insensitive matches (e.g., 303211 matches 303211468E03E29B022504C1). Leave blank to include all tags.
Tag Motion: Random movement (e.g., waving or spinning) at each grid point ensures reliable detection by varying tag orientation, as RFID signals are sensitive to positioning.
Antenna Coverage: All four antennas are active, but read rates vary by tag position (e.g., (1,5) is closer to antenna 2 at (3,6) than antenna 4 at (3,0)). Random motion helps balance reads.
Troubleshooting:
No Data Received:
Verify Speedway Connect’s URL matches the server’s IP (e.g., http://192.168.0.216:5050/rfid).
Ensure the reader is enabled (green status) and the tag is in range with sufficient motion.
Test the Flask endpoint:curl -X POST -d "test=data" http://<server-ip>:5050/rfid

Expect console output: Received data: test=data.


Network Issues:
Confirm the server’s IP (e.g., 192.168.0.216) using ifconfig (macOS/Linux) or ipconfig (Windows).
Check if port 5050 is open (sudo lsof -i :5050 on macOS) and not blocked by a firewall.


Low Read Rates:
Ensure the tag is moved actively (e.g., waved in multiple directions) at 6–12 inches above the ground to stay in the antennas’ read zones.
Check reader power (20–25 dBm recommended) in Speedway Connect’s Reader Settings.





License
This project is intended for research purposes and is not licensed for commercial use. Ensure compliance with Impinj Speedway Connect licensing for production environments.
Contact
For issues, modifications, or questions about the test setup, contact the project maintainer or refer to the comments in writetocsv.py for implementation details.
