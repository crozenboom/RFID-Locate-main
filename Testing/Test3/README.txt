Test 3 Documentation for R420 Reader

OBJECTIVE:
To calibrate the R420 reader’s RFID capabilities by measuring the signal strength (RSSI), phase angle, and other relevant data at 20 different locations within a 6ft x 6ft square. This test will help assess the accuracy of the R420’s performance in estimating the position of RFID tags within a specified area.

TEST SETUP:
• Area Dimensions: 6ft x 6ft square
• Number of Test Locations: 20 locations (evenly spaced within the 6ft x 6ft area)
• Duration per Test: 10-second interval at each location, cycling through antennas.
• Reader: R420 Impinj Reader
• Antennas: [List antenna IDs or configurations used]
• Reader IP Address: 192.168.0.219 (or specific address)
• Port: 5084
• Power Setting: Max power for all antenna transmissions during the test
• Tag EPC: The EPC of the used tag was 333030383333623264646439303134303030303030303030

CSV FILE FORMAT:
Each CSV file generated for each test location will contain the following columns:
Column Name		Description
Timestamp		Time of the reading (in UTC format)
EPC			Electronic Product Code of the RFID tag
AntennaID		Antenna ID number where the tag was detected
RSSI			Received Signal Strength Indicator (RSSI)
PhaseAngle		Phase angle of the detected signal
Frequency		Channel frequency of the tag reading
DopplerFrequency	Doppler frequency 

LOCATIONS:
• Coordinates: Each test location will have known (x, y) coordinates within the 6ft x 6ft area.
• Location Format: (x, y) values in feet.
• Spacing: Locations are evenly spaced to cover the entire square area.

Antennas (x, y):
1: (0, 3)
2: (3. 6)
3: (6, 3)
4: (3, 0)

Tags (x, y):
Names of the CSV correspond to their coordinates


Test Process:
1. Prepare the Reader and Antennas:
• Ensure the R420 reader is connected to the testing machine via the appropriate network.
• Set up antennas at predefined locations within the 6ft x 6ft test area.
• Configure the reader to use max power for all antenna transmissions during the test.
2. Tag Placement:
• Place an RFID tag at each of the 20 locations on the square. The tag should remain static during the test interval.
3. Data Collection:
• For each location, start a 10-second data collection interval, during which antennas will cycle every second.
• The reader will record the tag’s EPC, RSSI, and other signal data for each antenna that detects the tag.
4. Data Storage:
• Each location’s data will be saved in a CSV file.
• Example file name: Test1-2.csv for the first test at location 2.
5. Repeat for All 20 Locations:
• Complete the test for all 20 predefined locations, ensuring that data for each test is stored in separate CSV files.

Data Analysis:
1. Tag Location:
• Use the (x, y) coordinates of each location to match the corresponding data in the CSV files.
2. RSSI Data:
• Analyze the RSSI values from each antenna to assess signal strength at each location.
• Look for patterns or anomalies in the RSSI data across different locations.
3. Error Estimation:
• Compare the known (x, y) tag locations with the estimated tag positions (based on the RSSI and phase angle data).
• Calculate the errors in position estimation.
4. Antenna Performance:
• Evaluate the performance of each antenna based on the tag’s ability to be detected and the accuracy of RSSI readings at various locations.

Expected Output:
• A series of 20 CSV files with the recorded data for each of the 20 test locations.
• Analysis reports comparing the actual tag locations with estimated positions based on RSSI and phase angle data.
• Performance metrics for the reader and antennas used during the test.

Notes:
• Ensure the reader is connected before starting the test.
• Use a consistent method to mark the (x, y) locations within the square to ensure accuracy.
• If any anomalies or issues arise during the test, document them for further analysis.
• Keep test area clear of metal objects.
