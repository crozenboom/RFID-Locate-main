# RFID Fingerprinting Test

## Overview:
- 6x6 ft grid
- 16 grid points
- 4 antennas
- at least 100 reads per test
- spinning tags

## Grid Layout:
Size: 6 ft x 6 ft square grid.
Coordinate System: (x, y) coordinates, with (0, 0) at the bottom-left corner.
Antenna Locations:
Antenna 1: (0, 3)
Antenna 2: (3, 6)
Antenna 3: (6, 3)
Antenna 4: (3, 0)

## Tag Locations:

The following table maps each fingerprint test file to its corresponding location coordinates.

| File Name           | Location |
|---------------------|----------|
| Fingerprint1-1.csv  | (0, 6)   |
| Fingerprint1-2.csv  | (2, 6)   |
| Fingerprint1-3.csv  | (4, 6)   |
| Fingerprint1-4.csv  | (6, 6)   |
| Fingerprint1-5.csv  | (0, 4)   |
| Fingerprint1-6.csv  | (2, 4)   |
| Fingerprint1-7.csv  | (4, 4)   |
| Fingerprint1-8.csv  | (6, 4)   |
| Fingerprint1-9.csv  | (0, 2)   |
| Fingerprint1-10.csv | (2, 2)   |
| Fingerprint1-11.csv | (4, 2)   |
| Fingerprint1-12.csv | (6, 2)   |
| Fingerprint1-13.csv | (0, 0)   |
| Fingerprint1-14.csv | (2, 0)   |
| Fingerprint1-15.csv | (4, 0)   |
| Fingerprint1-16.csv | (6, 0)   |

## Data Collection:
- HTTP Post via Speedway Connect
- Data Fields: 
    - antenna_port
    - epc
    - first_seen_timestamp
    - mac_address
    - peak_rssi
    - reader_name
    - tid 
    - timestamp
    - user_memory
