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

## Data Collection:
- HTTP Post 
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
