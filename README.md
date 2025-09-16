# RFID Localization Prediction Repository

## Purpose
The purpose of this repository is to use RFID technology to estimate the location of a RFID tag. Make sure to read Lessons Learned and Physics Informaed RFID Models at the bottom.

## Hardware Used
- Impinj R420 Reader
- Antennas:
    - Antenna 1: S9028PCLJ (LHCP, 902–928 MHz)
    - Antenna 2: S8658PL (RHCP, 865–868 MHz)
    - Antenna 3: S9028PCRJ (RHCP, 902–928 MHz)
    - Antenna 4: S9028PCLJ (LHCP, 902–928 MHz)
- Cables:
    - Laird P/N 092463
    - 10 ft extension Shireen RFC 195 Low Loss 50 ohms 40/24
- RFID tags (various EPCs)

## Software Used
- Reader website
    - connect to reader network (or use ethernet cable)
    - go to http://speedwayr-12-77-b9.local
    - user: root
    - password: impinj
    used for installing software on reader (like speedway connect) and viewing reader information. Make sure speedway connect is installed before next step.
- Speedway Connect
    https://support.atlasrfidstore.com/article/13-using-speedway-connect
    - connect to reader network (or use ethernet cable)
    - go to https://speedwayr-12-77-b9.local
    - user: root
    - password: impinj
    used for connection, configuration, and HTTP post (https://support.atlasrfidstore.com/article/62-using-http-post-in-speedway-connect). For extra configuration and info, ssh can be used to access the reader (ssh root@192.168.0.219 pass: impinj)
- Python
    1. Make sure Python is installed:
    ```bash
    python3 --version
    ```
    2. Create virtual environment
    ```bash
    python3 -m venv venv
    ```
    3. Activate virtual environment
    - macOS:
    ```bash
    source venv/bin/activate
    ```
    4. Install dependencies
    ```bash
    pip3 install pandas numpy fastapi pydantic uvicorn scipy matplotlib sllurp
    ```
    5. Deactivate dependencies
    ```bash
    deactivate
    ```
- sllurp (https://github.com/sllurp/sllurp)
    
    How to Use
    1. Make sure sllurp is installed
    ```bash 
    pip show sllurp
    ```
    2. Using sllurp
    ```bash
    sllurp --help
    ```
    This will show available sllurp commands. For more config info see /Inventory/Reader_Settings.md

    3. example command:
    ```bash
    sllurp log 192.168.0.219 -o example.csv -t 60 -a 0 -X 0 --impinj-reports
    ```

- VS Code
- MS Excel

## Brief Description of Folders
### Inventory
- Mainly experimental interfacing python with sllurp or HTTP post to get reads and manipulate output.
### Locate
- Used for location prediction experiments using various physical methods
### Models
- Holds the training and application scripts for AI location prediction
### Testing
- Holds all testing data collected during experiments

## Note on PDF Documents
The three documents:
- Proposal for Enhanced RFID and Data Management System
- Review of RFID system early testing
- RFID Project Overview
- Physics Informed RFID Models

are all mostly early documentation. This is to show the progression of the project overtime. These documents are outdated and should not be used for the continuation of the project, but only to show where the project has been to avoid repeated mistakes.

**EXCEPTION:**
- **Physics Informed RFID Models**
    - Contains distance equations
    - Use RSSI path loss first
    - Then use Phase-based ranging

## Lessons Learned
This section is to sum up the lessons learned from this project and to hopefully promote the current prioities at the time of writing this. 
### Successes:
- Run inventory
- Log data in real time
- Configure RFID reader
- Alternatively post to API endpoint
- Calculate distance using RSSI in one dimension

### Future Improvements
- RSSI two dimensional localization prediction
    - Use channel index (unique RSSI grouping per frequency)
    - Use Kalman Filter (smoothing)
- Figure out phase angle
    - Use for more precise localization
    - Find patterns in test data
- Figure out doppler frequency
- Better Antennas
- No interference in testing room

### Steps to Success
- Calculate location using RSSI in 2 dimensions
- Refine location with phase angle 
- Reintroduce AI prediction

