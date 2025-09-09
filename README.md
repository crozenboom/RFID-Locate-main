# RFID Localization Prediction Repository

## Purpose
The purpose of this repository is to use RFID technology to estimate the location of a RFID tag

## Hardware Used
- Impinj R420 Reader
- Antennas
    - Antenna 1: S9028PCLJ (LHCP, 902–928 MHz)
    - Antenna 2: S8658PL (RHCP, 865–868 MHz)
    - Antenna 3: S9028PCRJ (RHCP, 902–928 MHz)
    - Antenna 4: S9028PCLJ (LHCP, 902–928 MHz)
- F to F Coaxial Cables
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
    2. Create Virtual Environment
    ```bash
    python3 -m venv venv
    ```
    3. Activate Virtual Environment
    - macOS:
    ```bash
    source venv/bin/activate
    ```
    4. Install Dependencies
    ```bash
    pip install pandas numpy fastapi pydantic uvicorn scipy matplotlib sllurp
    ```
    5. Deactivate Dependencies
    ```bash
    deactivate
    ```
- sllurp (https://github.com/sllurp/sllurp)
    How to Install
- VS Code
- MS Excel

## Descriptions of Folders
- Inventory
    Mainly experimental interfacing python with sllurp or HTTP post to get reads and manipulate output.
- Locate


