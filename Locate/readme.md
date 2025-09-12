# Locate
These scripts use a combination of distance equations for each antenna and trilateration using at least 3 antennas' distance values to output a coordinate on the (x, y) plane.

## FileLocate_Channel.py
- Reads RFID RSSI data from CSV and estimates 2D tag coordinates.  
- Calculates distances per antenna using RSSI, RSSI0, and path-loss exponent.  
- Processes data in chunks; requires ≥3 antennas per group to predict location.  
- Uses least-squares optimization to compute X, Y coordinates.  
- Saves results to `Coordinate Prediction.csv`.  
- Generates static plot (`static_plot.png`) and animated plot (`animation.gif`).  
- Logs progress and warnings to console.

## FileLocate_Kalman.py
- Implements 1D Kalman filter to smooth RSSI readings per antenna.  
- Reads CSV of raw RSSI and antenna data.  
- Filters RSSI row-by-row and calculates distances using path loss model.  
- Requires at least 3 antennas with data to compute (X, Y) location via least squares.  
- Outputs results per row/group to `Coordinate Prediction.csv`.  
- Generates static plot (`static_plot.png`) of predicted locations with antenna markers.  
- Generates animated plot (`animation.gif`) showing path progression over time.  
- Handles failed convergence or insufficient data gracefully. 

## Interpolate.py
- Interpolates RSSI values based on given power levels using UnivariateSpline.  
- Generates a smooth curve for power levels from 10 dBm to 32.75 dBm at 0.25 dBm increments.  
- Outputs interpolated values with constant distance (6 in) to `interpolated_rssi.csv`.  
- Plots original RSSI data and interpolated curve for visualization.  
- Includes markers for original data points and a smooth line for interpolated values.  
- Labels axes, adds grid, legend, and title for clarity. 

## LiveLocate.py
- FastAPI server receiving RFID RSSI data via `/rfid_data` POST endpoint.  
- Buffers RSSI by antenna and 1-second time windows.  
- Uses calibration CSV to compute P(d₀) per antenna.  
- Calculates distances from RSSI using path-loss model.  
- Performs trilateration (least-squares) if ≥3 antennas have data.  
- Appends computed (x, y) positions with timestamp to `tag_positions.csv`.  
- Prints position info to console.  
- Runs on all network interfaces at port 8000 using uvicorn. 

## noideaifthiswillwork,py
- Processes RFID phase-angle CSVs in chunks to compute distances per antenna.  
- Uses linear fit of unwrapped phase vs frequency for distance estimation.  
- Filters out antennas with insufficient reads, duplicate frequencies, or invalid distances.  
- Performs trilateration with least-squares if ≥3 antennas provide valid distances.  
- Saves predicted (X, Y) per group to `maybelocation.csv`.  
- Generates static plot (`static_plot.png`) and animated GIF (`animation.gif`) of predicted tag locations.  
- Prints progress, distances, and status messages to console. 

## TOF.py
- Loads RFID CSV with Time of Flight (ToF) per antenna.  
- Skips first 6 comment lines, applies custom headers, and parses timestamps.  
- Computes time differences between consecutive readings per antenna.  
- Calculates scaled distance assuming signal travels slower than light and converts to inches.  
- Prints average distance per antenna to console. 

## Updated_Locate.py
- Loads reference RSSI CSV and user power/EPC input.  
- Reads RFID data CSV, optionally filtered by EPC.  
- Computes average RSSI per antenna.  
- Calculates distance using standard RSSI path loss formula.  
- Prints distances per antenna to console. 