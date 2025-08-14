import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import time
from datetime import datetime
import math
from scipy.optimize import least_squares

# FastAPI app
app = FastAPI()

# Calibration data storage
calibration_data = {}
d0 = 10.61  # Reference distance in feet (sqrt(15^2 / 2) for 15x15 ft room, tag at center)
n = 3  # Path loss exponent

# Antenna positions (in feet, 15x15 ft room)
antennas = {
    1: np.array([0, 0]),    # Bottom left
    2: np.array([0, 15]),   # Top left
    3: np.array([15, 15]),  # Top right
    4: np.array([15, 0])    # Bottom right
}

# Output CSV file
output_csv = "tag_positions.csv"

# Initialize output CSV
def init_output_csv():
    pd.DataFrame(columns=["timestamp", "x", "y"]).to_csv(output_csv, index=False)

# Pydantic model for incoming Sllurp data
class RFIDData(BaseModel):
    antenna_id: int
    rssi: float
    timestamp: float

# Load calibration CSV and compute P(d₀) for each antenna
def load_calibration_csv(file_path: str):
    try:
        df = pd.read_csv(file_path)
        # Assume CSV has columns: antenna_id, rssi
        p_d0 = df.groupby("antenna_id")["rssi"].mean().to_dict()
        calibration_data.update(p_d0)
        print(f"Calibration P(d₀): {calibration_data}")
    except Exception as e:
        print(f"Error loading calibration CSV: {e}")

# Calculate distance from RSSI
def calculate_distance(p_d: float, antenna_id: int) -> float:
    p_d0 = calibration_data.get(antenna_id)
    if p_d0 is None:
        raise ValueError(f"No calibration data for antenna {antenna_id}")
    # d = d₀ * 10^((P(d₀) - P(d)) / (10n))
    return d0 * 10 ** ((p_d0 - p_d) / (10 * n))

# Trilateration function (least-squares optimization)
def trilaterate(positions: List[np.ndarray], distances: List[float]) -> np.ndarray:
    def residuals(xy, positions, distances):
        return np.array([np.sqrt((xy[0] - p[0])**2 + (xy[1] - p[1])**2) - d for p, d in zip(positions, distances)])
    
    # Initial guess at room center
    initial_guess = np.array([7.5, 7.5])
    result = least_squares(residuals, initial_guess, args=(positions, distances), bounds=([0, 0], [15, 15]))
    return result.x

# Store incoming RFID data
rssi_buffer: Dict[float, Dict[int, List[float]]] = {}

# FastAPI endpoint to receive Sllurp data
@app.post("/rfid_data")
async def receive_rfid_data(data: List[RFIDData]):
    global rssi_buffer
    current_time = time.time()
    window_start = math.floor(current_time)
    
    # Buffer RSSI data by antenna and time window
    for item in data:
        if item.antenna_id not in antennas:
            continue
        if window_start not in rssi_buffer:
            rssi_buffer[window_start] = {aid: [] for aid in antennas}
        rssi_buffer[window_start][item.antenna_id].append(item.rssi)
    
    # Process data for completed time windows
    process_rssi_buffer()
    
    return {"status": "received"}

# Process buffered RSSI data
def process_rssi_buffer():
    global rssi_buffer
    current_time = time.time()
    completed_windows = [t for t in rssi_buffer if t < current_time - 1]
    
    for window in completed_windows:
        antenna_rssi = rssi_buffer[window]
        distances = []
        valid_antennas = []
        
        # Calculate mean RSSI and distance per antenna
        for antenna_id, rssi_list in antenna_rssi.items():
            if rssi_list:
                mean_rssi = np.mean(rssi_list)
                try:
                    distance = calculate_distance(mean_rssi, antenna_id)
                    distances.append(distance)
                    valid_antennas.append(antennas[antenna_id])
                except ValueError:
                    continue
        
        # Perform trilateration if at least 3 antennas have data
        if len(valid_antennas) >= 3:
            try:
                xy = trilaterate(valid_antennas, distances)
                timestamp = datetime.fromtimestamp(window).strftime("%Y-%m-%d %H:%M:%S")
                pd.DataFrame([{"timestamp": timestamp, "x": xy[0], "y": xy[1]}]).to_csv(
                    output_csv, mode="a", header=False, index=False
                )
                print(f"Position at {timestamp}: x={xy[0]:.2f}, y={xy[1]:.2f}")
            except Exception as e:
                print(f"Trilateration failed for window {window}: {e}")
        
        # Clean up processed window
        del rssi_buffer[window]

# Main function to start the server
if __name__ == "__main__":
    # Load calibration CSV (update path as needed)
    load_calibration_csv("center_test.csv")
    init_output_csv()
    uvicorn.run(app, host="0.0.0.0", port=8000)