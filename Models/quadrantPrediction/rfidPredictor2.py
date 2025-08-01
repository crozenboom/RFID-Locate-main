from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import logging
import queue
from typing import Dict
from datetime import datetime
import asyncio
import uvicorn
import sys
import socket
import os

# Configure loggers
data_logger = logging.getLogger("rfid_data")
data_logger.setLevel(logging.INFO)
data_handler = logging.FileHandler("rfid_data.log")
data_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
data_logger.addHandler(data_handler)

prediction_logger = logging.getLogger("rfid_prediction")
prediction_logger.setLevel(logging.INFO)
prediction_handler = logging.FileHandler("rfid_predictions.log")
prediction_handler.setFormatter(logging.Formatter('%(message)s'))
prediction_logger.addHandler(prediction_handler)

# FastAPI app
app = FastAPI(title="RFID Quadrant Prediction API", version="1.0")

# Load model and scaler
try:
    model = joblib.load('model_full_features.pkl')
    scaler = joblib.load('scaler_full_features.pkl')
    data_logger.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    data_logger.error(f"Model or scaler file not found: {str(e)}")
    sys.exit(1)

# Queue for tag data
tag_queue = queue.Queue()

# Quadrant mapping
QUADRANT_MAP = {'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'Q4': 'Q4'}

# Initialize CSV log file with headers if it doesn't exist
log_file = 'live_predictions.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        'log_time', 'timestamp', 'reader', 'antenna', 'epc',
        'predicted_quadrant', 'confidence'
    ]).to_csv(log_file, index=False)

# Process tag data for prediction
def process_tag_data(row):
    data = {}
    # Fill RSSI values
    for i in range(1, 5):
        data[f'rssi_{i}'] = row.get(f'rssi_{i}', -80.0)
    # Phase angles (used: 3 and 4)
    data['phase_angle_3'] = row.get('phase_angle_3', 0.0)
    data['phase_angle_4'] = row.get('phase_angle_4', 0.0)
    # Doppler frequencies (used: 1 to 4)
    for i in range(1, 5):
        data[f'doppler_frequency_{i}'] = row.get(f'doppler_frequency_{i}', 0.0)

    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'phase_angle_3', 'phase_angle_4',
        'doppler_frequency_1', 'doppler_frequency_2',
        'doppler_frequency_3', 'doppler_frequency_4'
    ]

    df_row = pd.DataFrame([data])[features]
    try:
        X_scaled = scaler.transform(df_row)
        prediction = model.predict(X_scaled)[0]
        confidence = float(np.max(model.predict_proba(X_scaled)[0]))
        quadrant = QUADRANT_MAP.get(str(prediction), 'Unknown')
        prediction_logger.info(quadrant)
        return quadrant, confidence
    except Exception as e:
        data_logger.error(f"Error in prediction for EPC {row.get('EPC')}: {str(e)}")
        return 'Unknown', 0.0

# Async tag processor
async def process_tags():
    recent_data = pd.DataFrame(columns=['timestamp', 'EPC'], dtype=float)
    recent_data['EPC'] = recent_data['EPC'].astype(str)

    while True:
        try:
            tags = []
            try:
                while True:
                    tags.append(tag_queue.get_nowait())
            except queue.Empty:
                pass

            if not tags:
                await asyncio.sleep(0.1)
                continue

            df = pd.DataFrame(tags)
            if df.empty or 'AntennaID' not in df.columns or 'ImpinjPeakRSSI' not in df.columns:
                await asyncio.sleep(0.1)
                continue

            df['timestamp'] = (df['LastSeen'] / 1000000).round(3)
            df['time_window'] = (df['LastSeen'] / 1000000 // 0.5) * 0.5

            pivoted = df.pivot_table(
                index='time_window',
                columns='AntennaID',
                values='ImpinjPeakRSSI',
                aggfunc='mean'
            ).reset_index()

            pivoted.columns = [f'rssi_{col}' if col in [1, 2, 3, 4] else col for col in pivoted.columns]
            for i in range(1, 5):
                col = f'rssi_{i}'
                if col not in pivoted.columns:
                    pivoted[col] = -80.0
                pivoted[col] = pivoted[col].fillna(-80.0)

            epc_map = df.groupby('time_window')['EPC'].last().to_dict()
            pivoted['EPC'] = pivoted['time_window'].map(epc_map)

            # Add dummy phase and doppler values for now
            pivoted['phase_angle_3'] = 0.0
            pivoted['phase_angle_4'] = 0.0
            for i in range(1, 5):
                pivoted[f'doppler_frequency_{i}'] = 0.0

            log_data = []
            for _, row in pivoted.iterrows():
                if row[['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].eq(-80.0).sum() > 3:
                    continue
                quadrant, confidence = process_tag_data(row)
                log_data.append({
                    'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': row['time_window'] * 1000,
                    'reader': 'reader1',
                    'antenna': 0,
                    'epc': row['EPC'],
                    'predicted_quadrant': quadrant,
                    'confidence': confidence
                })

            if log_data:
                pd.DataFrame(log_data).to_csv(log_file, mode='a', header=False, index=False)
        except Exception as e:
            data_logger.error(f"Error processing tags: {str(e)}")
        await asyncio.sleep(0.1)

@app.post("/rfid-data")
async def receive_rfid_data(payload: Dict):
    try:
        tags = payload.get('tag_reads', [])
        if not tags:
            return {"status": "success", "message": "No tags received"}

        for tag in tags:
            antenna_id = tag.get('antennaPort', tag.get('AntennaPort', tag.get('antenna_id', tag.get('AntennaID', 0))))
            rssi = float(tag.get('rssi', tag.get('peakRssi', tag.get('PeakRSSI', -80.0))))
            if not -100 <= rssi <= -30:
                continue
            tag_data = {
                'AntennaID': int(antenna_id),
                'EPC': tag.get('epc', tag.get('EPC', '')),
                'LastSeen': tag.get('firstSeenTimestamp', tag.get('timestamp', int(datetime.now().timestamp() * 1000))),
                'ImpinjPeakRSSI': rssi
            }
            tag_queue.put(tag_data)
        return {"status": "success", "message": f"Received {len(tags)} tags"}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error processing RFID data: {str(e)}")

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0

async def main():
    asyncio.create_task(process_tags())
    config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == '__main__':
    if is_port_in_use(8000):
        print("Port 8000 is in use.")
        sys.exit(1)
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())