from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
import numpy as np
import logging
import queue
from typing import Dict, List
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

# Quadrant mapping (updated to handle string labels)
QUADRANT_MAP = {'Q1': 'Q1', 'Q2': 'Q2', 'Q3': 'Q3', 'Q4': 'Q4'}

# Initialize CSV log file
log_file = 'live_predictions.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        'log_time', 'timestamp', 'reader', 'antenna', 'epc',
        'predicted_quadrant', 'confidence'
    ]).to_csv(log_file, index=False)

# Process tag data for prediction
def process_tag_data(row):
    data = {}
    for i in range(1, 5):
        col = f'rssi_{i}'
        data[col] = row.get(col, -80.0)
    data['rssi_1_2_diff'] = data['rssi_1'] - data['rssi_2']
    data['rssi_3_4_diff'] = data['rssi_3'] - data['rssi_4']
    for i in range(1, 5):
        data[f'rssi_{i}_mean'] = row.get(f'rssi_{i}_mean', data[f'rssi_{i}'])
    
    data_logger.info(f"Processed features for EPC {row.get('EPC')}: {data}")
    
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'rssi_1_2_diff', 'rssi_3_4_diff',
        'rssi_1_mean', 'rssi_2_mean', 'rssi_3_mean', 'rssi_4_mean'
    ]
    df_row = pd.DataFrame([data])[features]
    try:
        X_scaled = scaler.transform(df_row)
        data_logger.info(f"Scaled features for EPC {row.get('EPC')}: {X_scaled.tolist()}")
        
        prediction = model.predict(X_scaled)[0]  # Keep as string (e.g., 'Q3')
        confidence = float(np.max(model.predict_proba(X_scaled)[0]))
        quadrant = QUADRANT_MAP.get(str(prediction), 'Unknown')  # Handle string prediction
        
        prediction_logger.info(quadrant)
        
        return quadrant, confidence
    except Exception as e:
        data_logger.error(f"Error in prediction for EPC {row.get('EPC')}: {str(e)}")
        return 'Unknown', 0.0

# Async function to process tags
async def process_tags():
    recent_data = pd.DataFrame(
        columns=['timestamp', 'EPC', 'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4'],
        dtype=float
    )
    recent_data['EPC'] = recent_data['EPC'].astype(str)
    
    while True:
        try:
            tags = []
            try:
                while True:
                    tags.append(tag_queue.get_nowait())
            except queue.Empty:
                pass

            if tags:
                df = pd.DataFrame(tags)
                data_logger.info(f"Raw tag data (first 5 rows): {df[['AntennaID', 'EPC', 'LastSeen', 'ImpinjPeakRSSI']].head().to_dict()}")
                data_logger.info(f"Unique AntennaIDs received: {sorted(df['AntennaID'].unique())}")
                
                if df.empty or 'AntennaID' not in df.columns or 'ImpinjPeakRSSI' not in df.columns:
                    data_logger.warning("Invalid or empty tag data received")
                    await asyncio.sleep(0.1)
                    continue
                
                # Round timestamp to milliseconds and convert to seconds
                df['timestamp'] = (df['LastSeen'] / 1000000).round(3)
                # Aggregate data within a time window (e.g., 0.5 seconds)
                df['time_window'] = (df['LastSeen'] / 1000000 // 0.5) * 0.5
                
                # Pivot on time_window instead of timestamp and EPC
                pivoted = df.pivot_table(
                    index='time_window',
                    columns='AntennaID',
                    values='ImpinjPeakRSSI',
                    aggfunc='mean'
                ).reset_index()
                
                if pivoted.empty:
                    data_logger.warning("Pivot table is empty, likely due to insufficient antenna data")
                    await asyncio.sleep(0.1)
                    continue
                
                pivoted.columns = [f'rssi_{col}' if col in [1, 2, 3, 4] else col for col in pivoted.columns]
                for i in range(1, 5):
                    col = f'rssi_{i}'
                    if col not in pivoted.columns:
                        pivoted[col] = -80.0
                    pivoted[col] = pivoted[col].fillna(-80.0)
                
                # Add EPC for logging (take the most recent EPC in the window)
                epc_map = df.groupby('time_window')['EPC'].last().to_dict()
                pivoted['EPC'] = pivoted['time_window'].map(epc_map)
                
                new_data = pivoted[['time_window', 'EPC', 'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].copy()
                new_data['EPC'] = new_data['EPC'].astype(str)
                recent_data = pd.concat([recent_data, new_data], ignore_index=True)
                recent_data = recent_data.tail(100)
                
                for i in range(1, 5):
                    pivoted[f'rssi_{i}_mean'] = pivoted[f'rssi_{i}'].rolling(window=5, min_periods=1).mean().fillna(pivoted[f'rssi_{i}'])
                
                data_logger.info(f"Pivot table columns: {pivoted.columns.tolist()}")
                data_logger.info(f"Pivot table size: {len(pivoted)} rows")
                data_logger.info(f"Processed pivoted row (first): {pivoted[['EPC', 'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4', 'rssi_1_mean', 'rssi_2_mean', 'rssi_3_mean', 'rssi_4_mean']].iloc[0].to_dict() if not pivoted.empty else 'Empty'}")
                
                log_data = []
                for _, row in pivoted.iterrows():
                    # Relax filter to allow at least one valid RSSI value
                    if row[['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].eq(-80.0).sum() > 3:
                        data_logger.warning(f"Skipping EPC {row['EPC']}: Too many default RSSI values")
                        continue
                    quadrant, confidence = process_tag_data(row)
                    if quadrant == 'Unknown':
                        data_logger.warning(f"Prediction failed for EPC {row['EPC']}: Invalid prediction")
                        # Log failed predictions with default values
                        log_data.append({
                            'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'timestamp': row['time_window'] * 1000,
                            'reader': 'reader1',
                            'antenna': 0,
                            'epc': row['EPC'],
                            'predicted_quadrant': 'Unknown',
                            'confidence': 0.0
                        })
                        continue
                    print(f"EPC: {row['EPC']}, Quadrant: {quadrant}, Confidence: {confidence:.4f}")
                    data_logger.info(f"EPC: {row['EPC']}, Quadrant: {quadrant}, Confidence: {confidence:.4f}, RSSI: {[row[f'rssi_{i}'] for i in range(1, 5)]}")
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
                    data_logger.info(f"Logged {len(log_data)} predictions to {log_file}")
                else:
                    data_logger.warning("No valid predictions to log")
        except Exception as e:
            data_logger.error(f"Error processing tags: {str(e)}")
        await asyncio.sleep(0.1)

# FastAPI endpoint to receive RFID data
@app.post("/rfid-data")
async def receive_rfid_data(payload: Dict):
    try:
        data_logger.info(f"Raw payload: {payload}")
        tags = payload.get('tag_reads', [])
        if not tags:
            data_logger.warning("No tag_reads in payload")
            return {"status": "success", "message": "No tags received"}

        processed_tags = []
        antenna_ids = set()
        for tag in tags:
            antenna_id = tag.get('antennaPort', tag.get('AntennaPort', tag.get('antenna_id', tag.get('AntennaID', 0))))
            rssi = float(tag.get('rssi', tag.get('peakRssi', tag.get('PeakRSSI', -80.0))))
            if not -100 <= rssi <= -30:
                data_logger.warning(f"Invalid RSSI {rssi} for EPC {tag.get('epc', tag.get('EPC', ''))}")
                continue
            antenna_ids.add(antenna_id)
            tag_data = {
                'AntennaID': int(antenna_id),
                'EPC': tag.get('epc', tag.get('EPC', '')),
                'LastSeen': tag.get('firstSeenTimestamp', tag.get('timestamp', tag.get('Timestamp', int(datetime.now().timestamp() * 1000)))),
                'ImpinjPeakRSSI': rssi
            }
            processed_tags.append(tag_data)
            tag_queue.put(tag_data)
            data_logger.info(f"Processed tag: {tag_data}")
        data_logger.info(f"Received AntennaIDs: {sorted(antenna_ids)}")
        
        return {"status": "success", "message": f"Received {len(processed_tags)} tags"}
    except Exception as e:
        data_logger.error(f"Error processing RFID data: {str(e)}")
        raise HTTPException(status_code=422, detail=f"Error processing RFID data: {str(e)}")

# Check if port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0

# Main function to start FastAPI
async def main():
    asyncio.create_task(process_tags())
    try:
        print("Starting FastAPI server on port 8000...")
        print("Waiting for RFID reader POSTs at /rfid-data...")
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        data_logger.error(f"FastAPI server error: {str(e)}")
        print(f"Error starting FastAPI server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Please free it or use a different port.")
        print("To find and kill the process, run: `lsof -i :8000` and `kill -9 <PID>`")
        sys.exit(1)

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(main())