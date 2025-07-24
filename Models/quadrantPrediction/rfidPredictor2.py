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

# Configure separate loggers
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
    model = joblib.load('model_rssi.pkl')
    scaler = joblib.load('scaler_rssi.pkl')
    data_logger.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    data_logger.error(f"Model or scaler file not found: {str(e)}")
    sys.exit(1)

# Queue for tag data
tag_queue = queue.Queue()

# Quadrant mapping
QUADRANT_MAP = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}

# Initialize CSV log file
log_file = 'live_predictions.csv'
if not os.path.exists(log_file):
    pd.DataFrame(columns=[
        'log_time', 'timestamp', 'reader', 'antenna', 'rssi', 'epc',
        'phase_angle', 'channel_index', 'doppler_frequency', 'predicted_quadrant', 'confidence'
    ]).to_csv(log_file, index=False)

# Process tag data for prediction
def process_tag_data(row):
    data = {}
    for col in ['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']:
        data[col] = row.get(col, -80.0)
    data['rssi_1_2_diff'] = data['rssi_1'] - data['rssi_2']
    data['rssi_3_4_diff'] = data['rssi_3'] - data['rssi_4']
    
    data_logger.info(f"Processed RSSI for EPC {row.get('EPC')}: {data}")
    
    df_row = pd.DataFrame([data])
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'rssi_1_2_diff', 'rssi_3_4_diff'
    ]
    X_scaled = scaler.transform(df_row[features])
    data_logger.info(f"Scaled features for EPC {row.get('EPC')}: {X_scaled.tolist()}")
    
    prediction = model.predict(X_scaled)[0]
    confidence = float(np.max(model.predict_proba(X_scaled)[0]))
    quadrant = QUADRANT_MAP.get(str(prediction), 0)
    
    # Log just the quadrant number
    prediction_logger.info(str(quadrant))
    
    return quadrant, confidence

# Async function to process tags
async def process_tags():
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
                
                df['LastSeenGroup'] = (df['LastSeen'] // 100000).astype(int)
                pivoted = df.pivot_table(
                    index=['LastSeenGroup', 'EPC'],
                    columns='AntennaID',
                    values=['ImpinjPeakRSSI'],
                    aggfunc='mean'
                ).reset_index()

                pivoted_columns = ['LastSeenGroup', 'EPC']
                antenna_cols = {}
                for col in pivoted.columns[2:]:
                    try:
                        antenna_id = int(col[1])
                        pivoted_columns.append(f'impinjpeakrssi_{antenna_id}')
                        antenna_cols[col] = f'impinjpeakrssi_{antenna_id}'
                    except (ValueError, TypeError):
                        data_logger.warning(f"Unexpected column in pivot table: {col}")
                pivoted.columns = pivoted_columns
                data_logger.info(f"Pivoted columns: {list(pivoted.columns)}")
                data_logger.info(f"Sample pivoted row: {pivoted.iloc[0].to_dict() if not pivoted.empty else 'Empty'}")
                
                for i in range(1, 5):
                    col = f'impinjpeakrssi_{i}'
                    if col not in pivoted.columns:
                        available_cols = [c for c in pivoted.columns if c.startswith('impinjpeakrssi_')]
                        pivoted[col] = pivoted[available_cols].mean(axis=1, skipna=True)
                    pivoted[col] = pivoted[col].fillna(-80.0)
                
                for i in range(1, 5):
                    pivoted[f'rssi_{i}'] = pivoted[f'impinjpeakrssi_{i}']
                
                data_logger.info(f"Processed pivoted row (first): {pivoted[['EPC', 'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].iloc[0].to_dict() if not pivoted.empty else 'Empty'}")
                
                # Log predictions to CSV
                log_data = []
                for _, row in pivoted.iterrows():
                    quadrant, confidence = process_tag_data(row)
                    print(f"EPC: {row['EPC']}, Quadrant: {quadrant}")
                    data_logger.info(f"EPC: {row['EPC']}, Quadrant: {quadrant}, RSSI: {[row[f'rssi_{i}'] for i in range(1, 5)]}")
                    log_data.append({
                        'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'timestamp': row['LastSeenGroup'] * 100000,
                        'reader': 'reader1',  # Placeholder, update if available
                        'antenna': 0,  # Aggregate, so no single antenna
                        'rssi': row[['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']].mean(),
                        'epc': row['EPC'],
                        'phase_angle': 0.0,  # Not used
                        'channel_index': 0.0,  # Not used
                        'doppler_frequency': 0.0,  # Not used
                        'predicted_quadrant': f'Q{quadrant}',
                        'confidence': confidence
                    })
                pd.DataFrame(log_data).to_csv(log_file, mode='a', header=False, index=False)
                data_logger.info(f"Logged predictions to {log_file}")
        except Exception as e:
            data_logger.error(f"Error processing tags: {str(e)}", exc_info=True)
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
            antenna_ids.add(antenna_id)
            tag_data = {
                'AntennaID': int(antenna_id),
                'EPC': tag.get('epc', tag.get('EPC', '')),
                'LastSeen': tag.get('firstSeenTimestamp', tag.get('timestamp', tag.get('Timestamp', int(datetime.now().timestamp() * 1000)))),
                'ImpinjPeakRSSI': float(tag.get('peakRssi', tag.get('PeakRSSI', -80.0)))
            }
            processed_tags.append(tag_data)
            tag_queue.put(tag_data)
            data_logger.info(f"Processed tag: {tag_data}")
        data_logger.info(f"Received AntennaIDs: {sorted(antenna_ids)}")
        
        return {"status": "success", "message": f"Received {len(tags)} tags"}
    except Exception as e:
        data_logger.error(f"Error processing RFID data: {str(e)}", exc_info=True)
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
        data_logger.error(f"FastAPI server error: {str(e)}", exc_info=True)
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