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
data_logger.setLevel(logging.DEBUG)
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

# Initialize CSV log file with headers
log_file = 'live_predictions.csv'
if not os.path.exists(log_file):
    try:
        pd.DataFrame(columns=[
            'log_time', 'timestamp', 'reader', 'antenna', 'epc', 'predicted_quadrant', 'confidence',
            'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
            'std_rssi_1', 'std_rssi_2', 'std_rssi_3', 'std_rssi_4',
            'doppler_rssi_1', 'doppler_rssi_2', 'doppler_rssi_3', 'doppler_rssi_4',
            'rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3', 'rssi_diff_1_3', 'rssi_diff_4_2'
        ]).to_csv(log_file, index=False)
        with open(log_file, 'w') as f:
            f.write('# Live Predictions\n')
            f.write('# This file contains predicted quadrants for RFID data.\n')
            f.write('# - log_time: Time of prediction logging.\n')
            f.write('# - timestamp: Time of the RFID reading (ms).\n')
            f.write('# - reader: RFID reader identifier.\n')
            f.write('# - antenna: Antenna ID (0 for aggregated).\n')
            f.write('# - epc: Tag EPC identifier.\n')
            f.write('# - predicted_quadrant: Predicted quadrant (Q1-Q4).\n')
            f.write('# - confidence: Prediction confidence score.\n')
            f.write('# - rssi_1 to rssi_4: Mean RSSI for antennas 1-4.\n')
            f.write('# - std_rssi_1 to std_rssi_4: Std RSSI for antennas 1-4.\n')
            f.write('# - doppler_rssi_1 to doppler_rssi_4: Mean Doppler RSSI for antennas 1-4.\n')
            f.write('# - rssi_diff_X_Y: RSSI difference between antennas X and Y.\n')
        pd.DataFrame(columns=[
            'log_time', 'timestamp', 'reader', 'antenna', 'epc', 'predicted_quadrant', 'confidence',
            'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
            'std_rssi_1', 'std_rssi_2', 'std_rssi_3', 'std_rssi_4',
            'doppler_rssi_1', 'doppler_rssi_2', 'doppler_rssi_3', 'doppler_rssi_4',
            'rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3', 'rssi_diff_1_3', 'rssi_diff_4_2'
        ]).to_csv(log_file, mode='a', index=False)
        data_logger.info(f"Initialized {log_file} with headers.")
    except Exception as e:
        data_logger.error(f"Failed to initialize {log_file}: {str(e)}")
        sys.exit(1)

def process_tag_data(row):
    data = {}
    for i in range(1, 5):
        data[f'rssi_{i}'] = row.get(f'rssi_{i}', -80.0)
        data[f'std_rssi_{i}'] = row.get(f'std_rssi_{i}', 0.0)
        data[f'doppler_rssi_{i}'] = row.get(f'doppler_rssi_{i}', 0.0)
    data['rssi_diff_1_2'] = data['rssi_1'] - data['rssi_2']
    data['rssi_diff_3_4'] = data['rssi_3'] - data['rssi_4']
    data['rssi_diff_1_4'] = data['rssi_1'] - data['rssi_4']
    data['rssi_diff_2_3'] = data['rssi_2'] - data['rssi_3']
    data['rssi_diff_1_3'] = data['rssi_1'] - data['rssi_3']
    data['rssi_diff_4_2'] = data['rssi_4'] - data['rssi_2']
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'std_rssi_1', 'std_rssi_2', 'std_rssi_3', 'std_rssi_4',
        'doppler_rssi_1', 'doppler_rssi_2', 'doppler_rssi_3', 'doppler_rssi_4',
        'rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3',
        'rssi_diff_1_3', 'rssi_diff_4_2'
    ]
    df_row = pd.DataFrame([data])[features]
    try:
        X_scaled = scaler.transform(df_row)
        y_pred = model.predict(X_scaled)[0]
        confidence = float(np.max(model.predict_proba(X_scaled)[0]))
        quadrant = QUADRANT_MAP.get(str(y_pred), 'Unknown')
        prediction_logger.info(f"Predicted quadrant: {quadrant}, confidence: {confidence}, EPC: {row.get('EPC', 'unknown')}")
        return quadrant, confidence, data
    except Exception as e:
        data_logger.error(f"Error in prediction for EPC {row.get('EPC', 'unknown')}: {str(e)}")
        return 'Unknown', 0.0, data

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
                data_logger.debug("No tags in queue, waiting...")
                await asyncio.sleep(0.1)
                continue
            data_logger.info(f"Received {len(tags)} tags for processing.")
            df = pd.DataFrame(tags)
            if df.empty or 'AntennaID' not in df.columns or 'ImpinjPeakRSSI' not in df.columns:
                data_logger.warning("DataFrame empty or missing required columns.")
                await asyncio.sleep(0.1)
                continue
            if not df['AntennaID'].isin([1, 2, 3, 4]).all():
                data_logger.warning(f"Invalid AntennaID values: {df['AntennaID'].unique()}")
                await asyncio.sleep(0.1)
                continue
            invalid_rssi = (df['ImpinjPeakRSSI'] < -100) | (df['ImpinjPeakRSSI'] > -30)
            if invalid_rssi.any():
                data_logger.debug(f"Invalid RSSI values: {df[invalid_rssi][['EPC', 'ImpinjPeakRSSI']].to_dict(orient='records')}")
                df = df[~invalid_rssi]
            if df.empty:
                data_logger.warning("No valid tags after RSSI filtering.")
                await asyncio.sleep(0.1)
                continue
            df['timestamp'] = (df['LastSeen'] / 1000000).round(3)
            df['time_window'] = (df['LastSeen'] / 1000000 // 1.0) * 1.0
            values = ['ImpinjPeakRSSI']
            agg_funcs = ['mean', 'std']
            if 'Doppler' in df.columns:
                values.append('Doppler')
                agg_funcs.append('mean')
                data_logger.debug("Doppler data found in input tags")
            else:
                data_logger.warning("No Doppler data in input tags; setting doppler_rssi_1-4 to 0.0")
            pivoted = df.pivot_table(
                index='time_window',
                columns='AntennaID',
                values=values,
                aggfunc=agg_funcs
            ).reset_index()
            flat_columns = ['time_window']
            for col in pivoted.columns[1:]:
                stat, val, ant = col
                if val == 'ImpinjPeakRSSI' and stat == 'mean':
                    flat_columns.append(f'rssi_{ant}')
                elif val == 'ImpinjPeakRSSI':
                    flat_columns.append(f'{stat}_rssi_{ant}')
                elif val == 'Doppler':
                    flat_columns.append(f'doppler_rssi_{ant}')
            pivoted.columns = flat_columns
            for i in range(1, 5):
                mean_col = f'rssi_{i}'
                std_col = f'std_rssi_{i}'
                doppler_col = f'doppler_rssi_{i}'
                if mean_col not in pivoted.columns:
                    pivoted[mean_col] = -80.0
                    pivoted[std_col] = 0.0
                    pivoted[doppler_col] = 0.0
                pivoted[mean_col] = pivoted[mean_col].fillna(-80.0)
                pivoted[std_col] = pivoted[std_col].fillna(0.0)
                pivoted[doppler_col] = pivoted[doppler_col].fillna(0.0)
            epc_map = df.groupby('time_window')['EPC'].last().to_dict()
            pivoted['EPC'] = pivoted['time_window'].map(epc_map)
            if pivoted['EPC'].isna().all():
                data_logger.warning("No valid EPCs mapped to time windows.")
                await asyncio.sleep(0.1)
                continue
            data_logger.info(f"Processing {len(pivoted)} pivoted rows.")
            log_data = []
            for _, row in pivoted.iterrows():
                if pd.isna(row['EPC']):
                    data_logger.debug(f"Skipping row with missing EPC: {row.to_dict()}")
                    continue
                rssi_values = {}
                for i in range(1, 5):
                    rssi_values[f'rssi_{i}'] = row.get(f'rssi_{i}', -80.0)
                    rssi_values[f'std_rssi_{i}'] = row.get(f'std_rssi_{i}', 0.0)
                    rssi_values[f'doppler_rssi_{i}'] = row.get(f'doppler_rssi_{i}', 0.0)
                rssi_values['EPC'] = row['EPC']
                quadrant, confidence, features = process_tag_data(rssi_values)
                data_logger.debug(f"Antenna RSSI stats - A1: mean={row.get('rssi_1', -80.0):.2f}, std={row.get('std_rssi_1', 0.0):.2f}, doppler={row.get('doppler_rssi_1', 0.0):.2f}; "
                                f"A2: mean={row.get('rssi_2', -80.0):.2f}, std={row.get('std_rssi_2', 0.0):.2f}, doppler={row.get('doppler_rssi_2', 0.0):.2f}; "
                                f"A3: mean={row.get('rssi_3', -80.0):.2f}, std={row.get('std_rssi_3', 0.0):.2f}, doppler={row.get('doppler_rssi_3', 0.0):.2f}; "
                                f"A4: mean={row.get('rssi_4', -80.0):.2f}, std={row.get('std_rssi_4', 0.0):.2f}, doppler={row.get('doppler_rssi_4', 0.0):.2f}")
                log_data.append({
                    'log_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'timestamp': row['time_window'] * 1000,
                    'reader': 'reader1',
                    'antenna': 0,
                    'epc': str(row['EPC']),
                    'predicted_quadrant': quadrant,
                    'confidence': confidence,
                    'rssi_1': features['rssi_1'],
                    'rssi_2': features['rssi_2'],
                    'rssi_3': features['rssi_3'],
                    'rssi_4': features['rssi_4'],
                    'std_rssi_1': features['std_rssi_1'],
                    'std_rssi_2': features['std_rssi_2'],
                    'std_rssi_3': features['std_rssi_3'],
                    'std_rssi_4': features['std_rssi_4'],
                    'doppler_rssi_1': features['doppler_rssi_1'],
                    'doppler_rssi_2': features['doppler_rssi_2'],
                    'doppler_rssi_3': features['doppler_rssi_3'],
                    'doppler_rssi_4': features['doppler_rssi_4'],
                    'rssi_diff_1_2': features['rssi_diff_1_2'],
                    'rssi_diff_3_4': features['rssi_diff_3_4'],
                    'rssi_diff_1_4': features['rssi_diff_1_4'],
                    'rssi_diff_2_3': features['rssi_diff_2_3'],
                    'rssi_diff_1_3': features['rssi_diff_1_3'],
                    'rssi_diff_4_2': features['rssi_diff_4_2']
                })
            if log_data:
                try:
                    pd.DataFrame(log_data).to_csv(log_file, mode='a', header=False, index=False)
                    data_logger.info(f"Appended {len(log_data)} predictions to {log_file}")
                except Exception as e:
                    data_logger.error(f"Error writing to {log_file}: {str(e)}")
            else:
                data_logger.warning("No valid predictions to log.")
        except Exception as e:
            data_logger.error(f"Error processing tags: {str(e)}")
        await asyncio.sleep(0.1)

@app.post("/rfid-data")
async def receive_rfid_data(payload: Dict):
    try:
        data_logger.debug(f"Received payload: {payload}")
        tags = payload.get('tag_reads', [])
        if not tags:
            data_logger.info("No tags received in payload.")
            return {"status": "success", "message": "No tags received"}
        valid_tags = []
        for tag in tags:
            antenna_id = tag.get('antennaPort', tag.get('AntennaPort', tag.get('antenna_id', tag.get('AntennaID', 0))))
            rssi = float(tag.get('rssi', tag.get('peakRssi', tag.get('PeakRSSI', -80.0))))
            epc = tag.get('epc', tag.get('EPC', ''))
            doppler = float(tag.get('doppler', tag.get('Doppler', 0.0)))
            if isinstance(epc, str) and epc.startswith("b'") and epc.endswith("'"):
                epc = epc[2:-1]
            elif isinstance(epc, bytes):
                epc = epc.decode('utf-8', errors='ignore')
            if not -100 <= rssi <= -30:
                data_logger.debug(f"Invalid RSSI {rssi} for tag {epc}, skipping.")
                continue
            if not isinstance(antenna_id, int) or antenna_id not in [1, 2, 3, 4]:
                data_logger.debug(f"Invalid AntennaID {antenna_id} for tag {epc}, skipping.")
                continue
            tag_data = {
                'AntennaID': int(antenna_id),
                'EPC': epc,
                'LastSeen': tag.get('firstSeenTimestamp', tag.get('timestamp', int(datetime.now().timestamp() * 1000))),
                'ImpinjPeakRSSI': rssi,
                'Doppler': doppler
            }
            valid_tags.append(tag_data)
            tag_queue.put(tag_data)
        data_logger.info(f"Queued {len(valid_tags)} valid tags out of {len(tags)} received.")
        return {"status": "success", "message": f"Received {len(tags)} tags, queued {len(valid_tags)}"}
    except Exception as e:
        data_logger.error(f"Error processing RFID data: {str(e)}")
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