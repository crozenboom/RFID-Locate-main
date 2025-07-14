from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import APIKeyHeader
import joblib
import pandas as pd
import numpy as np
import logging
import threading
import queue
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
import aiohttp
import asyncio
import requests
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
import plotly.graph_objs as go
import uvicorn
import sys
import socket
from dateutil.parser import parse as parse_date

# Configure logging
logging.basicConfig(filename='rfid_predictions.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# FastAPI app
fastapi_app = FastAPI(title="RFID Quadrant Prediction API", version="1.0")

# Load model and scaler
try:
    model = joblib.load('rfid_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    logging.error(f"Model or scaler file not found: {str(e)}")
    sys.exit(1)

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")
API_KEY = "cachengo"  # Replace with your actual key

# Configuration for RFID reader and output endpoint
RFID_READER_ENDPOINT = "http://<your-rfid-reader-ip>:8080/rfid-data"  # Replace with actual RFID reader endpoint
OUTPUT_ENDPOINT = "http://<your-output-endpoint>"  # Replace with actual endpoint to send predictions
POLL_INTERVAL = 0.5  # Seconds between polls to RFID reader

# Pydantic models
class RFIDSensorData(BaseModel):
    EPC: str
    AntennaID: int
    PeakRSSI: float
    PhaseAngle: float
    DopplerFrequency: float
    Timestamp: str

class PredictionResponse(BaseModel):
    epc: str
    rssi_1: float
    rssi_2: float
    rssi_3: float
    rssi_4: float
    phase_angle_1: float
    phase_angle_2: float
    phase_angle_3: float
    phase_angle_4: float
    doppler_frequency_1: float
    doppler_frequency_2: float
    doppler_frequency_3: float
    doppler_frequency_4: float
    quadrant: str
    timestamp: float
    links: dict

# Queue for tag data
tag_queue = queue.Queue()

# Process tag data for prediction
def process_tag_data(row):
    data = {}
    for col in ['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4']:
        data[col] = row.get(col, -80.0)
    for col in ['phase_angle_1', 'phase_angle_2', 'phase_angle_3', 'phase_angle_4']:
        data[col] = row.get(col, 0.0)
    for col in ['doppler_frequency_1', 'doppler_frequency_2', 'doppler_frequency_3', 'doppler_frequency_4']:
        data[col] = row.get(col, 0.0)

    data['rssi_1_2_diff'] = data['rssi_1'] - data['rssi_2']
    data['rssi_3_4_diff'] = data['rssi_3'] - data['rssi_4']
    data['phase_angle_1_2_diff'] = data['phase_angle_1'] - data['phase_angle_2']
    data['phase_angle_3_4_diff'] = data['phase_angle_3'] - data['phase_angle_4']

    df_row = pd.DataFrame([data])
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'phase_angle_1', 'phase_angle_2', 'phase_angle_3', 'phase_angle_4',
        'doppler_frequency_1', 'doppler_frequency_2', 'doppler_frequency_3', 'doppler_frequency_4',
        'rssi_1_2_diff', 'rssi_3_4_diff', 'phase_angle_1_2_diff', 'phase_angle_3_4_diff'
    ]
    X_scaled = scaler.transform(df_row[features])
    prediction = model.predict(X_scaled)[0]
    return data, prediction

# Async function to poll RFID reader
async def poll_rfid_reader():
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(RFID_READER_ENDPOINT, headers={'X-API-Key': API_KEY}) as response:
                    if response.status == 200:
                        tags = await response.json()
                        for tag in tags:
                            tag_data = {
                                'AntennaID': tag['AntennaID'],
                                'EPC': tag['EPC'],
                                'LastSeen': parse_date(tag['Timestamp']).timestamp() * 1000,
                                'ImpinjPeakRSSI': tag['PeakRSSI'],
                                'Phase Angle': tag['PhaseAngle'],
                                'Doppler Frequency': tag['DopplerFrequency']
                            }
                            tag_queue.put(tag_data)
                            logging.info(f"Received tag: {tag_data}")
                    else:
                        logging.warning(f"RFID reader request failed: {response.status}")
            except Exception as e:
                logging.error(f"Error polling RFID reader: {str(e)}")
            await asyncio.sleep(POLL_INTERVAL)

# Async function to send predictions
async def send_predictions():
    async with aiohttp.ClientSession() as session:
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
                    pivoted = df.pivot_table(
                        index=['LastSeen', 'EPC'],
                        columns='AntennaID',
                        values=['ImpinjPeakRSSI', 'Phase Angle', 'Doppler Frequency'],
                        aggfunc='first'
                    ).reset_index()
                    pivoted.columns = ['LastSeen', 'EPC'] + [
                        f'{col[0].lower().replace(" ", "_")}_{col[1]}' if col[1] else col[0]
                        for col in pivoted.columns[2:]
                    ]

                    for i in range(1, 5):
                        pivoted[f'impinjpeakrssi_{i}'] = pivoted.get(f'impinjpeakrssi_{i}', pd.Series([-80.0] * len(pivoted)))
                        pivoted[f'phase_angle_{i}'] = pivoted.get(f'phase_angle_{i}', pd.Series([0.0] * len(pivoted)))
                        pivoted[f'doppler_frequency_{i}'] = pivoted.get(f'doppler_frequency_{i}', pd.Series([0.0] * len(pivoted)))

                    for _, row in pivoted.iterrows():
                        data, prediction = process_tag_data(row)
                        output = {
                            'epc': row['EPC'],
                            'rssi_1': float(data['rssi_1']),
                            'rssi_2': float(data['rssi_2']),
                            'rssi_3': float(data['rssi_3']),
                            'rssi_4': float(data['rssi_4']),
                            'phase_angle_1': float(data['phase_angle_1']),
                            'phase_angle_2': float(data['phase_angle_2']),
                            'phase_angle_3': float(data['phase_angle_3']),
                            'phase_angle_4': float(data['phase_angle_4']),
                            'doppler_frequency_1': float(data['doppler_frequency_1']),
                            'doppler_frequency_2': float(data['doppler_frequency_2']),
                            'doppler_frequency_3': float(data['doppler_frequency_3']),
                            'doppler_frequency_4': float(data['doppler_frequency_4']),
                            'quadrant': str(prediction),
                            'timestamp': float(row['LastSeen']),
                            'links': {
                                'self': f"/predictions/{row['EPC']}",
                                'all_predictions': '/predictions',
                                'dashboard': '/dashboard'
                            }
                        }
                        logging.info(f"Prediction: {output}")
                        print(f"EPC: {output['epc']}, RSSI: {output['rssi_1']:.1f}, {output['rssi_2']:.1f}, "
                              f"{output['rssi_3']:.1f}, {output['rssi_4']:.1f}, "
                              f"Phase: {output['phase_angle_1']:.1f}, {output['phase_angle_2']:.1f}, "
                              f"{output['phase_angle_3']:.1f}, {output['phase_angle_4']:.1f}, "
                              f"Doppler: {output['doppler_frequency_1']:.1f}, {output['doppler_frequency_2']:.1f}, "
                              f"{output['doppler_frequency_3']:.1f}, {output['doppler_frequency_4']:.1f}, "
                              f"Quadrant: {output['quadrant']}")

                        async with session.post(OUTPUT_ENDPOINT, json=output) as resp:
                            if resp.status != 200:
                                logging.error(f"Failed to send prediction for EPC {output['epc']}: {resp.status}")
            except Exception as e:
                logging.error(f"Error processing predictions: {str(e)}")
            await asyncio.sleep(0.1)

# FastAPI endpoint to receive RFID data (backup)
@fastapi_app.post("/rfid-data")
async def receive_rfid_data(tags: List[RFIDSensorData]):
    try:
        for tag in tags:
            tag_data = {
                'AntennaID': tag.AntennaID,
                'EPC': tag.EPC,
                'LastSeen': parse_date(tag.Timestamp).timestamp() * 1000,
                'ImpinjPeakRSSI': tag.PeakRSSI,
                'Phase Angle': tag.PhaseAngle,
                'Doppler Frequency': tag.DopplerFrequency
            }
            tag_queue.put(tag_data)
            logging.info(f"Received tag: {tag_data}")
        return {"status": "success", "message": f"Received {len(tags)} tags"}
    except Exception as e:
        logging.error(f"Error processing RFID data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing RFID data: {str(e)}")

# FastAPI endpoint to get predictions
@fastapi_app.get("/predictions", response_model=List[PredictionResponse])
async def get_latest_predictions(api_key: str = Depends(api_key_header)):
    try:
        tags = []
        try:
            while True:
                tags.append(tag_queue.get_nowait())
        except queue.Empty:
            pass
        if not tags:
            logging.info("No new tag data available.")
            raise HTTPException(status_code=404, detail="No new tag data available.")

        df = pd.DataFrame(tags)
        pivoted = df.pivot_table(
            index=['LastSeen', 'EPC'],
            columns='AntennaID',
            values=['ImpinjPeakRSSI', 'Phase Angle', 'Doppler Frequency'],
            aggfunc='first'
        ).reset_index()
        pivoted.columns = ['LastSeen', 'EPC'] + [
            f'{col[0].lower().replace(" ", "_")}_{col[1]}' if col[1] else col[0]
            for col in pivoted.columns[2:]
        ]

        for i in range(1, 5):
            pivoted[f'impinjpeakrssi_{i}'] = pivoted.get(f'impinjpeakrssi_{i}', pd.Series([-80.0] * len(pivoted)))
            pivoted[f'phase_angle_{i}'] = pivoted.get(f'phase_angle_{i}', pd.Series([0.0] * len(pivoted)))
            pivoted[f'doppler_frequency_{i}'] = pivoted.get(f'doppler_frequency_{i}', pd.Series([0.0] * len(pivoted)))

        outputs = []
        for _, row in pivoted.iterrows():
            data, prediction = process_tag_data(row)
            output = {
                'epc': row['EPC'],
                'rssi_1': float(data['rssi_1']),
                'rssi_2': float(data['rssi_2']),
                'rssi_3': float(data['rssi_3']),
                'rssi_4': float(data['rssi_4']),
                'phase_angle_1': float(data['phase_angle_1']),
                'phase_angle_2': float(data['phase_angle_2']),
                'phase_angle_3': float(data['phase_angle_3']),
                'phase_angle_4': float(data['phase_angle_4']),
                'doppler_frequency_1': float(data['doppler_frequency_1']),
                'doppler_frequency_2': float(data['doppler_frequency_2']),
                'doppler_frequency_3': float(data['doppler_frequency_3']),
                'doppler_frequency_4': float(data['doppler_frequency_4']),
                'quadrant': str(prediction),
                'timestamp': float(row['LastSeen']),
                'links': {
                    'self': f"/predictions/{row['EPC']}",
                    'all_predictions': '/predictions',
                    'dashboard': '/dashboard'
                }
            }
            outputs.append(output)
            logging.info(f"Read: {output}")
            print(f"EPC: {output['epc']}, RSSI: {output['rssi_1']:.1f}, {output['rssi_2']:.1f}, "
                  f"{output['rssi_3']:.1f}, {output['rssi_4']:.1f}, "
                  f"Phase: {output['phase_angle_1']:.1f}, {output['phase_angle_2']:.1f}, "
                  f"{output['phase_angle_3']:.1f}, {output['phase_angle_4']:.1f}, "
                  f"Doppler: {output['doppler_frequency_1']:.1f}, {output['doppler_frequency_2']:.1f}, "
                  f"{output['doppler_frequency_3']:.1f}, {output['doppler_frequency_4']:.1f}, "
                  f"Quadrant: {output['quadrant']}")
        return outputs
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Dash app
dash_app = Dash(__name__, url_base_pathname='/dashboard/')
quadrants = {'Q1': (3.75, 11.25), 'Q2': (11.25, 11.25), 'Q3': (3.75, 3.75), 'Q4': (11.25, 3.75)}
recent_reads = []

dash_app.layout = html.Div([
    dcc.Graph(id='live-plot'),
    html.Table(
        id='data-table',
        children=[
            html.Tr([html.Th(col) for col in ['EPC', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
                                              'Phase_1', 'Phase_2', 'Phase_3', 'Phase_4',
                                              'Doppler_1', 'Doppler_2', 'Doppler_3', 'Doppler_4',
                                              'Quadrant']]),
        ], style={'width': '100%', 'fontSize': 12}
    ),
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0)
])

@dash_app.callback(
    [Output('live-plot', 'figure'), Output('data-table', 'children')],
    Input('interval-component', 'n_intervals')
)
def update_graph_and_table(n):
    global recent_reads
    try:
        response = requests.get('http://localhost:8000/predictions', headers={'X-API-Key': API_KEY})
        response.raise_for_status()
        data_list = response.json()
        if not data_list:
            raise Exception("No new predictions")
        data = data_list[-1]

        quadrant = data['quadrant']
        x, y = quadrants[quadrant]

        recent_reads.append([
            data['epc'],
            f"{data['rssi_1']:.1f}", f"{data['rssi_2']:.1f}", f"{data['rssi_3']:.1f}", f"{data['rssi_4']:.1f}",
            f"{data['phase_angle_1']:.1f}", f"{data['phase_angle_2']:.1f}",
            f"{data['phase_angle_3']:.1f}", f"{data['phase_angle_4']:.1f}",
            f"{data['doppler_frequency_1']:.1f}", f"{data['doppler_frequency_2']:.1f}",
            f"{data['doppler_frequency_3']:.1f}", f"{data['doppler_frequency_4']:.1f}",
            quadrant
        ])
        recent_reads = recent_reads[-5:]

        table_rows = [
            html.Tr([html.Td(col) for col in ['EPC', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
                                              'Phase_1', 'Phase_2', 'Phase_3', 'Phase_4',
                                              'Doppler_1', 'Doppler_2', 'Doppler_3', 'Doppler_4',
                                              'Quadrant']]),
            *[html.Tr([html.Td(cell) for cell in row]) for row in recent_reads]
        ]

        return {
            'data': [go.Scatter(x=[x], y=[y], mode='markers', marker=dict(size=20, color='blue'))],
            'layout': go.Layout(
                xaxis=dict(range=[0, 15], title='X (ft)'),
                yaxis=dict(range=[0, 15], title='Y (ft)'),
                title='Live Quadrant Prediction (Impinj R420)'
            )
        }, table_rows
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
        return {'data': [], 'layout': go.Layout(title='No Data (Waiting for RFID Reader)')}, [
            html.Tr([html.Td(col) for col in ['EPC', 'RSSI_1', 'RSSI_2', 'RSSI_3', 'RSSI_4',
                                              'Phase_1', 'Phase_2', 'Phase_3', 'Phase_4',
                                              'Doppler_1', 'Doppler_2', 'Doppler_3', 'Doppler_4',
                                              'Quadrant']])
        ]

# Run Dash in a separate thread
def run_dash():
    try:
        dash_app.run(host='0.0.0.0', port=8050, debug=False)
    except Exception as e:
        logging.error(f"Dash server error: {str(e)}")
        print(f"Error starting Dash server: {str(e)}")
        sys.exit(1)

# Check if port is in use
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('0.0.0.0', port)) == 0

# Main function to start polling and FastAPI
async def main():
    asyncio.create_task(poll_rfid_reader())
    asyncio.create_task(send_predictions())
    threading.Thread(target=run_dash, daemon=True).start()
    try:
        print("Starting FastAPI server on port 8000 and Dash dashboard on port 8050...")
        print(f"Polling RFID reader at {RFID_READER_ENDPOINT} every {POLL_INTERVAL} seconds...")
        print(f"Sending predictions to {OUTPUT_ENDPOINT}")
        uvicorn.run(fastapi_app, host="0.0.0.0", port=8000)
    except Exception as e:
        logging.error(f"FastAPI server error: {str(e)}")
        print(f"Error starting FastAPI server: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    if is_port_in_use(8000):
        print("Error: Port 8000 is already in use. Please free it or use a different port.")
        print("To find and kill the process, run: `lsof -i :8000` and `kill -9 <PID>`")
        sys.exit(1)
    if is_port_in_use(8050):
        print("Error: Port 8050 is already in use. Please free it or use a different port.")
        print("To find and kill the process, run: `lsof -i :8050` and `kill -9 <PID>`")
        sys.exit(1)

    asyncio.run(main())