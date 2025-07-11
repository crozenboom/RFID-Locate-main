import requests
import time
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000/rfid-data"
API_KEY = "cachengo"  # Replace with your actual API key
HEADERS = {"X-API-Key": API_KEY}

# Mock RFID data
mock_tags = [
    {
        "EPC": "E2801160600002040A1B2C3D",
        "AntennaID": 1,
        "PeakRSSI": -60.5,
        "PhaseAngle": 45.0,
        "DopplerFrequency": 0.1,
        "Timestamp": datetime.utcnow().isoformat()
    },
    {
        "EPC": "E2801160600002040A1B2C3D",
        "AntennaID": 2,
        "PeakRSSI": -65.0,
        "PhaseAngle": 50.0,
        "DopplerFrequency": 0.2,
        "Timestamp": datetime.utcnow().isoformat()
    },
    {
        "EPC": "E2801160600002040A1B2C3D",
        "AntennaID": 3,
        "PeakRSSI": -62.0,
        "PhaseAngle": 48.0,
        "DopplerFrequency": 0.15,
        "Timestamp": datetime.utcnow().isoformat()
    },
    {
        "EPC": "E2801160600002040A1B2C3D",
        "AntennaID": 4,
        "PeakRSSI": -63.5,
        "PhaseAngle": 47.0,
        "DopplerFrequency": 0.12,
        "Timestamp": datetime.utcnow().isoformat()
    }
]

def send_mock_data():
    try:
        response = requests.post(API_URL, json=mock_tags, headers=HEADERS)
        response.raise_for_status()
        print(f"Successfully sent mock data: {response.json()}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending mock data: {str(e)}")

if __name__ == "__main__":
    print("Sending mock RFID data to /rfid-data endpoint...")
    while True:
        send_mock_data()
        time.sleep(1)  # Simulate periodic data from RFID reader