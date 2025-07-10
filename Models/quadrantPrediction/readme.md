RFID Quadrant Prediction
Predicts RFID tag locations in a 15x15 ft grid (Q1–Q4) using an Impinj R420 reader and RandomForestClassifier. Scripts train models on static, linear, or circular data or run real-time predictions.
Scripts

quadPredictLine.py

What: Trains a RandomForestClassifier on static and linear path data (RSSI, phase, Doppler from 4 antennas).
Run:python quadPredictLine.py


Needs: Dataset (e.g., CSV) with static and linear path data.


quadPredictCircle.py

What: Trains a RandomForestClassifier on static and circular path data.
Run:python quadPredictCircle.py


Needs: Dataset with static and circular path data.


alltogethernow.py

What: Trains a RandomForestClassifier on static, linear, and circular data combined.
Run:python alltogethernow.py


Needs: Combined dataset from linear and circular paths.


rfidPredictor.py

What: Runs real-time quadrant prediction using HTTP POST data from Speedway Connect, serves via FastAPI, and shows results on a Dash dashboard.
Run:
Set Speedway Connect to POST to http://localhost:8000/rfid-data at https://192.168.0.219 (login: root/impinj).
Edit rfidPredictor.py, set API_KEY = "your_key".
Run:python rfidPredictor.py


Check:
API: curl http://localhost:8000/health
Dashboard: http://localhost:8050/dashboard
Test POST: `curl -X POST http://localhost:8000/rfid-data -H "Content-Type: application/json" -d '[{"EPC": "EPC_1234", "AntennaID": 1, "PeakRSSI": -60, "PhaseAngle": 1000, "DopplerFrequency": 10, "Timestamp": "2025-07-10T14:30:00Z"}]'







Setup

Go to quadrantPrediction folder.
Create venv:python -m venv venv
source venv/bin/activate


Install:pip install fastapi uvicorn joblib pandas numpy dash python-dateutil scikit-learn


Ensure rfid_model.pkl and scaler.pkl are present.

Notes

Logs errors to rfid_predictions.log (for rfidPredictor.py).
Free ports if stuck: lsof -i :8000; kill -9 <PID>.
