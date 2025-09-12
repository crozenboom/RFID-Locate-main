# Models 

## Moving Models
### circleModel.py
Processes circular test data, calculates distances (RSSI/phase), trilaterates/projects positions, engineers features, trains gradient boosting model for x/y prediction, saves/loads model, predicts/smooths on tests, visualizes paths (plots/animation); CLI radius option.

### linearModel.py
Processes linear test data, extracts antenna features (RSSI/phase), assigns coordinates/quadrants, engineers differences, trains gradient boosting regressor (x/y) and random forest classifier (quadrants), evaluates accuracy (MSE/MAE, confusion matrix), visualizes results (plots), saves models and scaler. 

### train_combo.py
Processes stationary + dynamic linear/circular RFID data, extracts antenna features (RSSI/phase/doppler/channel), engineers differences, assigns coords/quadrants, applies trilateration for position estimates, trains random forest classifier for quadrants, evaluates (MSE/MAE, accuracy, confusion matrix), visualizes predictions/feature importance, saves models, scaler, and plots.

## Quadrant Prediction
### quadPredictLine.py
Trains a RandomForestClassifier on static and linear path data (RSSI, phase, Doppler from 4 antennas).

### quadPredictCircle.py
Trains a RandomForestClassifier on static and circular path data.

### alltogethernow.py
Trains a RandomForestClassifier on static, linear, and circular data combined.

### rfidPredictor.py
Predicts RFID tag quadrants using a simple RSSI-only model with mean RSSI values and two differences (rssi_1_2_diff and rssi_3_4_diff). Logs predicted quadrants and general tag data. Lightweight and suitable for basic quadrant prediction.

### rfidPredictor2.py
Predicts RFID tag quadrants using a full-feature model that incorporates RSSI mean, RSSI standard deviation, Doppler values, and multiple RSSI differences. Logs detailed predictions with confidence scores and maintains a structured CSV for live monitoring. Designed for robust production use.

- Set Speedway Connect to POST to http://localhost:8000/rfid-data at https://192.168.0.219 (login: root/impinj).
- Edit rfidPredictor.py, set API_KEY = "your_key".
- Run: python rfidPredictor.py
- Logs errors to rfid_predictions.log
- to Check:
    - API: curl http://localhost:8000/health
    - Dashboard: http://localhost:8050/dashboard
    - Test POST: `curl -X POST http://localhost:8000/rfid-data -H "Content-Type: application/json" -d '[{"EPC": "EPC_1234", "AntennaID": 1, "PeakRSSI": -60, "PhaseAngle": 1000, "DopplerFrequency": 10, "Timestamp": "2025-07-10T14:30:00Z"}]'

### quadPredictTrain.py
Designed for both static and moving RFID tags, it collects RSSI, phase, and doppler data from multiple antennas. It interpolates positions for moving tags, applies scaling and normalization, balances classes with oversampling, and trains machine learning models (like XGBoost or Random Forest). Outputs include trained models, performance metrics (accuracy, confusion matrix), and plots of predicted vs. actual trajectories. Ideal for tracking real-time movement in 2D/3D spaces.

### quadPredictTrain2.py
Focused only on static tag localization using RSSI data. It fills missing values, normalizes features, and trains a Random Forest model to classify tag positions at predefined reference points. Outputs include the trained model and basic accuracy metrics. It’s simpler, intended for stationary tag location prediction without trajectory estimation.

## Stationary Models
### modelCompare.py
Compares performance of Random Forest and Gradient Boosting models trained on Test 8 static data

### randomForest.py
Trains a Random Forest model on Circle Test 0
