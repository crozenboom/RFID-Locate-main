import pandas as pd
import numpy as np
from joblib import load

# Load CSV file
csv_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/WalkingTest/CircleTest2.csv'  # Replace with your CSV file path
data = pd.read_csv(csv_file)

# Load scaler and model
scaler = load('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Models/quadrantPrediction/scaler.pkl')  # Replace with your scaler file path
model = load('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Models/quadrantPrediction/rfid_model.pkl')  # Replace with your model file path

# Preprocess data: Pivot to get RSSI and phase_angle per reader/antenna
# Combine reader and antenna into a single identifier (e.g., '192.168.0.219:5084_1')
data['reader_antenna'] = data['reader'] + '_' + data['antenna'].astype(str)

# Pivot RSSI and phase_angle
rssi_pivot = data.pivot(index='timestamp', columns='reader_antenna', values='rssi').fillna(-100)
phase_pivot = data.pivot(index='timestamp', columns='reader_antenna', values='phase_angle').fillna(0)

# Combine features into a single DataFrame
features = rssi_pivot.join(phase_pivot, rsuffix='_phase')

# Select feature columns (adjust to match scaler/model training)
# Example: ['192.168.0.219:5084_1', '192.168.0.219:5084_1_phase', ...]
feature_columns = features.columns  # Use all columns or specify subset
# If your scaler/model expects specific columns, list them, e.g.:
# feature_columns = ['192.168.0.219:5084_1', '192.168.0.219:5084_1_phase']

# Apply scaler to features
scaled_features = scaler.transform(features[feature_columns])

# Make predictions
predictions = model.predict(scaled_features)

# Add predictions to DataFrame
features['predicted_quadrant'] = predictions

# Save results to a new CSV
output_file = 'rfid_predictions.csv'
features[['predicted_quadrant']].to_csv(output_file)

# Print summary of predictions
print("Prediction Summary:")
print(features['predicted_quadrant'].value_counts(dropna=False))

# Optional: If you have ground-truth labels in the CSV (e.g., 'true_quadrant')
if 'true_quadrant' in data.columns:
    # Merge ground truth by timestamp
    ground_truth = data[['timestamp', 'true_quadrant']].drop_duplicates()
    results = features.join(ground_truth.set_index('timestamp'), how='left')
    accuracy = (results['predicted_quadrant'] == results['true_quadrant']).mean()
    print(f"Accuracy: {accuracy:.2%}")