import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

######################################################
#----------------- SETUP AND LOADING -----------------#
######################################################

# Directories and file paths
base_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/'
circle_data_dir = os.path.join(base_dir, 'Testing/MovementTesting/CircleTests/CircleTest1')
model_path = os.path.join(base_dir, 'Models/stationaryModels/best_model.pkl')
scaler_path = os.path.join(base_dir, 'Models/stationaryModels/scaler.pkl')

# Load model and scaler
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print("Model or scaler file not found. Ensure 'best_model.pkl' and 'scaler.pkl' exist.")
    exit(1)

# Load circular test data (use circletest1-1.csv)
test_file = os.path.join(circle_data_dir, 'circletest1-1.csv')
if not os.path.exists(test_file):
    print(f"CSV file not found: {test_file}")
    exit(1)

df = pd.read_csv(test_file)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['test_id'] = 'circletest1-1'

# Debug: Check raw data
print("Raw data shape:", df.shape)
print("Unique antennas:", df['antenna'].unique())

######################################################
#--------------- DATA PROCESSING ----------------#
######################################################

# Pivot data by timestamp and antenna
pivot_data = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean',
    'doppler_frequency': 'mean'
}).unstack()

# Flatten column names
pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]
pivot_data = pivot_data.reset_index()

# Debug: Check pivot data
print("Pivot data shape:", pivot_data.shape)
print("Pivot data columns:", pivot_data.columns.tolist())

# Sort by timestamp
pivot_data = pivot_data.sort_values('timestamp')

# Handle missing values
pivot_data = pivot_data.fillna(pivot_data.mean(numeric_only=True))

# Features for prediction (match static model)
feature_columns = [
    'rssi_Ant1', 'rssi_Ant2', 'rssi_Ant3', 'rssi_Ant4',
    'phase_angle_Ant1', 'phase_angle_Ant2', 'phase_angle_Ant3', 'phase_angle_Ant4',
    'doppler_frequency_Ant1', 'doppler_frequency_Ant2', 'doppler_frequency_Ant3', 'doppler_frequency_Ant4',
]

X = pivot_data[feature_columns]
try:
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
except ValueError as e:
    print(f"Scaler transform error: {e}")
    exit(1)
predictions = model.predict(X_scaled)
pivot_data['pred_x'] = predictions[:, 0]
pivot_data['pred_y'] = predictions[:, 1]

# Debug: Check predictions
print("Predictions shape:", pivot_data[['pred_x', 'pred_y']].shape)
print("Sample predictions (first 5):")
print(pivot_data[['timestamp', 'pred_x', 'pred_y']].head())

if pivot_data.empty or len(pivot_data) < 2:
    print("Error: No valid data for animation. Check CSV content.")
    exit(1)

# Debug: Plot predictions statically
plt.figure(figsize=(8, 8))
plt.scatter(pivot_data['pred_x'], pivot_data['pred_y'], c='red', s=10, label='Predicted')
plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='green', marker='^', s=200, label='Antennas')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Static Plot of Predicted Positions')
plt.legend()
plt.grid(True)
plt.savefig('predictions_static.png')
plt.close()
print("Static predictions plot saved as 'predictions_static.png'")

######################################################
#------------------- ANIMATION -----------------------#
######################################################

# Set up plot
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('Predicted RFID Tag Movement (circletest1-1)')
ax.grid(True)

# Plot antennas
ax.scatter([x for x, y in antennas], [y for x, y in antennas], 
           color='green', marker='^', s=200, label='Antennas')

# Initialize predicted point
pred_point, = ax.plot([], [], 'ro', label='Predicted', markersize=10)
ax.legend()

# Animation initialization
def init():
    pred_point.set_data([], [])
    return pred_point,

# Animation update
def update(frame):
    x = pivot_data['pred_x'].iloc[frame]
    y = pivot_data['pred_y'].iloc[frame]
    if pd.isna(x) or pd.isna(y):
        print(f"Warning: Invalid data at frame {frame}")
        return pred_point,
    pred_point.set_data([x], [y])  # Ensure sequence
    return pred_point,

# Create animation
ani = FuncAnimation(fig, update, frames=len(pivot_data), init_func=init, 
                    blit=True, interval=50)

# Save animation
try:
    ani.save('tag_movement.mp4', writer='ffmpeg', fps=20)
    print("Animation saved as 'tag_movement.mp4'")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure ffmpeg is installed: 'brew install ffmpeg'")

plt.close()