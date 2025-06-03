import pandas as pd
import numpy as np
import os
import glob
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

######################################################
#----------------- SETUP AND METADATA ----------------#
######################################################

# Directories
base_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/'
circle_base_dir = os.path.join(base_dir, 'Testing/MovementTesting/CircleTests')
metadata_file = os.path.join(circle_base_dir, 'CircleMetadata.csv')
output_dir = os.path.join(base_dir, 'Models/movingModels')
test_file = os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-1.csv')  # User-specified test file

# Load metadata
try:
    readme_data = pd.read_csv(metadata_file)
except FileNotFoundError:
    print(f"Metadata file not found: {metadata_file}")
    exit(1)

# Antennas
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Trilateration functions
def rssi_to_distance(rssi, P_tx=0, PL_0=-40, n=2):
    return 10 ** ((P_tx - rssi - PL_0) / (10 * n))

def phase_to_distance(phase_angle, wavelength=0.33):
    phase_rad = np.deg2rad(phase_angle % 360)
    return (phase_rad * wavelength) / (4 * np.pi)

def trilaterate(distances, antenna_positions):
    def residuals(x, distances, positions):
        return [np.sqrt((x[0] - p[0])**2 + (x[1] - p[1])**2) - d for p, d in zip(positions, distances)]
    initial_guess = [7.5, 7.5]
    result = least_squares(residuals, initial_guess, args=(distances, antenna_positions))
    return result.x

def project_to_circle(x, y, center_x=7.5, center_y=7.5, radius=3.5):
    dx = x - center_x
    dy = y - center_y
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return center_x + radius, center_y
    scale = radius / dist
    return center_x + dx * scale, center_y + dy * scale

# Debug: Check metadata
print("Metadata shape:", readme_data.shape)
print("Metadata columns:", readme_data.columns.tolist())

######################################################
#--------------- TRAIN MODELS ----------------#
######################################################

def train_circle_test(circle_test_num):
    model_path = os.path.join(output_dir, f'best_model_dynamic_CircleTest{circle_test_num}.pkl')
    scaler_path = os.path.join(output_dir, f'scaler_dynamic_CircleTest{circle_test_num}.pkl')
    
    # Skip if model exists
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        print(f"Model and scaler for CircleTest{circle_test_num} already exist. Skipping training.")
        return

    print(f"\nTraining model for CircleTest{circle_test_num}")
    circle_data_dir = os.path.join(circle_base_dir, f'CircleTest{circle_test_num}')
    files = glob.glob(os.path.join(circle_data_dir, 'circletest*.csv'))
    
    if not files:
        print(f"No CSV files found in {circle_data_dir}")
        return

    data_list = []
    for file in files:
        filename = os.path.basename(file).replace('.csv', '')
        meta = readme_data[readme_data['raw_CSV_filename'] == filename]
        if meta.empty:
            print(f"No metadata for {filename}. Skipping.")
            continue

        radius = meta['radius_true'].iloc[0]
        df = pd.read_csv(file)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df['test_id'] = filename

        # Pivot to get features per antenna
        pivot_temp = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
            'rssi': 'mean',
            'phase_angle': 'mean',
            'doppler_frequency': 'mean'
        }).unstack()
        pivot_temp.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_temp.columns]
        pivot_temp = pivot_temp.reset_index()

        # Compute true (x, y) via trilateration and projection
        distances = []
        for idx, row in pivot_temp.iterrows():
            dists = []
            for i in range(1, 5):
                rssi = row[f'rssi_Ant{i}']
                phase = row[f'phase_angle_Ant{i}']
                if pd.isna(rssi) or pd.isna(phase):
                    dists.append(np.nan)
                    continue
                d_rssi = rssi_to_distance(rssi)
                d_phase = phase_to_distance(phase)
                d = 0.7 * d_rssi + 0.3 * d_phase
                dists.append(d)
            if all(np.isnan(dists)):
                continue
            dists = np.nan_to_num(dists, nan=np.nanmean(dists))
            try:
                x, y = trilaterate(dists, antennas)
                x_true, y_true = project_to_circle(x, y, radius=radius)
                row['x_coord'] = x_true
                row['y_coord'] = y_true
                distances.append(row)
            except:
                continue
        
        if distances:
            data_list.append(pd.DataFrame(distances))

    if not data_list:
        print(f"No valid data for CircleTest{circle_test_num}")
        return

    all_data = pd.concat(data_list, ignore_index=True)

    # Debug: Check data
    print(f"CircleTest{circle_test_num} raw data shape:", all_data.shape)
    print(f"Unique tests:", all_data['test_id'].unique())

    # Feature engineering
    all_data['time_diff'] = all_data.groupby('test_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    for ant in range(1, 5):
        for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
            all_data[f'{col}_lag1'] = all_data.groupby('test_id')[col].shift(1).fillna(all_data[col].mean())

    for i, (ax, ay) in enumerate(antennas, 1):
        all_data[f'distance_Ant{i}'] = np.sqrt((all_data['x_coord'] - ax)**2 + (all_data['y_coord'] - ay)**2)

    all_data = all_data.fillna(all_data.mean(numeric_only=True))

    # Debug: Features
    print(f"CircleTest{circle_test_num} processed data shape:", all_data.shape)
    print("Columns:", all_data.columns.tolist())

    # Train model
    feature_columns = [
        'rssi_Ant1', 'rssi_Ant2', 'rssi_Ant3', 'rssi_Ant4',
        'phase_angle_Ant1', 'phase_angle_Ant2', 'phase_angle_Ant3', 'phase_angle_Ant4',
        'doppler_frequency_Ant1', 'doppler_frequency_Ant2', 'doppler_frequency_Ant3', 'doppler_frequency_Ant4',
        'time_diff',
        'rssi_Ant1_lag1', 'rssi_Ant2_lag1', 'rssi_Ant3_lag1', 'rssi_Ant4_lag1',
        'phase_angle_Ant1_lag1', 'phase_angle_Ant2_lag1', 'phase_angle_Ant3_lag1', 'phase_angle_Ant4_lag1',
        'doppler_frequency_Ant1_lag1', 'doppler_frequency_Ant2_lag1', 'doppler_frequency_Ant3_lag1', 'doppler_frequency_Ant4_lag1'
    ]
    X = all_data[feature_columns]
    y = all_data[['x_coord', 'y_coord']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

    model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
    param_grid = {
        'estimator__n_estimators': [100, 200],
        'estimator__learning_rate': [0.01, 0.05],
        'estimator__max_depth': [3, 4]
    }

    print(f"Tuning GradientBoosting for CircleTest{circle_test_num}...")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)

    best_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
    mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
    print(f"Test MSE (x, y): {mse[0]:.2f}, {mse[1]:.2f}")
    print(f"Test MAE (x, y): {mae[0]:.2f}, {mae[1]:.2f}")

    os.makedirs(output_dir, exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"Model and scaler saved as '{model_path}' and '{scaler_path}'")

# Train models for all CircleTests
for i in range(1, 6):
    train_circle_test(i)

######################################################
#--------------- PREDICT AND VISUALIZE ----------------#
######################################################

# Determine CircleTest for test file
filename = os.path.basename(test_file).replace('.csv', '')
meta = readme_data[readme_data['raw_CSV_filename'] == filename]
if meta.empty:
    print(f"No metadata for {filename}")
    exit(1)
circle_test_num = int(filename.split('-')[0].replace('circletest', ''))
radius = meta['radius_true'].iloc[0]

# Load model and scaler
model_path = os.path.join(output_dir, f'best_model_dynamic_CircleTest{circle_test_num}.pkl')
scaler_path = os.path.join(output_dir, f'scaler_dynamic_CircleTest{circle_test_num}.pkl')
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print(f"Model or scaler file not found: {model_path}, {scaler_path}")
    exit(1)

# Load test data
if not os.path.exists(test_file):
    print(f"CSV file not found: {test_file}")
    exit(1)

df = pd.read_csv(test_file)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
df['test_id'] = filename

# Debug: Check raw data
print("\nTest data shape:", df.shape)
print("Unique antennas:", df['antenna'].unique())

# Pivot data
pivot_data = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean',
    'doppler_frequency': 'mean'
}).unstack()
pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]
pivot_data = pivot_data.reset_index()

# Debug: Check pivot data
print("Pivot data shape:", pivot_data.shape)
print("Pivot data columns:", pivot_data.columns.tolist())

# Sort by timestamp
pivot_data = pivot_data.sort_values('timestamp')

# Add temporal features
pivot_data['time_diff'] = pivot_data['timestamp'].diff().dt.total_seconds().fillna(0)
for ant in range(1, 5):
    for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
        pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1).fillna(pivot_data[col].mean())

# Add distance features (initially use center point)
for i, (ax, ay) in enumerate(antennas, 1):
    pivot_data[f'distance_Ant{i}'] = np.sqrt((7.5 - ax)**2 + (7.5 - ay)**2)

# Handle missing values
pivot_data = pivot_data.fillna(pivot_data.mean(numeric_only=True))

# Features for prediction
feature_columns = [
    'rssi_Ant1', 'rssi_Ant2', 'rssi_Ant3', 'rssi_Ant4',
    'phase_angle_Ant1', 'phase_angle_Ant2', 'phase_angle_Ant3', 'phase_angle_Ant4',
    'doppler_frequency_Ant1', 'doppler_frequency_Ant2', 'doppler_frequency_Ant3', 'doppler_frequency_Ant4',
    'time_diff',
    'rssi_Ant1_lag1', 'rssi_Ant2_lag1', 'rssi_Ant3_lag1', 'rssi_Ant4_lag1',
    'phase_angle_Ant1_lag1', 'phase_angle_Ant2_lag1', 'phase_angle_Ant3_lag1', 'phase_angle_Ant4_lag1',
    'doppler_frequency_Ant1_lag1', 'doppler_frequency_Ant2_lag1', 'doppler_frequency_Ant3_lag1', 'doppler_frequency_Ant4_lag1'
]

if not all(col in pivot_data.columns for col in feature_columns):
    missing = [col for col in feature_columns if col not in pivot_data.columns]
    print(f"Missing features: {missing}")
    exit(1)

X = pivot_data[feature_columns]

# Debug: Check features
print("Feature data shape:", X.shape)
print("Scaler expected features:", scaler.feature_names_in_)

# Scale features
try:
    X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
except ValueError as e:
    print(f"Scaler transform error: {e}")
    exit(1)

# Predict (x, y)
predictions = model.predict(X_scaled)
pivot_data['pred_x'] = predictions[:, 0]
pivot_data['pred_y'] = predictions[:, 1]

# Refine with trilateration and circle projection
for idx, row in pivot_data.iterrows():
    dists = []
    for i in range(1, 5):
        rssi = row[f'rssi_Ant{i}']
        phase = row[f'phase_angle_Ant{i}']
        if pd.isna(rssi) or pd.isna(phase):
            dists.append(np.nan)
            continue
        d_rssi = rssi_to_distance(rssi)
        d_phase = phase_to_distance(phase)
        d = 0.7 * d_rssi + 0.3 * d_phase
        dists.append(d)
    if all(np.isnan(dists)):
        continue
    dists = np.nan_to_num(dists, nan=np.nanmean(dists))
    try:
        x, y = trilaterate(dists, antennas)
        x_proj, y_proj = project_to_circle(x, y, radius=radius)
        pivot_data.at[idx, 'pred_x'] = x_proj
        pivot_data.at[idx, 'pred_y'] = y_proj
    except:
        continue

# Smooth predictions
pivot_data['pred_x'] = pivot_data['pred_x'].rolling(window=5, min_periods=1, center=True).mean()
pivot_data['pred_y'] = pivot_data['pred_y'].rolling(window=5, min_periods=1, center=True).mean()

# Debug: Check predictions
print("Predictions shape:", pivot_data[['pred_x', 'pred_y']].shape)
print("Sample predictions (first 5):")
print(pivot_data[['timestamp', 'pred_x', 'pred_y']].head())

if pivot_data.empty or len(pivot_data) < 2:
    print("Error: No valid data for animation. Check CSV content.")
    exit(1)

# Static plot
plt.figure(figsize=(8, 8))
plt.scatter(pivot_data['pred_x'], pivot_data['pred_y'], c='red', s=10, label='Predicted')
plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='green', marker='^', s=200, label='Antennas')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'Predicted RFID Tag Movement ({filename}, Radius {radius})')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(output_dir, 'predictions_static.png'))
print("Static predictions plot saved as 'predictions_static.png'")
plt.close()

# Animation
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title(f'Predicted RFID Tag Movement ({filename}, Radius {radius})')
ax.grid(True)
ax.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
pred_point, = ax.plot([], [], 'ro', label='Predicted', markersize=10)
ax.legend()

def init():
    pred_point.set_data([], [])
    return pred_point,

def update(frame):
    x = pivot_data['pred_x'].iloc[frame]
    y = pivot_data['pred_y'].iloc[frame]
    if pd.isna(x) or pd.isna(y):
        print(f"Warning: Invalid data at frame {frame}")
        return pred_point,
    pred_point.set_data([x], [y])
    return pred_point,

ani = FuncAnimation(fig, update, frames=len(pivot_data), init_func=init, blit=True, interval=50)
try:
    ani.save(os.path.join(output_dir, 'tag_movement.mp4'), writer='ffmpeg', fps=20)
    print("Animation saved as 'tag_movement.mp4'")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure ffmpeg is installed: 'brew install ffmpeg'")

plt.close()