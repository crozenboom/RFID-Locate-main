import pandas as pd
import numpy as np
import os
import pickle
import sys
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

# Test files (all are clockwise)
test_files = [
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-1.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-2.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-3.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-4.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-5.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-6.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-7.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-8.csv'), 'radius': 3.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest2', 'circletest2-1.csv'), 'radius': 4.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest2', 'circletest2-2.csv'), 'radius': 4.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest3', 'circletest3-1.csv'), 'radius': 5.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest3', 'circletest3-2.csv'), 'radius': 5.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest4', 'circletest4-1.csv'), 'radius': 1.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest4', 'circletest4-2.csv'), 'radius': 1.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest5', 'circletest5-1.csv'), 'radius': 8.5},
    {'path': os.path.join(circle_base_dir, 'CircleTest5', 'circletest5-2.csv'), 'radius': 8.5}
]

# Handle command-line argument
selected_radius = None
if len(sys.argv) > 1:
    try:
        radius_map = {1: 3.5, 2: 4.5, 3: 5.5, 4: 1.5, 5: 8.5}
        selected_radius = radius_map[int(sys.argv[1])]
        print(f"Selected radius: {selected_radius}")
        test_files = [f for f in test_files if f['radius'] == selected_radius]
    except (ValueError, KeyError):
        print("Invalid argument. Use 1 (3.5), 2 (4.5), 3 (5.5), 4 (1.5), or 5 (8.5). Predicting all radii.")
        selected_radius = None

# Feature columns
feature_columns = [
    'rssi_Ant1', 'rssi_Ant2', 'rssi_Ant3', 'rssi_Ant4',
    'phase_angle_Ant1', 'phase_angle_Ant2', 'phase_angle_Ant3', 'phase_angle_Ant4',
    'doppler_frequency_Ant1', 'doppler_frequency_Ant2', 'doppler_frequency_Ant3', 'doppler_frequency_Ant4',
    'rssi_Ant1_norm', 'rssi_Ant2_norm', 'rssi_Ant3_norm', 'rssi_Ant4_norm',
    'phase_angle_Ant1_norm', 'phase_angle_Ant2_norm', 'phase_angle_Ant3_norm', 'phase_angle_Ant4_norm',
    'doppler_frequency_Ant1_norm', 'doppler_frequency_Ant2_norm', 'doppler_frequency_Ant3_norm', 'doppler_frequency_Ant4_norm',
    'time_diff',
    'rssi_Ant1_lag1', 'rssi_Ant2_lag1', 'rssi_Ant3_lag1', 'rssi_Ant4_lag1',
    'phase_angle_Ant1_lag1', 'phase_angle_Ant2_lag1', 'phase_angle_Ant3_lag1', 'phase_angle_Ant4_lag1',
    'doppler_frequency_Ant1_lag1', 'doppler_frequency_Ant2_lag1', 'doppler_frequency_Ant3_lag1', 'doppler_frequency_Ant4_lag1',
    'angle',
    'angular_velocity'
]

# Load metadata
try:
    readme_data = pd.read_csv(metadata_file)
    readme_data.columns = readme_data.columns.str.strip()
except FileNotFoundError:
    print(f"Metadata file not found: {metadata_file}")
    exit(1)

# Debug: Check metadata
print("Metadata shape:", readme_data.shape)
print("Metadata columns:", readme_data.columns.tolist())
print("Unique raw_CSV_filename:", readme_data['raw_CSV_filename'].unique())

# Antennas
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Trilateration functions
def rssi_to_distance(rssi, P_tx=0, PL_0=-30, n=2.5):
    distance = 10 ** ((P_tx - rssi - PL_0) / (10 * n))
    return np.clip(distance, 0.1, 20)

def phase_to_distance(phase, wavelength=0.33):
    phase_rad = np.deg2rad(phase % 360)
    distance = (phase_rad * wavelength) / (4 * np.pi)
    return np.clip(distance, 0.1, 10)

def trilaterate(distances, antenna_positions):
    def residuals(x, distances, positions):
        return [np.sqrt((x[0] - p[0])**2 + (x[1] - p[1])**2) - d for p, d in zip(positions, distances)]
    initial_guess = [7.5, 7.5]
    bounds = ([0, 0], [15, 15])
    try:
        result = least_squares(residuals, initial_guess, args=(distances, antenna_positions), bounds=bounds)
        return result.x
    except Exception as e:
        print(f"Trilateration error: {e}")
        return [np.nan, np.nan]

def project_to_circle(x, y, center_x=7.5, center_y=7.5, radius=3.5):
    dx = x - center_x
    dy = y - center_y
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return center_x + radius, center_y
    scale = radius / dist
    return center_x + dx * scale, center_y + dy * scale

def compute_angle(x, y, center_x=7.5, center_y=7.5):
    return np.degrees(np.arctan2(y - center_y, x - center_x)) % 360

def compute_angular_velocity(angles, time_diffs):
    angular_velocity = np.zeros_like(angles)
    for i in range(1, len(angles)):
        angle_diff = angles[i] - angles[i-1]
        if angle_diff > 180:
            angle_diff -= 360
        elif angle_diff < -180:
            angle_diff += 360
        angular_velocity[i] = angle_diff / time_diffs[i] if time_diffs[i] > 0 else 0
    return angular_velocity

######################################################
#--------------- TRAIN SINGLE MODEL ----------------#
######################################################

model_path = os.path.join(output_dir, 'best_model_dynamic.pkl')
scaler_path = os.path.join(output_dir, 'scaler_dynamic.pkl')

# Force retraining
if os.path.exists(model_path):
    os.remove(model_path)
if os.path.exists(scaler_path):
    os.remove(scaler_path)

print("\nTraining model on clockwise CircleTests")
all_files = []
for test_file in test_files:
    if os.path.exists(test_file['path']):
        all_files.append(test_file['path'])
    else:
        print(f"File not found: {test_file['path']}")

print("Training files:", [os.path.basename(f) for f in all_files])
if not all_files:
    print(f"No valid CSV files found in {circle_base_dir}")
    exit(1)

data_list = []
raw_trilaterated_points = []
projected_points = []
trilateration_attempts = 0
trilateration_successes = 0

for file in all_files:
    filename = os.path.basename(file).replace('.csv', '')
    meta_key = filename.lower()
    print(f"Processing file: {filename}, meta_key: {meta_key}")

    df = pd.read_csv(file)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['test_id'] = filename

    meta = readme_data[readme_data['raw_CSV_filename'] == meta_key]
    if meta.empty:
        print(f"No metadata for {meta_key}. Skipping.")
        continue

    radius = meta['radius_true'].iloc[0]
    print(f"Radius for {filename}: {radius}")

    pivot_data = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
        'rssi': 'mean',
        'phase_angle': 'mean',
        'doppler_frequency': 'mean'
    }).unstack()
    pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]
    pivot_data = pivot_data.reset_index()

    distances = []
    for idx, row in pivot_data.iterrows():
        trilateration_attempts += 1
        dists = []
        for i in range(1, 5):
            rssi = row[f'rssi_Ant{i}']
            phase = row[f'phase_angle_Ant{i}']
            if pd.isna(rssi) or pd.isna(phase):
                dists.append(np.nan)
                continue
            d_rssi = rssi_to_distance(rssi)
            d_phase = phase_to_distance(phase)
            d = 0.4 * d_rssi + 0.6 * d_phase
            dists.append(d)
        if all(np.isnan(dists)):
            continue
        dists = np.nan_to_num(dists, nan=np.nanmean(dists))
        try:
            x, y = trilaterate(dists, antennas)
            if not np.any(np.isnan([x, y])):
                trilateration_successes += 1
                raw_trilaterated_points.append({'x': x, 'y': y, 'radius': radius})
            x_true, y_true = project_to_circle(x, y, radius=radius)
            projected_points.append({'x': x_true, 'y': y_true, 'radius': radius})
            row['x_coord'] = x_true
            row['y_coord'] = y_true
            distances.append(row)
        except Exception as e:
            print(f"Trilateration failed for {filename} row {idx}: {e}")
            continue

    if distances:
        data_list.append(pd.DataFrame(distances))

if not data_list:
    print("No valid data loaded. Exiting.")
    exit(1)

all_data = pd.concat(data_list, ignore_index=True)

# Feature engineering
all_data = all_data.sort_values(['test_id', 'timestamp'])
all_data['time_diff'] = all_data.groupby('test_id')['timestamp'].diff().dt.total_seconds().fillna(0)
for ant in range(1, 5):
    for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
        all_data[f'{col}_lag1'] = all_data.groupby('test_id')[col].shift(1).fillna(all_data[col].mean())
        all_data[f'{col}_norm'] = (all_data[col] - all_data[col].mean()) / all_data[col].std()
all_data['angle'] = compute_angle(all_data['x_coord'], all_data['y_coord'])
all_data['angular_velocity'] = compute_angular_velocity(all_data['angle'].values, all_data['time_diff'].values)
all_data = all_data.fillna(all_data.mean(numeric_only=True))

# Debug: Check features and data
missing_features = [col for col in feature_columns if col not in all_data.columns]
if missing_features:
    print(f"Missing features in training data: {missing_features}")
    exit(1)
print("Training data features:", all_data.columns.tolist())
print("Raw data shape:", all_data.shape)
print("Unique tests:", all_data['test_id'].unique())
print("Training data quadrant counts:")
print("Bottom-left (x<7.5, y<7.5):", ((all_data['x_coord'] < 7.5) & (all_data['y_coord'] < 7.5)).sum())
print("Bottom-right (x>7.5, y<7.5):", ((all_data['x_coord'] > 7.5) & (all_data['y_coord'] < 7.5)).sum())
print("Top-left (x<7.5, y>7.5):", ((all_data['x_coord'] < 7.5) & (all_data['y_coord'] > 7.5)).sum())
print("Top-right (x>7.5, y>7.5):", ((all_data['x_coord'] > 7.5) & (all_data['y_coord'] > 7.5)).sum())
print(f"Trilateration success rate: {trilateration_successes/trilateration_attempts:.2%} ({trilateration_successes}/{trilateration_attempts})")

# Plot raw trilaterated points
raw_tril_df = pd.DataFrame(raw_trilaterated_points)
print(f"Raw trilaterated points: {len(raw_trilaterated_points)}")
if not raw_tril_df.empty:
    plt.figure(figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    radii = [3.5, 4.5, 5.5, 1.5, 8.5]
    for i, radius in enumerate(radii):
        subset = raw_tril_df[raw_tril_df['radius'] == radius]
        if not subset.empty:
            plt.scatter(subset['x'], subset['y'], c=colors[i], s=50, label=f'Radius {radius}', alpha=0.3)
    plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Raw Trilaterated Points (Before Projection)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'raw_trilateration.png'))
    print("Raw trilateration plot saved as 'raw_trilateration.png'")
else:
    print("No raw trilaterated points to plot.")
plt.close()

# Plot projected points (training targets)
proj_df = pd.DataFrame(projected_points)
print(f"Projected points: {len(projected_points)}")
if not proj_df.empty:
    plt.figure(figsize=(10, 10))
    for i, radius in enumerate(radii):
        subset = proj_df[proj_df['radius'] == radius]
        if not subset.empty:
            plt.scatter(subset['x'], subset['y'], c=colors[i], s=50, label=f'Radius {radius}', alpha=0.3)
    plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Projected Training Targets')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'training_targets.png'))
    print("Training targets plot saved as 'training_targets.png'")
else:
    print("No projected points to plot.")
plt.close()

X = all_data[feature_columns]
y = all_data[['x_coord', 'y_coord']]

# Debug: Check coordinate variation
print("Training coordinate stats:")
print(f"X: Min={y['x_coord'].min():.2f}, Max={y['x_coord'].max():.2f}, Mean={y['x_coord'].mean():.2f}, Std={y['x_coord'].std():.2f}")
print(f"Y: Min={y['y_coord'].min():.2f}, Max={y['y_coord'].max():.2f}, Mean={y['y_coord'].mean():.2f}, Std={y['y_coord'].std():.2f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

model = MultiOutputRegressor(GradientBoostingRegressor(random_state=42))
param_grid = {
    'estimator__n_estimators': [200, 300],
    'estimator__learning_rate': [0.01, 0.05],
    'estimator__max_depth': [4, 5]
}

print("Tuning GradientBoosting...")
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

best_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")

y_pred = best_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
print(f"Test MSE (x, y): {mse[0]:.2f}, {mse[1]:.2f}")
print(f"Test MAE (x, y): {mae[0]:.2f}, {mae[1]:.2f}")

# Debug: Check predicted coordinate variation
print("Predicted coordinate stats:")
print(f"X: Min={y_pred[:, 0].min():.2f}, Max={y_pred[:, 0].max():.2f}, Mean={y_pred[:, 0].mean():.2f}, Std={np.std(y_pred[:, 0]):.2f}")
print(f"Y: Min={y_pred[:, 1].min():.2f}, Max={y_pred[:, 1].max():.2f}, Mean={y_pred[:, 1].mean():.2f}, Std={np.std(y_pred[:, 1]):.2f}")

# Feature importance
for i, est in enumerate(best_model.estimators_):
    importances = pd.Series(est.feature_importances_, index=feature_columns)
    print(f"\nFeature importances for {'x_coord' if i == 0 else 'y_coord'}:")
    print(importances.sort_values(ascending=False).head(10))

os.makedirs(output_dir, exist_ok=True)
with open(model_path, 'wb') as f:
    pickle.dump(best_model, f)
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Model and scaler saved as '{model_path}' and '{scaler_path}'")

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Model or scaler file not found: {model_path}, {scaler_path}")
    exit(1)

######################################################
#--------------- PREDICT AND VISUALIZE ----------------#
######################################################

all_predictions = []
for test_file in test_files:
    file_path = test_file['path']
    expected_radius = test_file['radius']
    if not os.path.exists(file_path):
        print(f"Test file not found: '{file_path}'. Skipping.")
        continue

    filename = os.path.basename(file_path).replace('.csv', '')
    meta_key = filename.lower()
    meta = readme_data[readme_data['raw_CSV_filename'] == meta_key]
    if meta.empty:
        print(f"No metadata for {meta_key}. Skipping.")
        continue

    radius = meta['radius_true'].iloc[0]
    print(f"\nPredicting for {filename}, Radius {radius}")

    df = pd.read_csv(file_path)
    df = df.sort_values('timestamp')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['test_id'] = filename

    print(f"Test data shape for {filename}:", df.shape)
    print("Unique antennas:", df['antenna'].unique())

    pivot_data = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
        'rssi': 'mean',
        'phase_angle': 'mean',
        'doppler_frequency': 'mean'
    }).unstack()
    pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]
    pivot_data = pivot_data.reset_index()

    print(f"Pivot data shape for {filename}:", pivot_data.shape)

    # Feature engineering
    pivot_data['time_diff'] = pivot_data['timestamp'].diff().dt.total_seconds().fillna(0)
    for ant in range(1, 5):
        for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
            pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1).fillna(pivot_data[col].mean())
            pivot_data[f'{col}_norm'] = (pivot_data[col] - all_data[col].mean()) / all_data[col].std()

    pivot_data['raw_x'] = np.nan
    pivot_data['raw_y'] = np.nan
    pivot_data['pred_x_unsmoothed'] = np.nan
    pivot_data['pred_y_unsmoothed'] = np.nan
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
            d = 0.4 * d_rssi + 0.6 * d_phase
            dists.append(d)
        if all(np.isnan(dists)):
            continue
        dists = np.nan_to_num(dists, nan=np.nanmean(dists))
        try:
            x, y = trilaterate(dists, antennas)
            pivot_data.loc[idx, 'raw_x'] = x
            pivot_data.loc[idx, 'raw_y'] = y
        except Exception as e:
            print(f"Prediction trilateration failed for {filename} row {idx}: {e}")
            continue

    pivot_data['angle'] = compute_angle(pivot_data['raw_x'], pivot_data['raw_y'])
    pivot_data['angular_velocity'] = compute_angular_velocity(pivot_data['angle'].values, pivot_data['time_diff'].values)
    pivot_data = pivot_data.fillna(pivot_data.mean(numeric_only=True))

    # Debug: Check features
    missing = [col for col in feature_columns if col not in pivot_data.columns]
    if missing:
        print(f"Missing features for {filename}: {missing}")
        continue

    X = pivot_data[feature_columns]
    try:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
        predictions = model.predict(X_scaled)
        pivot_data['pred_x_unsmoothed'] = predictions[:, 0]
        pivot_data['pred_y_unsmoothed'] = predictions[:, 1]
        pivot_data['pred_x'] = predictions[:, 0]
        pivot_data['pred_y'] = predictions[:, 1]
    except Exception as e:
        print(f"Scaler/prediction error for {filename}: {e}")
        continue

    # Smooth predictions
    pivot_data['pred_x'] = pivot_data['pred_x'].rolling(window=5, min_periods=1, center=True).mean()
    pivot_data['pred_y'] = pivot_data['pred_y'].rolling(window=5, min_periods=1, center=True).mean()
    pivot_data['raw_x'] = pivot_data['raw_x'].rolling(window=5, min_periods=1, center=True).mean()
    pivot_data['raw_y'] = pivot_data['raw_y'].rolling(window=5, min_periods=1, center=True).mean()

    pivot_data['radius'] = radius
    all_predictions.append(pivot_data)

if not all_predictions:
    print("No valid predictions generated. Exiting.")
    exit(1)

all_predictions_df = pd.concat(all_predictions, ignore_index=True)

print("\nTotal predictions shape:", all_predictions_df[['pred_x', 'pred_y']].shape)
print("Sample predictions (first 5):")
print(all_predictions_df[['timestamp', 'raw_x', 'raw_y', 'pred_x', 'pred_y', 'pred_x_unsmoothed', 'pred_y_unsmoothed', 'radius']].head())

# Unsmoothed predictions plot
plt.figure(figsize=(12, 10))
colors = ['red', 'blue', 'green', 'purple', 'orange']
radii = [3.5] if selected_radius else [3.5, 4.5, 5.5, 1.5, 8.5]
color_map = {3.5: 'red', 4.5: 'blue', 5.5: 'green', 1.5: 'purple', 8.5: 'orange'}
theta = np.linspace(0, 2*np.pi, 100)
for radius in radii:
    subset = all_predictions_df[all_predictions_df['radius'] == radius]
    color = color_map[radius]
    plt.scatter(subset['pred_x_unsmoothed'], subset['pred_y_unsmoothed'], c=color, marker='o', s=20, alpha=0.8, label=f'Unsmoothed Predicted (R={radius})')
    plt.plot(7.5 + radius * np.cos(theta), 7.5 + radius * np.sin(theta), c=color, ls='--', lw=1, label=f'Ideal (R={radius})')
plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Unsmoothed Predictions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'unsmoothed_predictions.png'), bbox_inches='tight')
print("Unsmoothed predictions saved as 'unsmoothed_predictions.png'")
plt.close()

# Static plot
plt.figure(figsize=(12, 10))
for radius in radii:
    subset = all_predictions_df[all_predictions_df['radius'] == radius]
    color = color_map[radius]
    plt.scatter(subset['raw_x'], subset['raw_y'], c=color, marker='x', s=20, alpha=0.5, label=f'Raw (R={radius})')
    plt.scatter(subset['pred_x'], subset['pred_y'], c=color, marker='o', s=20, alpha=0.8, label=f'Predicted (R={radius})')
    plt.plot(7.5 + radius * np.cos(theta), 7.5 + radius * np.sin(theta), c=color, ls='--', lw=1, label=f'Ideal (R={radius})')
plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('RFID Tag Movement: Clockwise Circle Projection')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_static.png'), bbox_inches='tight')
print("Static predictions saved as 'predictions_static.png'")
plt.close()

# Animation
fig, ax = plt.subplots(figsize=(12, 10))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('RFID Tag Movement: Clockwise Circle Projection')
ax.grid(True)
ax.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')

raw_points = [ax.plot([], [], 'x', c=color_map[r], alpha=0.5, label=f'Raw (R={r})', markersize=10)[0] for r in radii]
proj_points = [ax.plot([], [], 'o', c=color_map[r], alpha=0.8, label=f'Pred (R={r})', markersize=5)[0] for r in radii]
ideal_lines = [ax.plot(7.5 + r * np.cos(theta), 7.5 + r * np.sin(theta), c=color_map[r], ls='--', lw=1, label=f'Ideal (R={r})')[0] for r in radii]
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

def init():
    for point in raw_points + proj_points:
        point.set_data([], [])
    return raw_points + proj_points

def update(frame):
    for i, radius in enumerate(radii):
        subset = all_predictions_df[all_predictions_df['radius'] == radius]
        if frame < len(subset):
            raw_x, raw_y = subset['raw_x'].iloc[frame], subset['raw_y'].iloc[frame]
            pred_x, pred_y = subset['pred_x'].iloc[frame], subset['pred_y'].iloc[frame]
            if not pd.isna(raw_x) and not pd.isna(raw_y):
                raw_points[i].set_data([raw_x], [raw_y])
            if not pd.isna(pred_x) and not pd.isna(pred_y):
                proj_points[i].set_data([pred_x], [pred_y])
    return raw_points + proj_points

min_length = min(len(all_predictions_df[all_predictions_df['radius'] == r]) for r in radii)
ani = FuncAnimation(fig, update, frames=min_length, init_func=init, interval=20, blit=True)
try:
    ani.save(os.path.join(output_dir, 'tag_movement.mp4'), writer='ffmpeg', fps=30)
    print("Animation saved as 'tag_movement.mp4'")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure ffmpeg is installed: 'brew install ffmpeg'")

plt.close()