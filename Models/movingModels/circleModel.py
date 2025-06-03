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

# Test files (corrected to match metadata)
test_files = [
    os.path.join(circle_base_dir, 'CircleTest1', 'circletest1-1.csv'),  # Radius 3.5
    os.path.join(circle_base_dir, 'CircleTest2', 'circletest2-1.csv'),  # Radius 4.5
    os.path.join(circle_base_dir, 'CircleTest3', 'circletest3-1.csv'),  # Radius 5.5
    os.path.join(circle_base_dir, 'CircleTest4', 'circletest4-1.csv'),  # Radius 1.5
    os.path.join(circle_base_dir, 'CircleTest5', 'circletest5-1.csv')   # Radius 8.5
]

# Load metadata
try:
    readme_data = pd.read_csv(metadata_file)
    readme_data.columns = readme_data.columns.str.strip()  # Remove whitespace
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
    return 10 ** ((P_tx - rssi - PL_0) / (10 * n))

def phase_to_distance(phase_angle, wavelength=0.33):
    phase_rad = np.deg2rad(phase_angle % 360)
    return (phase_rad * wavelength) / (4 * np.pi)

def trilaterate(distances, antenna_positions):
    def residuals(x, distances, positions):
        return [np.sqrt((x[0] - p[0])**2 + (x[1] - p[1])**2) - d for p, d in zip(positions, distances)]
    initial_guess = [7.5, 7.5]
    bounds = ([0, 0], [15, 15])  # Constrain to 15x15 grid
    result = least_squares(residuals, initial_guess, args=(distances, antenna_positions), bounds=bounds)
    return result.x

def project_to_circle(x, y, center_x=7.5, center_y=7.5, radius=3.5):
    dx = x - center_x
    dy = y - center_y
    dist = np.sqrt(dx**2 + dy**2)
    if dist == 0:
        return center_x + radius, center_y
    scale = radius / dist
    return center_x + dx * scale, center_y + dy * scale

# Angular feature
def compute_angle(x, y, center_x=7.5, center_y=7.5):
    return np.arctan2(y - center_y, x - center_x)

######################################################
#--------------- TRAIN SINGLE MODEL ----------------#
######################################################

model_path = os.path.join(output_dir, 'best_model_dynamic.pkl')
scaler_path = os.path.join(output_dir, 'scaler_dynamic.pkl')

# Train model if it doesn't exist
if not (os.path.exists(model_path) and os.path.exists(scaler_path)):
    print("\nTraining model on all CircleTests")
    all_files = []
    for i in range(1, 6):
        circle_data_dir = os.path.join(circle_base_dir, f'CircleTest{i}')
        files = glob.glob(os.path.join(circle_data_dir, 'circletest*.csv'))
        all_files.extend(files)

    print("Discovered CSV files:", [os.path.basename(f) for f in all_files])
    if not all_files:
        print(f"No CSV files found in {circle_base_dir}")
        exit(1)

    data_list = []
    raw_trilaterated_points = []
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
        print(f"Radius for {meta_key}: {radius}")

        pivot_temp = df.groupby(['test_id', 'timestamp', 'antenna']).agg({
            'rssi': 'mean',
            'phase_angle': 'mean',
            'doppler_frequency': 'mean'
        }).unstack()
        pivot_temp.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_temp.columns]
        pivot_temp = pivot_temp.reset_index()

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
                raw_trilaterated_points.append({'x': x, 'y': y, 'radius': radius})
                x_true, y_true = project_to_circle(x, y, radius=radius)
                row['x_coord'] = x_true
                row['y_coord'] = y_true
                distances.append(row)
            except:
                continue

        if distances:
            data_list.append(pd.DataFrame(distances))

    if not data_list:
        print("No valid data loaded. Exiting.")
        exit(1)

    all_data = pd.concat(data_list, ignore_index=True)

    # Debug: Data distribution
    print("Raw data shape:", all_data.shape)
    print("Unique tests:", all_data['test_id'].unique())
    print("Training data quadrant counts:")
    print("Bottom-left (x<7.5, y<7.5):", ((all_data['x_coord'] < 7.5) & (all_data['y_coord'] < 7.5)).sum())
    print("Bottom-right (x>7.5, y<7.5):", ((all_data['x_coord'] > 7.5) & (all_data['y_coord'] < 7.5)).sum())
    print("Top-left (x<7.5, y>7.5):", ((all_data['x_coord'] < 7.5) & (all_data['y_coord'] > 7.5)).sum())
    print("Top-right (x>7.5, y>7.5):", ((all_data['x_coord'] > 7.5) & (all_data['y_coord'] > 7.5)).sum())

    # Plot raw trilaterated points
    raw_tril_df = pd.DataFrame(raw_trilaterated_points)
    plt.figure(figsize=(10, 10))
    colors = ['red', 'blue', 'green', 'purple', 'orange']
    radii = [3.5, 4.5, 5.5, 1.5, 8.5]
    for i, radius in enumerate(radii):
        subset = raw_tril_df[raw_tril_df['radius'] == radius]
        plt.scatter(subset['x'], subset['y'], c=colors[i], s=10, label=f'Radius {radius}', alpha=0.3)
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
    plt.close()

    # Feature engineering
    all_data['time_diff'] = all_data.groupby('test_id')['timestamp'].diff().dt.total_seconds().fillna(0)
    for ant in range(1, 5):
        for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
            all_data[f'{col}_lag1'] = all_data.groupby('test_id')[col].shift(1).fillna(all_data[col].mean())
            all_data[f'{col}_norm'] = (all_data[col] - all_data[col].mean()) / all_data[col].std()
    all_data['angle'] = compute_angle(all_data['x_coord'], all_data['y_coord'])
    for i, (ax, ay) in enumerate(antennas, 1):
        all_data[f'distance_Ant{i}'] = np.sqrt((all_data['x_coord'] - ax)**2 + (all_data['y_coord'] - ay)**2)

    all_data = all_data.fillna(all_data.mean(numeric_only=True))

    # Debug: Features
    print("Processed data shape:", all_data.shape)
    print("Columns:", all_data.columns.tolist())

    # Train model
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
        'angle'
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

    os.makedirs(output_dir, exist_ok=True)
    with open(model_path, 'wb') as file:
        pickle.dump(best_model, file)
    with open(scaler_path, 'wb') as file:
        pickle.dump(scaler, file)
    print(f"Model and scaler saved as '{model_path}' and '{scaler_path}'")

# Load model and scaler
try:
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    with open(scaler_path, 'rb') as file:
        scaler = pickle.load(file)
except FileNotFoundError:
    print(f"Model or scaler file not found: {model_path}, {scaler_path}")
    exit(1)

######################################################
#--------------- PREDICT AND VISUALIZE ----------------#
######################################################

# Process test files for each radius
all_predictions = []
for test_file in test_files:
    if not os.path.exists(test_file):
        print(f"Test file not found: {test_file}. Skipping.")
        continue

    filename = os.path.basename(test_file).replace('.csv', '')
    meta_key = filename.lower()
    meta = readme_data[readme_data['raw_CSV_filename'] == meta_key]
    if meta.empty:
        print(f"No metadata for {meta_key}. Skipping.")
        continue

    radius = meta['radius_true'].iloc[0]
    print(f"\nPredicting for {filename}, Radius {radius}")

    df = pd.read_csv(test_file)
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

    pivot_data['time_diff'] = pivot_data['timestamp'].diff().dt.total_seconds().fillna(0)
    for ant in range(1, 5):
        for col in [f'rssi_Ant{ant}', f'phase_angle_Ant{ant}', f'doppler_frequency_Ant{ant}']:
            pivot_data[f'{col}_lag1'] = pivot_data[col].shift(1).fillna(pivot_data[col].mean())
            pivot_data[f'{col}_norm'] = (pivot_data[col] - pivot_data[col].mean()) / pivot_data[col].std()
    for i, (ax, ay) in enumerate(antennas, 1):
        pivot_data[f'distance_Ant{i}'] = np.sqrt((7.5 - ax)**2 + (7.5 - ay)**2)

    pivot_data = pivot_data.fillna(pivot_data.mean(numeric_only=True))

    pivot_data['raw_x'] = np.nan
    pivot_data['raw_y'] = np.nan
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
            pivot_data.at[idx, 'raw_x'] = x
            pivot_data.at[idx, 'raw_y'] = y
            x_proj, y_proj = project_to_circle(x, y, radius=radius)
            pivot_data.at[idx, 'pred_x'] = x_proj
            pivot_data.at[idx, 'pred_y'] = y_proj
        except:
            continue

    pivot_data['angle'] = compute_angle(pivot_data['pred_x'], pivot_data['pred_y'])
    X = pivot_data[feature_columns]
    try:
        X_scaled = pd.DataFrame(scaler.transform(X), columns=X.columns)
    except ValueError as e:
        print(f"Scaler transform error for {filename}: {e}")
        continue

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

# Debug: Predictions
print("\nTotal predictions shape:", all_predictions_df[['pred_x', 'pred_y']].shape)
print("Sample predictions (first 5):")
print(all_predictions_df[['timestamp', 'raw_x', 'raw_y', 'pred_x', 'pred_y', 'radius']].head())

# Static plot with raw and projected points
plt.figure(figsize=(12, 12))
colors = ['red', 'blue', 'green', 'purple', 'orange']
radii = [3.5, 4.5, 5.5, 1.5, 8.5]
theta = np.linspace(0, 2*np.pi, 100)
for i, radius in enumerate(radii):
    subset = all_predictions_df[all_predictions_df['radius'] == radius]
    plt.scatter(subset['raw_x'], subset['raw_y'], c=colors[i], marker='x', s=20, alpha=0.3, label=f'Raw (R={radius})')
    plt.scatter(subset['pred_x'], subset['pred_y'], c=colors[i], marker='o', s=20, alpha=0.8, label=f'Projected (R={radius})')
    plt.plot(7.5 + radius * np.cos(theta), 7.5 + radius * np.sin(theta), c=colors[i], ls='--', lw=1, label=f'Ideal R={radius}')
plt.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('RFID Tag Movement: Raw Trilateration vs. Circle Projection')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'predictions_static.png'), bbox_inches='tight')
print("Static predictions plot saved as 'predictions_static.png'")
plt.close()

# Animation with raw and projected points
fig, ax = plt.subplots(figsize=(12, 12))
ax.set_xlim(0, 15)
ax.set_ylim(0, 15)
ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_title('RFID Tag Movement: Raw Trilateration vs. Circle Projection')
ax.grid(True)
ax.scatter([x for x, y in antennas], [y for x, y in antennas], c='black', marker='^', s=200, label='Antennas')

# Initialize points
raw_points = [ax.plot([], [], 'x', c=colors[i], alpha=0.3, label=f'Raw (R={radius})', markersize=10)[0] for i, radius in enumerate(radii)]
proj_points = [ax.plot([], [], 'o', c=colors[i], alpha=0.8, label=f'Proj (R={radius})', markersize=10)[0] for i, radius in enumerate(radii)]
ideal_lines = [ax.plot(7.5 + radius * np.cos(theta), 7.5 + radius * np.sin(theta), c=colors[i], ls='--', lw=1, label=f'Ideal R={radius}')[0] for i, radius in enumerate(radii)]
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
            proj_x, proj_y = subset['pred_x'].iloc[frame], subset['pred_y'].iloc[frame]
            if not (pd.isna(raw_x) or pd.isna(raw_y)):
                raw_points[i].set_data([raw_x], [raw_y])
            if not (pd.isna(proj_x) or pd.isna(proj_y)):
                proj_points[i].set_data([proj_x], [proj_y])
    return raw_points + proj_points

min_length = min(len(all_predictions_df[all_predictions_df['radius'] == r]) for r in radii)
ani = FuncAnimation(fig, update, frames=min_length, init_func=init, blit=True, interval=50)
try:
    ani.save(os.path.join(output_dir, 'tag_movement.mp4'), writer='ffmpeg', fps=20)
    print("Animation saved as 'tag_movement.mp4'")
except Exception as e:
    print(f"Error saving animation: {e}")
    print("Ensure ffmpeg is installed: 'brew install ffmpeg'")

plt.close()