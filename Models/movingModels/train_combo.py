import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def get_quadrant(x, y):
    if pd.isna(x) or pd.isna(y):
        return np.nan
    if x > 7.5 and y > 7.5: return 1
    elif x <= 7.5 and y > 7.5: return 2
    elif x <= 7.5 and y <= 7.5: return 3
    else: return 4

def interpolate_line_coords(start_x, start_y, end_x, end_y):
    return (start_x + end_x) / 2, (start_y + end_y) / 2

def rssi_to_distance(rssi, n=2.5, A=-40):
    """Convert RSSI (dBm) to distance (m)."""
    return np.power(10, (A - rssi) / (10 * n))

def trilateration(rssi_values, antenna_positions):
    """Estimate (x, y) using trilateration with least squares."""
    def residual(u, rssi, positions):
        x, y = u
        distances = rssi_to_distance(rssi)
        return np.array([
            np.sqrt((x - px)**2 + (y - py)**2) - d for d, (px, py) in zip(distances, positions)
        ])
    
    # Initial guess: center of area
    x0 = np.array([7.5, 7.5])
    
    # Use valid RSSI values
    valid_mask = ~np.isnan(rssi_values)
    if np.sum(valid_mask) < 3:
        return np.nan, np.nan  # Not enough valid points
    
    rssi_valid = rssi_values[valid_mask]
    positions_valid = [pos for pos, valid in zip(antenna_positions, valid_mask) if valid]
    
    try:
        result = least_squares(residual, x0, args=(rssi_valid, positions_valid), bounds=([0, 0], [15, 15]))
        x, y = result.x
        return x, y
    except:
        return np.nan, np.nan

# Load metadata
line_metadata = pd.read_csv('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv')
circle_metadata = pd.read_csv('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleMetadata.csv')
line_metadata.columns = line_metadata.columns.str.strip()
circle_metadata.columns = circle_metadata.columns.str.strip()
line_metadata = line_metadata[line_metadata['raw_CSV_filename'].str.startswith('movetest')]
circle_metadata = circle_metadata[circle_metadata['raw_CSV_filename'].str.startswith('movetest')]

# Filter metadata
antennas = [(0, 0), (0, 4), (4, 4), (4, 0)]

# Load stationary data
data_dir = '/Users/cash/RFID-Library/Testing/Test8/'
stationary_data_list = []
for file in glob.glob(os.path.join(data_dir, '*.csv')):
    filename = os.path.basename(file)
    try:
        x_coord, y_coord = map(float, filename.replace('.csv', '').strip('()').split(','))
    except ValueError:
        continue
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ['filename', 'antenna', 'rssi', 'phase_angle', 'doppler_frequency', 'channel_index']):
        continue
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    df['test_id'] = filename
    df['data_type'] = 'stationary'
    df['quadrant'] = get_quadrant(x_coord, y_coord)
    df['sample_weight'] = 6.0
    stationary_data_list.append(df)
stationary_data = pd.concat(stationary_data_list, ignore_index=True)

# Load dynamic linear data
dynamic_linear_dir = '/Users/cash/RFID-Library/Testing/MovementTesting/MoveTest/'
dynamic_linear_data_list = []
for _, row in line_metadata.iterrows():
    test_id = row['raw_CSV_filename']
    csv_path = os.path.join(dynamic_linear_dir, f'{test_id}.csv')
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if any(df['antenna'].value_counts().get(ant, 0) < 10 for ant in [1, 2, 3, 4]):
        continue
    x_coord, y_coord = interpolate_line_coords(row['start_x_true'], row['start_y_true'], row['end_x_true'], row['end_y_true'])
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    df['test_id'] = test_id
    df['data_type'] = 'dynamic_linear'
    df['quadrant'] = get_quadrant(x_coord, y_coord)
    df['sample_weight'] = 1.0
    dynamic_linear_data_list.append(df)
dynamic_linear_data = pd.concat(dynamic_linear_data_list, ignore_index=True)

# Load dynamic circular data
dynamic_circular_dir = '/Users/cash/RFID-Library/Testing/MovementTesting/CircleTest/'
dynamic_circular_data_list = []
for _, row in circle_metadata.iterrows():
    test_id = row['raw_CSV_filename']
    csv_path = os.path.join(dynamic_circular_dir, f'{test_id}.csv')
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    if any(df['antenna'].value_counts().get(ant, 0) < 10 for ant in [1, 2, 3, 4]):
        continue
    x_coord = row['center_x_true']
    y_coord = row['center_y_true']
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    df['test_id'] = test_id
    df['data_type'] = 'dynamic_circular'
    df['quadrant'] = get_quadrant(x_coord, y_coord)
    df['sample_weight'] = 1.0
    dynamic_circular_data_list.append(df)
dynamic_circular_data = pd.concat(dynamic_circular_data_list, ignore_index=True)

if dynamic_linear_data.empty and dynamic_circular_data.empty:
    print("No dynamic data loaded.")
    exit(1)

# Pivot datasets
stationary_pivot = stationary_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle', 'doppler_frequency', 'channel_index', 'sample_weight'],
    aggfunc='mean'
).reset_index()
stationary_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' if isinstance(col, tuple) and col[1] else col[0] for col in stationary_pivot.columns]

dynamic_linear_pivot = dynamic_linear_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle', 'doppler_frequency', 'channel_index', 'sample_weight'],
    aggfunc='mean'
).reset_index()
dynamic_linear_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' if isinstance(col, tuple) and col[1] else col[0] for col in dynamic_linear_pivot.columns]

dynamic_circular_pivot = dynamic_circular_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle', 'doppler_frequency', 'channel_index', 'sample_weight'],
    aggfunc='mean'
).reset_index()
dynamic_circular_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' if isinstance(col, tuple) and col[1] else col[0] for col in dynamic_circular_pivot.columns]

# Combine datasets
all_data = pd.concat([stationary_pivot, dynamic_linear_pivot, dynamic_circular_pivot], ignore_index=True)

# Normalize doppler frequency
doppler_cols = [f'doppler_frequency_Ant{i}' for i in range(1, 5)]
for col in doppler_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].clip(lower=-10, upper=10)

# Fill NaNs
feature_cols = [col for col in all_data.columns if col.startswith(('rssi_Ant', 'phase_angle_Ant', 'doppler_frequency_Ant', 'channel_index_Ant'))]
all_data[feature_cols] = all_data[feature_cols].fillna(all_data[feature_cols].mean())

# Add differences
for i in range(1, 4):
    for j in range(i + 1, 5):
        all_data[f'rssi_diff_Ant{i}_Ant{j}'] = all_data[f'rssi_Ant{i}'] - all_data[f'rssi_Ant{j}']
        all_data[f'phase_diff_Ant{i}_Ant{j}'] = all_data[f'phase_angle_Ant{i}'] - all_data[f'phase_angle_Ant{j}']
        if f'doppler_frequency_Ant{i}' in all_data.columns and f'doppler_frequency_Ant{j}' in all_data.columns:
            all_data[f'doppler_diff_Ant{i}_Ant{j}'] = all_data[f'doppler_frequency_Ant{i}'] - all_data[f'doppler_frequency_Ant{j}']
        if f'channel_index_Ant{i}' in all_data.columns and f'channel_index_Ant{j}' in all_data.columns:
            all_data[f'channel_diff_Ant{i}_Ant{j}'] = all_data[f'channel_index_Ant{i}'] - all_data[f'channel_index_Ant{j}']

# Handle numeric NaNs
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

# Prepare features and targets
feature_columns = [col for col in all_data.columns if col.startswith(('rssi_Ant', 'phase_angle_Ant', 'doppler_frequency_Ant', 'channel_index_Ant', 'rssi_diff_Ant', 'phase_diff_Ant', 'doppler_diff_Ant', 'channel_diff_Ant'))]
X = all_data[feature_columns]
y = all_data[['x_coord', 'y_coord', 'quadrant']]
sample_weights = all_data['sample_weight_Ant1'].fillna(1.0).values

# Trilateration for coordinates
rssi_cols = [f'rssi_Ant{i}' for i in range(1, 5)]
y_pred_trilateration = []
for idx in X.index:
    rssi_values = all_data.loc[idx, rssi_cols].values
    x, y = trilateration(rssi_values, antennas)
    y_pred_trilateration.append([x, y])
y_pred_trilateration = np.array(y_pred_trilateration)

# Scale features for classifier
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
static_indices = all_data['test_id'].str.contains(r'^\(\d+,\d+\)\.csv$')
dynamic_indices = all_data['test_id'].str.startswith(('movetest', 'circletest'))
X_train = X[static_indices | dynamic_indices]
y_train = y[static_indices | dynamic_indices]
w_train = sample_weights[static_indices | dynamic_indices]
X_test = X[dynamic_indices]
y_test = y[dynamic_indices]
y_pred_test = y_pred_trilateration[dynamic_indices]

# Train classifier
classifier = RandomForestClassifier(n_estimators=1000, max_depth=5, class_weight='balanced', random_state=42)
classifier.fit(X_train, y_train['quadrant'], sample_weight=w_train)

# Evaluate trilateration
valid_mask = ~np.isnan(y_pred_test).any(axis=1)
mse_x = mean_squared_error(y_test['x_coord'][valid_mask], y_pred_test[valid_mask, 0])
mse_y = mean_squared_error(y_test['y_coord'][valid_mask], y_pred_test[valid_mask, 1])
mae_x = mean_absolute_error(y_test['x_coord'][valid_mask], y_pred_test[valid_mask, 0])
mae_y = mean_absolute_error(y_test['y_coord'][valid_mask], y_pred_test[valid_mask, 1])
avg_mse = (mse_x + mse_y) / 2
y_pred_quad = np.array([get_quadrant(x, y) for x, y in y_pred_test[valid_mask]])
quad_acc = accuracy_score(y_test['quadrant'][valid_mask], y_pred_quad)
print(f"Trilateration Results (Dynamic):")
print(f"  MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Avg MSE: {avg_mse:.2f}")
print(f"  MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")
print(f"  Quadrant Accuracy: {quad_acc:.2%}")

# Evaluate classifier
y_quad_pred = classifier.predict(X_test)
quad_accuracy = accuracy_score(y_test['quadrant'], y_quad_pred)
print(f"RandomForest Classifier Quadrant Accuracy (Dynamic): {quad_accuracy:.2%}")

# Save confusion matrix
cm = confusion_matrix(y_test['quadrant'], y_quad_pred, labels=[1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.xlabel('Predicted Quadrant')
plt.ylabel('Actual Quadrant')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# Feature importances
importances = classifier.feature_importances_
plt.figure(figsize=(12, 6))
plt.bar(feature_columns, importances)
plt.xticks(rotation=90)
plt.ylabel('Importance')
plt.title('Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')
plt.close()

# Save models
import pickle
with open('best_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Plot predictions
def plot_predictions(X, y, test_ids, data_type, filename, y_pred):
    if X.empty or y.empty:
        return
    unique_tests = test_ids.unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tests)))
    color_map = dict(zip(unique_tests, colors))
    plt.figure(figsize=(10, 10))
    for test_id in unique_tests:
        mask = test_ids == test_id
        plt.scatter(y[mask]['x_coord'], y[mask]['y_coord'], c=[color_map[test_id]], label=f'{test_id} Actual', alpha=0.5, s=100)
        plt.scatter(y_pred[mask, 0], y_pred[mask, 1], c=[color_map[test_id]], marker='x', label=f'{test_id} Predicted', alpha=0.5, s=100)
    plt.axvline(x=7.5, color='gray', linestyle='--')
    plt.axhline(y=7.5, color='gray', linestyle='--')
    plt.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Actual vs Predicted ({data_type})')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_predictions(X_test[all_data['test_id'][dynamic_indices].str.startswith('movetest')], 
                 y_test[all_data['test_id'][dynamic_indices].str.startswith('movetest')], 
                 all_data['test_id'][dynamic_indices & all_data['test_id'].str.startswith('movetest')], 
                 'Dynamic Linear', 'movetest2_predictions_trilateration.png', y_pred_test)
plot_predictions(X_test[all_data['test_id'][dynamic_indices].str.startswith('circletest')], 
                 y_test[all_data['test_id'][dynamic_indices].str.startswith('circletest')], 
                 all_data['test_id'][dynamic_indices & all_data['test_id'].str.startswith('circletest')], 
                 'Dynamic Circular', 'circletest1_predictions_trilateration.png', y_pred_test)

# Statistics
print("\nDoppler Frequency Stats:")
print(all_data[[f'doppler_frequency_Ant{i}' for i in range(1, 5)]].describe())
print("\nRSSI Stats:")
print(all_data[[f'rssi_Ant{i}' for i in range(1, 5)]].describe())
print("Plots saved: 'movetest2_predictions_trilateration.png', 'circletest1_predictions_trilateration.png', 'confusion_matrix.png', 'feature_importances.png'")