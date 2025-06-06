import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Function to assign quadrant
def get_quadrant(x, y):
    if x > 7.5 and y > 7.5:
        return 1  # Top-right
    elif x <= 7.5 and y > 7.5:
        return 2  # Top-left
    elif x <= 7.5 and y <= 7.5:
        return 3  # Bottom-left
    else:  # x > 7.5, y <= 7.5
        return 4  # Bottom-right

# Load metadata for dynamic data
metadata_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv'
metadata = pd.read_csv(metadata_path)

# Strip whitespace from column names
metadata.columns = metadata.columns.str.strip()
print(f"Metadata columns after stripping whitespace: {list(metadata.columns)}")

# Check required columns
required_cols = ['raw_CSV_filename', 'start_x_true', 'start_y_true', 'end_x_true', 'end_y_true']
missing_cols = [col for col in required_cols if col not in metadata.columns]
if missing_cols:
    print(f"Error: Missing required columns in LineMetadata.csv: {missing_cols}")
    print(f"Available columns: {list(metadata.columns)}")
    exit(1)

# Filter for MoveTest2
metadata = metadata[metadata['raw_CSV_filename'].str.startswith('movetest2-')]

# Antenna positions
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Load stationary data
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
all_files = glob.glob(os.path.join(data_dir, '*.csv'))
if not all_files:
    print(f"No CSV files found in {data_dir}. Please verify the directory path.")
    exit(1)

stationary_data_list = []
for file in all_files:
    filename = os.path.basename(file)
    coords = filename.replace('.csv', '').strip('()').split(',')
    try:
        x_coord, y_coord = float(coords[0]), float(coords[1])
    except ValueError:
        print(f"Skipping file {file}: Invalid coordinate format in filename")
        continue
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ['antenna', 'rssi', 'phase_angle']):
        print(f"Skipping file {file}: Missing required columns")
        continue
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    df['test_id'] = filename
    stationary_data_list.append(df)
stationary_data = pd.concat(stationary_data_list, ignore_index=True)
stationary_data['data_type'] = 'stationary'

# Load dynamic data
dynamic_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/MoveTest2/'
dynamic_data_list = []
for _, row in metadata.iterrows():
    test_id = row['raw_CSV_filename']
    csv_path = os.path.join(dynamic_dir, f'{test_id}.csv')
    if not os.path.exists(csv_path):
        print(f"CSV file {csv_path} not found. Skipping {test_id}.")
        continue
    df = pd.read_csv(csv_path)
    print(f"Test ID {test_id}: File found with {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    if not all(col in df.columns for col in ['antenna', 'rssi', 'phase_angle']):
        print(f"CSV file {csv_path} missing required columns. Skipping {test_id}.")
        continue
    unique_antennas = df['antenna'].unique()
    print(f"  Unique antennas: {unique_antennas}")
    df['test_id'] = test_id
    # Interpolate x, y coordinates per antenna
    df_list = []
    for antenna in unique_antennas:
        df_ant = df[df['antenna'] == antenna].copy()
        n_rows = len(df_ant)
        if n_rows == 0:
            continue
        df_ant['x_coord'] = np.linspace(row['start_x_true'], row['end_x_true'], n_rows)
        df_ant['y_coord'] = np.linspace(row['start_y_true'], row['end_y_true'], n_rows)
        df_list.append(df_ant)
    df = pd.concat(df_list, ignore_index=True)
    df['data_type'] = 'dynamic'
    dynamic_data_list.append(df)
dynamic_data = pd.concat(dynamic_data_list, ignore_index=True) if dynamic_data_list else pd.DataFrame()

if dynamic_data.empty:
    print("No dynamic data loaded. Please verify CSV files in", dynamic_dir)
    exit(1)

# Aggregate datasets
stationary_pivot = stationary_data.groupby(['test_id', 'x_coord', 'y_coord', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean'
}).unstack()
stationary_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in stationary_pivot.columns]
stationary_pivot = stationary_pivot.reset_index()

dynamic_pivot = dynamic_data.groupby(['test_id', 'x_coord', 'y_coord', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean'
}).unstack()
dynamic_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in dynamic_pivot.columns]
dynamic_pivot = dynamic_pivot.reset_index()

# Combine datasets
all_data = pd.concat([stationary_pivot, dynamic_pivot], ignore_index=True)

# Add RSSI differences as features
for i in range(1, 4):
    for j in range(i + 1, 5):
        all_data[f'rssi_diff_Ant{i}_Ant{j}'] = all_data[f'rssi_Ant{i}'] - all_data[f'rssi_Ant{j}']

# Add quadrant labels
all_data['quadrant'] = all_data.apply(lambda row: get_quadrant(row['x_coord'], row['y_coord']), axis=1)

# Handle missing values
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

# Define features
feature_columns = [col for col in all_data.columns if col.startswith(('rssi_Ant', 'phase_angle_Ant', 'rssi_diff_Ant'))]
X = all_data[feature_columns]
y = all_data[['x_coord', 'y_coord']]
y_quadrant = all_data['quadrant']

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test, y_quad_train, y_quad_test = train_test_split(
    X, y, y_quadrant, test_size=0.2, random_state=42
)

# Train coordinate regressor
regressor = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=100, max_depth=10, random_state=42
))
regressor.fit(X_train, y_train)

# Evaluate regressor
y_pred = regressor.predict(X_test)
mse_x = mean_squared_error(y_test['x_coord'], y_pred[:, 0])
mse_y = mean_squared_error(y_test['y_coord'], y_pred[:, 1])
mae_x = mean_absolute_error(y_test['x_coord'], y_pred[:, 0])
mae_y = mean_absolute_error(y_test['y_coord'], y_pred[:, 1])
avg_mse = (mse_x + mse_y) / 2

print(f"RandomForest Test Set MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Average MSE: {avg_mse:.2f}")
print(f"RandomForest Test Set MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")

# Train quadrant classifier
classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
classifier.fit(X_train, y_quad_train)

# Evaluate classifier
y_quad_pred = classifier.predict(X_test)
quad_accuracy = accuracy_score(y_quad_test, y_quad_pred)
print(f"Quadrant Prediction Accuracy: {quad_accuracy:.2%}")

# Save confusion matrix
cm = confusion_matrix(y_quad_test, y_quad_pred, labels=[1, 2, 3, 4])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[1, 2, 3, 4], yticklabels=[1, 2, 3, 4])
plt.xlabel('Predicted Quadrant')
plt.ylabel('Actual Quadrant')
plt.title('Quadrant Prediction Confusion Matrix')
plt.savefig('quadrant_confusion_matrix.png')
plt.close()

# Save models and scaler
import pickle
with open('best_regressor_movetest2.pkl', 'wb') as file:
    pickle.dump(regressor, file)
with open('best_classifier_movetest2.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('scaler_movetest2.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Models and scaler saved as 'best_regressor_movetest2.pkl', 'best_classifier_movetest2.pkl', and 'scaler_movetest2.pkl'")

# Plot results (MoveTest2 only)
dynamic_indices = all_data['test_id'].str.startswith('movetest2-')
X_dynamic = X[dynamic_indices]
y_dynamic = y[dynamic_indices]
y_pred_dynamic = regressor.predict(X_dynamic)

plt.figure(figsize=(10, 10))
plt.scatter(y_dynamic['x_coord'], y_dynamic['y_coord'], color='blue', label='Actual', alpha=0.5, s=100)
plt.scatter(y_pred_dynamic[:, 0], y_pred_dynamic[:, 1], color='red', label='Predicted', alpha=0.5, s=100)

# Add dotted lines
for actual, pred in zip(y_dynamic.values, y_pred_dynamic):
    plt.plot([actual[0], pred[0]], [actual[1], pred[1]], color='black', linestyle=':', linewidth=0.5)

# Label 10 random points
np.random.seed(42)
random_indices = np.random.choice(len(y_dynamic), size=min(10, len(y_dynamic)), replace=False)
for idx in random_indices:
    actual = y_dynamic.values[idx]
    pred = y_pred_dynamic[idx]
    plt.annotate(f'A: ({actual[0]:.1f}, {actual[1]:.1f})', 
                 (actual[0], actual[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue')
    plt.annotate(f'P: ({pred[0]:.1f}, {pred[1]:.1f})', 
                 (pred[0], pred[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=6, color='red')

# Add quadrant lines
plt.axvline(x=7.5, color='gray', linestyle='--', linewidth=1)
plt.axhline(y=7.5, color='gray', linestyle='--', linewidth=1)

# Plot antennas
plt.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Actual vs. Predicted RFID Tag Coordinates (MoveTest2)')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.grid(True)
plt.legend()
plt.savefig('model_predictions_movetest2.png')
plt.close()

print("Scatter plot saved as 'model_predictions_movetest2.png'")