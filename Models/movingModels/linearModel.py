import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
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
        return 2.0  # Top-left
    elif x <= 7.5 and y <= 7.5:
        return 3  # Bottom-left
    else:  # x > 7.5, y <= 7.5
        return 4  # Bottom-right

# Load metadata
metadata_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv'
metadata = pd.read_csv(metadata_path)
metadata.columns = metadata.columns.str.strip()
print(f"Metadata columns: {list(metadata.columns)}")

# Check required columns
required_cols = ['raw_CSV_filename', 'start_x_true', 'start_y_true', 'end_x_true', 'end_y_true']
if not all(col in metadata.columns for col in required_cols):
    print(f"Error: Missing columns in LineMetadata.csv")
    exit(1)

# Filter for MoveTest2
metadata = metadata[metadata['raw_CSV_filename'].str.startswith('movetest2-')]

# Antenna positions
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Load stationary data
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
stationary_data_list = []
for file in glob.glob(os.path.join(data_dir, '*.csv')):
    filename = os.path.basename(file)
    coords = filename.replace('.csv', '').strip('()').split(',')
    try:
        x_coord, y_coord = float(coords[0]), float(coords[1])
    except ValueError:
        print(f"Skipping {file}: Invalid filename")
        continue
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ['antenna', 'rssi', 'phase_angle']):
        print(f"Skipping {file}: Missing columns")
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
        print(f"CSV {csv_path} not found. Skipping.")
        continue
    df = pd.read_csv(csv_path)
    print(f"Test ID {test_id}: {len(df)} rows")
    print(f"  Columns: {list(df.columns)}")
    unique_antennas = df['antenna'].unique()
    print(f"  Unique antennas: {unique_antennas}")
    antenna_counts = df['antenna'].value_counts()
    print(f"  Antenna counts:\n{antenna_counts}\n")
    # Filter imbalanced tests
    if any(antenna_counts.get(ant, 0) < 15 for ant in [1, 2, 3, 4]):
        print(f"Skipping {test_id}: Insufficient antenna readings (<15).")
        continue
    df['test_id'] = test_id
    # Interpolate per antenna
    df_list = []
    sample_weights = []
    for antenna in unique_antennas:
        df_ant = df[df['antenna'] == antenna].copy()
        n_rows = len(df_ant)
        if n_rows == 0:
            continue
        df_ant['x_coord'] = np.linspace(row['start_x_true'], row['end_x_true'], n_rows)
        df_ant['y_coord'] = np.linspace(row['start_y_true'], row['end_y_true'], n_rows)
        # Weight inversely proportional to antenna count
        weight = 1 / antenna_counts.get(antenna, 1)
        sample_weights.extend([weight] * n_rows)
        df_list.append(df_ant)
    df = pd.concat(df_list, ignore_index=True)
    df['data_type'] = 'dynamic'
    df['sample_weight'] = sample_weights
    dynamic_data_list.append(df)
dynamic_data = pd.concat(dynamic_data_list, ignore_index=True)

if dynamic_data.empty:
    print("No dynamic data loaded.")
    exit(1)

# Aggregate datasets
stationary_pivot = stationary_data.groupby(['test_id', 'x_coord', 'y_coord', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean'
}).unstack()
stationary_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in stationary_pivot.columns]
stationary_pivot = stationary_pivot.reset_index()
print(f"Stationary pivot rows: {len(stationary_pivot)}")

dynamic_pivot = dynamic_data.groupby(['test_id', 'x_coord', 'y_coord', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean',
    'sample_weight': 'mean'
}).unstack()
dynamic_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' if col[0] != 'sample_weight' else f'sample_weight_Ant{int(col[1])}' for col in dynamic_pivot.columns]
dynamic_pivot = dynamic_pivot.reset_index()
print(f"Dynamic pivot rows: {len(dynamic_pivot)}")

# Combine datasets
all_data = pd.concat([stationary_pivot, dynamic_pivot], ignore_index=True)

# Add RSSI and phase differences
for i in range(1, 4):
    for j in range(i + 1, 5):
        all_data[f'rssi_diff_Ant{i}_Ant{j}'] = all_data[f'rssi_Ant{i}'] - all_data[f'rssi_Ant{j}']
        all_data[f'phase_diff_Ant{i}_Ant{j}'] = all_data[f'phase_angle_Ant{i}'] - all_data[f'phase_angle_Ant{j}']

# Add quadrant labels
all_data['quadrant'] = all_data.apply(lambda row: get_quadrant(row['x_coord'], row['y_coord']), axis=1)

# Handle missing values
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

# Define features
feature_columns = [col for col in all_data.columns if col.startswith(('rssi_Ant', 'phase_angle_Ant', 'rssi_diff_Ant', 'phase_diff_Ant'))]
X = all_data[feature_columns]
y = all_data[['x_coord', 'y_coord']]
y_quadrant = all_data['quadrant']

# Normalize sample weights
sample_weights = np.ones(len(all_data))
dynamic_indices = all_data['test_id'].str.startswith('movetest2-')
if dynamic_indices.sum() > 0:
    dynamic_weights = []
    for ant in [1, 2, 3, 4]:
        col = f'sample_weight_Ant{ant}'
        if col in all_data.columns:
            dynamic_weights.append(all_data.loc[dynamic_indices, col].fillna(1.0))
    if dynamic_weights:
        # Average weights across antennas
        dynamic_weights = np.mean(dynamic_weights, axis=0)
        # Normalize per test_id
        dynamic_weights = pd.Series(dynamic_weights, index=all_data[dynamic_indices].index)
        dynamic_weights = dynamic_weights.groupby(all_data[dynamic_indices]['test_id']).transform(lambda x: x / x.sum())
        sample_weights[dynamic_indices] = dynamic_weights.values

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
X_train, X_test, y_train, y_test, y_quad_train, y_quad_test, w_train, w_test = train_test_split(
    X, y, y_quadrant, sample_weights, test_size=0.2, random_state=42
)

# Train coordinate regressor
regressor = MultiOutputRegressor(GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
))
regressor.fit(X_train, y_train, sample_weight=w_train)

# Evaluate regressor
y_pred = regressor.predict(X_test)
mse_x = mean_squared_error(y_test['x_coord'], y_pred[:, 0])
mse_y = mean_squared_error(y_test['y_coord'], y_pred[:, 1])
mae_x = mean_absolute_error(y_test['x_coord'], y_pred[:, 0])
mae_y = mean_absolute_error(y_test['y_coord'], y_pred[:, 1])
avg_mse = (mse_x + mse_y) / 2
print(f"GradientBoosting Test Set MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Average MSE: {avg_mse:.2f}")
print(f"GradientBoosting Test Set MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")

# Train quadrant classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_quad_train, sample_weight=w_train)

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

# Plot feature importances
importances = classifier.feature_importances_
plt.figure(figsize=(10, 6))
plt.bar(feature_columns, importances)
plt.xticks(rotation=90)
plt.ylabel('Feature Importance')
plt.title('Feature Importances for Quadrant Classifier')
plt.tight_layout()
plt.savefig('feature_importances_movetest2.png')
plt.close()

# Save models and scaler
import pickle
with open('best_regressor_movetest2.pkl', 'wb') as file:
    pickle.dump(regressor, file)
with open('best_classifier_movetest2.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('scaler_movetest2.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Models and scaler saved.")

# Plot results (MoveTest2 only)
dynamic_indices = all_data['test_id'].str.startswith('movetest2-')
X_dynamic = X[dynamic_indices]
y_dynamic = y[dynamic_indices]
y_pred_dynamic = regressor.predict(X_dynamic)
test_ids_dynamic = all_data['test_id'][dynamic_indices]

# Color by test_id
unique_tests = test_ids_dynamic.unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_tests)))
color_map = dict(zip(unique_tests, colors))

plt.figure(figsize=(10, 10))
for test_id in unique_tests:
    mask = test_ids_dynamic == test_id
    plt.scatter(y_dynamic[mask]['x_coord'], y_dynamic[mask]['y_coord'], c=[color_map[test_id]], label=f'{test_id} Actual', alpha=0.5, s=100)
    plt.scatter(y_pred_dynamic[mask, 0], y_pred_dynamic[mask, 1], c=[color_map[test_id]], marker='x', label=f'{test_id} Predicted', alpha=0.5, s=100)

# Add quadrant lines
plt.axvline(x=7.5, color='gray', linestyle='--')
plt.axhline(y=7.5, color='gray', linestyle='--')

# Plot antennas
plt.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Actual vs. Predicted RFID Coordinates (MoveTest2)')
plt.xlim(0, 15)
plt.ylim(0, 15)
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('model_predictions_movetest2.png')
plt.close()

print("Plots saved as 'model_predictions_movetest2.png', 'quadrant_confusion_matrix.png', and 'feature_importances_movetest2.png'")