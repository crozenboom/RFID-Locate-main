import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
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

# Load metadata
line_metadata_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv'
circle_metadata_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleMetadata.csv'
line_metadata = pd.read_csv(line_metadata_path)
circle_metadata = pd.read_csv(circle_metadata_path)
line_metadata.columns = line_metadata.columns.str.strip()
circle_metadata.columns = circle_metadata.columns.str.strip()
print(f"Line metadata columns: {list(line_metadata.columns)}")
print(f"Circle metadata columns: {list(circle_metadata.columns)}")

# Check required columns
line_required_cols = ['raw_CSV_filename', 'start_x_true', 'start_y_true', 'end_x_true', 'end_y_true']
circle_required_cols = ['raw_CSV_filename', 'center_x_true', 'center_y_true', 'radius_true', 'direction']
if not all(col in line_metadata.columns for col in line_required_cols):
    print(f"Error: Missing columns in LineMetadata.csv")
    exit(1)
if not all(col in circle_metadata.columns for col in circle_required_cols):
    print(f"Error: Missing columns in CircleMetadata.csv")
    exit(1)

# Filter for MoveTest2
line_metadata = line_metadata[line_metadata['raw_CSV_filename'].str.startswith('movetest2-')]

# Antenna positions
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Load stationary data (Test8)
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
stationary_data_list = []
for file in glob.glob(os.path.join(data_dir, '*.csv')):
    filename = os.path.basename(file)
    test_id = filename  # Keep full filename
    try:
        x_coord, y_coord = map(float, filename.replace('.csv', '').strip('()').split(','))
    except ValueError:
        print(f"Skipping {file}: Invalid filename")
        continue
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ['antenna', 'rssi', 'phase_angle']):
        print(f"Skipping {file}: Missing columns")
        continue
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    df['test_id'] = test_id
    df['data_type'] = 'stationary'
    df['quadrant'] = get_quadrant(x_coord, y_coord)  # Add quadrant at data loading
    stationary_data_list.append(df)
stationary_data = pd.concat(stationary_data_list, ignore_index=True)
print(f"Stationary data rows: {len(stationary_data)}")
print("Stationary test IDs:", stationary_data['test_id'].unique())

# Load dynamic linear data (MoveTest2)
dynamic_linear_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/MoveTest2/'
dynamic_linear_data_list = []
for _, row in line_metadata.iterrows():
    test_id = row['raw_CSV_filename']
    csv_path = os.path.join(dynamic_linear_dir, f'{test_id}.csv')
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
        df_ant['quadrant'] = df_ant.apply(lambda row: get_quadrant(row['x_coord'], row['y_coord']), axis=1)
        # Weight inversely proportional to antenna count
        weight = 1 / antenna_counts.get(antenna, 1)
        sample_weights.extend([weight] * n_rows)
        df_list.append(df_ant)
    df = pd.concat(df_list, ignore_index=True)
    df['data_type'] = 'dynamic_linear'
    df['sample_weight'] = sample_weights
    dynamic_linear_data_list.append(df)
dynamic_linear_data = pd.concat(dynamic_linear_data_list, ignore_index=True)
print(f"Dynamic linear data rows: {len(dynamic_linear_data)}")

# Load dynamic circular data (MoveTest3)
dynamic_circular_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleTest1/'
dynamic_circular_data_list = []
for _, row in circle_metadata.iterrows():
    test_id = row['raw_CSV_filename']
    csv_path = os.path.join(dynamic_circular_dir, f'{test_id}.csv')
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
        # Generate circular path
        theta = np.linspace(0, 2 * np.pi, n_rows)
        if row['direction'] == 'counterclockwise':
            theta = theta[::-1]  # Reverse for counterclockwise
        df_ant['x_coord'] = row['center_x_true'] + row['radius_true'] * np.cos(theta)
        df_ant['y_coord'] = row['center_y_true'] + row['radius_true'] * np.sin(theta)
        df_ant['quadrant'] = df_ant.apply(lambda row: get_quadrant(row['x_coord'], row['y_coord']), axis=1)
        # Weight inversely proportional to antenna count
        weight = 1 / antenna_counts.get(antenna, 1)
        sample_weights.extend([weight] * n_rows)
        df_list.append(df_ant)
    df = pd.concat(df_list, ignore_index=True)
    df['data_type'] = 'dynamic_circular'
    df['sample_weight'] = sample_weights
    dynamic_circular_data_list.append(df)
dynamic_circular_data = pd.concat(dynamic_circular_data_list, ignore_index=True)
print(f"Dynamic circular data rows: {len(dynamic_circular_data)}")

if dynamic_linear_data.empty and dynamic_circular_data.empty:
    print("No dynamic data loaded.")
    exit(1)

# Aggregate datasets
stationary_pivot = stationary_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle'],
    aggfunc='mean'
)
stationary_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in stationary_pivot.columns]
stationary_pivot = stationary_pivot.reset_index()
print(f"Stationary pivot rows: {len(stationary_pivot)}")
print("Stationary pivot test IDs:", stationary_pivot['test_id'].unique())
print("Stationary pivot columns:", stationary_pivot.columns)

dynamic_linear_pivot = dynamic_linear_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle', 'sample_weight'],
    aggfunc='mean'
)
dynamic_linear_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in dynamic_linear_pivot.columns]
dynamic_linear_pivot = dynamic_linear_pivot.reset_index()
print(f"Dynamic linear pivot rows: {len(dynamic_linear_pivot)}")
print("Dynamic linear pivot columns:", dynamic_linear_pivot.columns)

dynamic_circular_pivot = dynamic_circular_data.pivot_table(
    index=['test_id', 'x_coord', 'y_coord', 'quadrant'],
    columns='antenna',
    values=['rssi', 'phase_angle', 'sample_weight'],
    aggfunc='mean'
)
dynamic_circular_pivot.columns = [f'{col[0]}_Ant{int(col[1])}' for col in dynamic_circular_pivot.columns]
dynamic_circular_pivot = dynamic_circular_pivot.reset_index()
print(f"Dynamic circular pivot rows: {len(dynamic_circular_pivot)}")
print("Dynamic circular pivot columns:", dynamic_circular_pivot.columns)

# Combine datasets
all_data = pd.concat([stationary_pivot, dynamic_linear_pivot, dynamic_circular_pivot], ignore_index=True)
print("all_data shape:", all_data.shape)
print("all_data columns:", all_data.columns)
print("all_data test IDs:", all_data['test_id'].unique())

# Fill NaNs in RSSI and phase columns before calculating differences
rssi_cols = [f'rssi_Ant{i}' for i in range(1, 5)]
phase_cols = [f'phase_angle_Ant{i}' for i in range(1, 5)]
for col in rssi_cols + phase_cols:
    if col in all_data.columns:
        all_data[col] = all_data[col].fillna(all_data[col].mean())

# Add RSSI and phase differences
for i in range(1, 4):
    for j in range(i + 1, 5):
        all_data[f'rssi_diff_Ant{i}_Ant{j}'] = all_data[f'rssi_Ant{i}'] - all_data[f'rssi_Ant{j}']
        all_data[f'phase_diff_Ant{i}_Ant{j}'] = all_data[f'phase_angle_Ant{i}'] - all_data[f'phase_angle_Ant{j}']

# Ensure quadrant is included
all_data['quadrant'] = all_data.apply(lambda row: get_quadrant(row['x_coord'], row['y_coord']), axis=1)

# Handle missing values in all numeric columns
numeric_cols = all_data.select_dtypes(include=[np.number]).columns
all_data[numeric_cols] = all_data[numeric_cols].fillna(all_data[numeric_cols].mean())

# Check for NaNs in features and targets
feature_columns = [col for col in all_data.columns if col.startswith(('rssi_Ant', 'phase_angle_Ant', 'rssi_diff_Ant', 'phase_diff_Ant'))]
X = all_data[feature_columns]
y = all_data[['x_coord', 'y_coord', 'quadrant']]
if X.isna().any().any():
    print("NaNs found in X:")
    print(X.isna().sum())
if y.isna().any().any():
    print("NaNs found in y:")
    print(y.isna().sum())
y_coords = y[['x_coord', 'y_coord']]
y_quadrant = y['quadrant']

# Normalize sample weights
sample_weights = np.ones(len(all_data))
for data_type in ['dynamic_linear', 'dynamic_circular']:
    indices = all_data['test_id'].str.startswith('movetest2-' if data_type == 'dynamic_linear' else 'circletest')
    if indices.sum() > 0:
        for ant in [1, 2, 3, 4]:
            col = f'sample_weight_Ant{ant}'
            if col in all_data.columns:
                weights = all_data.loc[indices, col].fillna(1.0)
                weights = weights.groupby(all_data[indices]['test_id']).transform(lambda x: x / x.sum())
                sample_weights[indices] = weights.values
                break  # Use first available antenna weight

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Check for NaNs after scaling
if X.isna().any().any():
    print("NaNs found in X after scaling:")
    print(X.isna().sum())

# Split data
X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, sample_weights, test_size=0.2, random_state=42
)

# Train GradientBoosting regressor
gb_regressor = MultiOutputRegressor(GradientBoostingRegressor(
    n_estimators=500, learning_rate=0.05, max_depth=3, random_state=42
))
gb_regressor.fit(X_train, y_train[['x_coord', 'y_coord']], sample_weight=w_train)

# Train RandomForest regressor
rf_regressor = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=500, max_depth=3, random_state=42
))
rf_regressor.fit(X_train, y_train[['x_coord', 'y_coord']], sample_weight=w_train)

# Train quadrant classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train['quadrant'], sample_weight=w_train)

# Evaluate regressors
for name, regressor in [('GradientBoosting', gb_regressor), ('RandomForest', rf_regressor)]:
    y_pred = regressor.predict(X_test)
    mse_x = mean_squared_error(y_test['x_coord'], y_pred[:, 0])
    mse_y = mean_squared_error(y_test['y_coord'], y_pred[:, 1])
    mae_x = mean_absolute_error(y_test['x_coord'], y_pred[:, 0])
    mae_y = mean_absolute_error(y_test['y_coord'], y_pred[:, 1])
    avg_mse = (mse_x + mse_y) / 2
    # Compute quadrant accuracy from predicted coordinates
    y_pred_quad = [get_quadrant(x, y) for x, y in y_pred]
    quad_acc = accuracy_score(y_test['quadrant'], y_pred_quad)
    print(f"\n{name} Test Set Results:")
    print(f"  MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Average MSE: {avg_mse:.2f}")
    print(f"  MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")
    print(f"  Quadrant Accuracy from Coordinates: {quad_acc:.2%}")

# Evaluate classifier
y_quad_pred = classifier.predict(X_test)
quad_accuracy = accuracy_score(y_test['quadrant'], y_quad_pred)
print(f"\nRandomForest Classifier Quadrant Prediction Accuracy: {quad_accuracy:.2%}")

# Save confusion matrix
cm = confusion_matrix(y_test['quadrant'], y_quad_pred, labels=[1, 2, 3, 4])
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
plt.savefig('feature_importances_combined.png')
plt.close()

# Save models and scaler
import pickle
with open('best_gb_regressor_combined.pkl', 'wb') as file:
    pickle.dump(gb_regressor, file)
with open('best_rf_regressor_combined.pkl', 'wb') as file:
    pickle.dump(rf_regressor, file)
with open('best_classifier_combined.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('scaler_combined.pkl', 'wb') as file:
    pickle.dump(scaler, file)
print("Models and scaler saved.")

# Function to plot predictions
def plot_predictions(X, y, test_ids, data_type, filename, regressor, model_name):
    if X.empty or y.empty:
        print(f"Skipping plot for {data_type} ({model_name}): No data available")
        return
    y_pred = regressor.predict(X)
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
    plt.title(f'Actual vs. Predicted RFID Coordinates ({data_type}, {model_name})')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Define indices with corrected regex
stationary_indices = all_data['test_id'].str.contains(r'^\(\d+,\d+\)\.csv$|^\(\d+,\d+\.csv$') | (all_data.get('data_type', '') == 'stationary')
dynamic_linear_indices = all_data['test_id'].str.startswith('movetest2-')
dynamic_circular_indices = all_data['test_id'].str.startswith('circletest')
print(f"Stationary indices count: {stationary_indices.sum()}")
print(f"Dynamic linear indices count: {dynamic_linear_indices.sum()}")
print(f"Dynamic circular indices count: {dynamic_circular_indices.sum()}")

# Plot for each dataset and model
for name, regressor in [('GradientBoosting', gb_regressor), ('RandomForest', rf_regressor)]:
    plot_predictions(X[stationary_indices], y[stationary_indices], all_data['test_id'][stationary_indices], 
                     'Stationary (Test8)', f'model_predictions_test8_{name}.png', regressor, name)
    plot_predictions(X[dynamic_linear_indices], y[dynamic_linear_indices], all_data['test_id'][dynamic_linear_indices], 
                     'Dynamic Linear (MoveTest2)', f'model_predictions_movetest2_{name}.png', regressor, name)
    plot_predictions(X[dynamic_circular_indices], y[dynamic_circular_indices], all_data['test_id'][dynamic_circular_indices], 
                     'Dynamic Circular (MoveTest3)', f'model_predictions_movetest3_{name}.png', regressor, name)

# Compare results by dataset
for data_type, indices in [
    ('Stationary (Test8)', stationary_indices),
    ('Dynamic Linear (MoveTest2)', dynamic_linear_indices),
    ('Dynamic Circular (MoveTest3)', dynamic_circular_indices)
]:
    if indices.sum() > 0:
        for name, regressor in [('GradientBoosting', gb_regressor), ('RandomForest', rf_regressor)]:
            y_pred = regressor.predict(X[indices])
            mse_x = mean_squared_error(y[indices]['x_coord'], y_pred[:, 0])
            mse_y = mean_squared_error(y[indices]['y_coord'], y_pred[:, 1])
            mae_x = mean_absolute_error(y[indices]['x_coord'], y_pred[:, 0])
            mae_y = mean_absolute_error(y[indices]['y_coord'], y_pred[:, 1])
            avg_mse = (mse_x + mse_y) / 2
            y_pred_quad = [get_quadrant(x, y) for x, y in y_pred]
            quad_acc = accuracy_score(y[indices]['quadrant'], y_pred_quad)
            print(f"\n{data_type} Results ({name}):")
            print(f"  MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Average MSE: {avg_mse:.2f}")
            print(f"  MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")
            print(f"  Quadrant Accuracy from Coordinates: {quad_acc:.2%}")
        # Classifier quadrant accuracy
        y_quad_pred = classifier.predict(X[indices])
        quad_acc = accuracy_score(y[indices]['quadrant'], y_quad_pred)
        print(f"  RandomForest Classifier Quadrant Accuracy: {quad_acc:.2%}")

print("Plots saved as 'model_predictions_test8_*.png', 'model_predictions_movetest2_*.png', 'model_predictions_movetest3_*.png', 'quadrant_confusion_matrix.png', and 'feature_importances_combined.png'")