import pandas as pd
import numpy as np
import glob
import os
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix
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

# Load metadata
line_metadata = pd.read_csv('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv')
circle_metadata = pd.read_csv('/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleMetadata.csv')
line_metadata.columns = line_metadata.columns.str.strip()
circle_metadata.columns = circle_metadata.columns.str.strip()

# Filter metadata
line_metadata = line_metadata[line_metadata['raw_CSV_filename'].str.startswith('movetest2-')]
circle_metadata = circle_metadata[circle_metadata['raw_CSV_filename'].str.startswith('circletest1-')]

antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]

# Load stationary data
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
stationary_data_list = []
for file in glob.glob(os.path.join(data_dir, '*.csv')):
    filename = os.path.basename(file)
    try:
        x_coord, y_coord = map(float, filename.replace('.csv', '').strip('()').split(','))
    except ValueError:
        continue
    df = pd.read_csv(file)
    if not all(col in df.columns for col in ['antenna', 'rssi', 'phase_angle', 'doppler_frequency', 'channel_index']):
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
dynamic_linear_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/MoveTest2/'
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
dynamic_circular_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleTest1/'
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
        all_data[col] = all_data[col].clip(lower=-10, upper=10)  # Cap at realistic RFID doppler range

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

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Split data
static_indices = all_data['test_id'].str.contains(r'^\(\d+,\d+\)\.csv$')
dynamic_indices = all_data['test_id'].str.startswith(('movetest2-', 'circletest'))
X_train = X[static_indices | dynamic_indices]
y_train = y[static_indices | dynamic_indices]
w_train = sample_weights[static_indices | dynamic_indices]
X_test = X[dynamic_indices]
y_test = y[dynamic_indices]

# Train models
rf_regressor = MultiOutputRegressor(RandomForestRegressor(n_estimators=500, max_depth=5, random_state=42))
rf_regressor.fit(X_train, y_train[['x_coord', 'y_coord']], sample_weight=w_train)

classifier = RandomForestClassifier(n_estimators=500, max_depth=5, class_weight='balanced', random_state=42)
classifier.fit(X_train, y_train['quadrant'], sample_weight=w_train)

# Evaluate regressor
y_pred = rf_regressor.predict(X_test)
mse_x = mean_squared_error(y_test['x_coord'], y_pred[:, 0])
mse_y = mean_squared_error(y_test['y_coord'], y_pred[:, 1])
mae_x = mean_absolute_error(y_test['x_coord'], y_pred[:, 0])
mae_y = mean_absolute_error(y_test['y_coord'], y_pred[:, 1])
avg_mse = (mse_x + mse_y) / 2
y_pred_quad = [get_quadrant(x, y) for x, y in y_pred]
quad_acc = accuracy_score(y_test['quadrant'], y_pred_quad)
print(f"RandomForest Regressor Results (Dynamic):")
print(f"  MSE (x, y): {mse_x:.2f}, {mse_y:.2f} | Avg MSE: {avg_mse:.2f}")
print(f"  MAE (x, y): {mae_x:.2f}, {mae_y:.2f}")
print(f"  Quadrant Accuracy: {quad_acc:.2%}")

# Evaluate classifier
y_quad_pred = classifier.predict(X_test)
quad_accuracy = accuracy_score(y_test['quadrant'], y_quad_pred)
print(f"RandomForest Classifier Quadrant Accuracy (Dynamic): {quad_accuracy:.2%}")

# Save confusion matrix
cm = confusion_matrix(y_test['quadrant'], y_quad_pred, labels=[1, 2,3,4])
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
with open('best_rf_regressor.pkl', 'wb') as file:
    pickle.dump(rf_regressor, file)
with open('best_classifier.pkl', 'wb') as file:
    pickle.dump(classifier, file)
with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

# Plot predictions
def plot_predictions(X, y, test_ids, data_type, filename, regressor):
    if X.empty or y.empty:
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
    plt.title(f'Actual vs Predicted ({data_type})')
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

plot_predictions(X[dynamic_indices & all_data['test_id'].str.startswith('movetest2-')], 
                 y[dynamic_indices & all_data['test_id'].str.startswith('movetest2-')], 
                 all_data['test_id'][dynamic_indices & all_data['test_id'].str.startswith('movetest2-')], 
                 'Dynamic Linear', 'movetest2_predictions_rf.png', rf_regressor)
plot_predictions(X[dynamic_indices & all_data['test_id'].str.startswith('circletest')], 
                 y[dynamic_indices & all_data['test_id'].str.startswith('circletest')], 
                 all_data['test_id'][dynamic_indices & all_data['test_id'].str.startswith('circletest')], 
                 'Dynamic Circular', 'circletest1_predictions_rf.png', rf_regressor)

# Statistics
print("\nDoppler Frequency Stats:")
print(all_data[[f'doppler_frequency_Ant{i}' for i in range(1, 5)]].describe())
print("\nRSSI Stats:")
print(all_data[[f'rssi_Ant{i}' for i in range(1, 5)]].describe())
print("Plots saved: 'movetest2_predictions_rf.png', 'circletest1_predictions_rf.png', 'confusion_matrix.png', 'feature_importances.png'")