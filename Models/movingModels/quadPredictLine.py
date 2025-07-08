import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import re
from time import time

# Define quadrant mapping function
def coordinates_to_quadrant(x, y):
    """Map x, y coordinates to quadrants in a 15x15 ft grid."""
    if x < 7.5 and y >= 7.5:
        return 'Q1'  # Top-left
    elif x >= 7.5 and y >= 7.5:
        return 'Q2'  # Top-right
    elif x < 7.5 and y < 7.5:
        return 'Q3'  # Bottom-left
    else:
        return 'Q4'  # Bottom-right

# Parse coordinates from Test8 filenames
def parse_test8_coordinates(filename):
    """Extract x, y coordinates from filenames like (12,3).csv."""
    match = re.match(r'\((\d+),(\d+)\)\.csv', filename)
    if match:
        x, y = float(match.group(1)), float(match.group(2))
        if 0 <= x <= 15 and 0 <= y <= 15:
            return x, y
        else:
            print(f"Warning: Coordinates ({x}, {y}) in {filename} are outside 15x15 grid.")
            return None, None
    else:
        print(f"Warning: Invalid filename format for {filename}, expecting (x,y).csv.")
        return None, None

# Interpolate linear path coordinates for dynamic data
def interpolate_line_path(timestamps, start_x, start_y, end_x, end_y, num_points=500):
    """Interpolate x, y coordinates for linear paths based on timestamps."""
    t = np.linspace(0, 1, num_points)
    timestamps = np.array(timestamps)
    t_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    x = start_x + (end_x - start_x) * t_normalized
    y = start_y + (end_y - start_y) * t_normalized
    return x, y

# Load and preprocess data
def load_and_preprocess_data(file_or_dir, data_dir, is_static=False):
    """Load and preprocess RFID data from CSV files, handling dynamic or static data."""
    data_frames = []
    
    if is_static:
        # Handle static data (Test8 files like (12,3).csv)
        for fname in os.listdir(data_dir):
            if not fname.endswith('.csv'):
                continue
            file_path = os.path.join(data_dir, fname)
            print(f"Processing static file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                # Log and clean doppler_frequency
                if 'doppler_frequency' in df.columns:
                    print(f"Doppler frequency range in {fname}: {df['doppler_frequency'].min()} to {df['doppler_frequency'].max()}")
                    print(f"Setting all doppler_frequency to 0 in static file {fname}.")
                    df['doppler_frequency'] = 0  # Static tags should have zero Doppler
                # Filter rssi outliers
                if 'rssi' in df.columns:
                    print(f"RSSI range in {fname}: {df['rssi'].min()} to {df['rssi'].max()}")
                    outliers_rssi = (df['rssi'] < -100) | (df['rssi'] > -40)
                    print(f"Outliers in rssi (<-100 or >-40 dBm) in {fname}: {outliers_rssi.sum()}")
                    df.loc[outliers_rssi, 'rssi'] = np.nan
                if 'phase_angle' in df.columns:
                    print(f"Phase angle range in {fname}: {df['phase_angle'].min()} to {df['phase_angle'].max()}")
                # Parse coordinates from filename
                x_true, y_true = parse_test8_coordinates(fname)
                if x_true is None or y_true is None:
                    print(f"Skipping {file_path} due to invalid coordinates.")
                    continue
                # Pivot data to get one row per timestamp
                df['timestamp'] = df['timestamp'].round(3)
                pivoted = df.pivot_table(
                    index='timestamp',
                    columns='antenna',
                    values=['rssi', 'phase_angle', 'doppler_frequency'],
                    aggfunc='mean'
                )
                pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
                pivoted = pivoted.reset_index()
                # Fill NaNs with mean for this file
                for col in pivoted.columns:
                    if col.startswith(('rssi_', 'phase_angle_', 'doppler_frequency_')):
                        pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
                # Check for remaining NaNs
                nan_counts = pivoted.isna().sum()
                print(f"NaN counts in {fname} after imputation: {nan_counts[nan_counts > 0].to_dict()}")
                if pivoted.isna().any().any():
                    print(f"Warning: Remaining NaNs in {fname} after imputation. Dropping rows.")
                    pivoted = pivoted.dropna()
                pivoted['x_true'] = x_true
                pivoted['y_true'] = y_true
                pivoted['quadrant'] = pivoted.apply(lambda r: coordinates_to_quadrant(r['x_true'], r['y_true']), axis=1)
                pivoted['filename'] = fname
                data_frames.append(pivoted)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    else:
        # Handle dynamic data (MoveTest2 with metadata)
        try:
            print(f"Looking for metadata file at: {os.path.abspath(file_or_dir)}")
            metadata = pd.read_csv(file_or_dir)
            metadata.columns = metadata.columns.str.strip()
            print(f"Metadata columns: {list(metadata.columns)}")
            required_dynamic_cols = ['raw_CSV_filename', 'path_#', 'start_x_true', 'start_y_true', 'end_x_true', 'end_y_true']
            if not all(col in metadata.columns for col in required_dynamic_cols):
                print(f"Error: Missing required columns in metadata. Required: {required_dynamic_cols}, Found: {list(metadata.columns)}")
                return None
            # Filter for MoveTest2 (path_# 3 and 4)
            metadata = metadata[metadata['raw_CSV_filename'].str.startswith('movetest2-')]
        except FileNotFoundError:
            print(f"Error: Metadata file '{file_or_dir}' not found.")
            return None
        
        print(f"Available files in {data_dir}:")
        print([f for f in os.listdir(data_dir) if f.endswith('.csv')])
        
        for _, row in metadata.iterrows():
            fname = row['raw_CSV_filename'] + '.csv'
            file_path = os.path.join(data_dir, fname)
            if not os.path.exists(file_path):
                print(f"Warning: File not found for {row['raw_CSV_filename']} at {file_path}")
                continue
            
            print(f"Processing file: {file_path}")
            try:
                df = pd.read_csv(file_path)
                # Filter doppler_frequency outliers
                if 'doppler_frequency' in df.columns:
                    print(f"Doppler frequency range in {fname}: {df['doppler_frequency'].min()} to {df['doppler_frequency'].max()}")
                    outliers_doppler = df['doppler_frequency'].abs() > 1000
                    print(f"Outliers in doppler_frequency (>1000 Hz) in {fname}: {outliers_doppler.sum()}")
                    df.loc[outliers_doppler, 'doppler_frequency'] = np.nan
                # Filter rssi outliers
                if 'rssi' in df.columns:
                    print(f"RSSI range in {fname}: {df['rssi'].min()} to {df['rssi'].max()}")
                    outliers_rssi = (df['rssi'] < -100) | (df['rssi'] > -40)
                    print(f"Outliers in rssi (<-100 or >-40 dBm) in {fname}: {outliers_rssi.sum()}")
                    df.loc[outliers_rssi, 'rssi'] = np.nan
                if 'phase_angle' in df.columns:
                    print(f"Phase angle range in {fname}: {df['phase_angle'].min()} to {df['phase_angle'].max()}")
                df['timestamp'] = df['timestamp'].round(3)
                pivoted = df.pivot_table(
                    index='timestamp',
                    columns='antenna',
                    values=['rssi', 'phase_angle', 'doppler_frequency'],
                    aggfunc='mean'
                )
                pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
                pivoted = pivoted.reset_index()
                # Fill NaNs with mean for this file
                for col in pivoted.columns:
                    if col.startswith(('rssi_', 'phase_angle_', 'doppler_frequency_')):
                        pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
                # Check for remaining NaNs
                nan_counts = pivoted.isna().sum()
                print(f"NaN counts in {fname} after imputation: {nan_counts[nan_counts > 0].to_dict()}")
                if pivoted.isna().any().any():
                    print(f"Warning: Remaining NaNs in {fname} after imputation. Dropping rows.")
                    pivoted = pivoted.dropna()
                x, y = interpolate_line_path(pivoted['timestamp'], row['start_x_true'], row['start_y_true'], row['end_x_true'], row['end_y_true'], num_points=500)
                pivoted['x_true'] = x
                pivoted['y_true'] = y
                pivoted['quadrant'] = pivoted.apply(lambda r: coordinates_to_quadrant(r['x_true'], r['y_true']), axis=1)
                pivoted['filename'] = row['raw_CSV_filename']
                data_frames.append(pivoted)
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
                continue
    
    if not data_frames:
        print("No valid data files loaded.")
        return None
    
    return pd.concat(data_frames, ignore_index=True)

# Generate synthetic stationary data
def generate_synthetic_static_data(n_points_per_quadrant=100):
    """Generate synthetic static data for training if real data is unavailable."""
    np.random.seed(42)
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    x_ranges = {'Q1': (0, 7.5), 'Q2': (7.5, 15), 'Q3': (0, 7.5), 'Q4': (7.5, 15)}
    y_ranges = {'Q1': (7.5, 15), 'Q2': (7.5, 15), 'Q3': (0, 7.5), 'Q4': (0, 7.5)}
    data = []
    
    for q in quadrants:
        x = np.random.uniform(x_ranges[q][0], x_ranges[q][1], n_points_per_quadrant)
        y = np.random.uniform(y_ranges[q][0], y_ranges[q][1], n_points_per_quadrant)
        for i in range(n_points_per_quadrant):
            row = {
                'timestamp': i,
                'x_true': x[i],
                'y_true': y[i],
                'quadrant': q
            }
            for ant in range(1, 5):
                row[f'rssi_{ant}'] = np.random.uniform(-80, -40)
                row[f'phase_angle_{ant}'] = np.random.uniform(0, 4096)
                row[f'doppler_frequency_{ant}'] = 0  # Static data
            data.append(row)
    
    return pd.DataFrame(data)

# Main processing
def main():
    """Main function to load data, train model, and generate outputs."""
    start_time = time()
    # File paths
    metadata_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv'
    data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/MoveTest2'
    static_data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8'
    
    # Load dynamic data
    print("Loading dynamic data...")
    dynamic_data = load_and_preprocess_data(metadata_file, data_dir, is_static=False)
    if dynamic_data is None:
        print("Exiting due to failure in loading dynamic data.")
        return
    
    # Load static data
    print("Loading static data...")
    if os.path.exists(static_data_dir):
        static_data = load_and_preprocess_data(None, static_data_dir, is_static=True)
        if static_data is None:
            print("No valid static data files found, using synthetic data.")
            static_data = generate_synthetic_static_data()
    else:
        print("Warning: Static data directory not found, generating synthetic data.")
        static_data = generate_synthetic_static_data()
    
    # Combine datasets
    data = pd.concat([static_data, dynamic_data], ignore_index=True)
    
    # Add pairwise feature differences
    data['rssi_1_2_diff'] = data['rssi_1'] - data['rssi_2']
    data['rssi_3_4_diff'] = data['rssi_3'] - data['rssi_4']
    data['phase_angle_1_2_diff'] = data['phase_angle_1'] - data['phase_angle_2']
    data['phase_angle_3_4_diff'] = data['phase_angle_3'] - data['phase_angle_4']
    
    # Log sample counts per quadrant
    print("Sample counts per quadrant:")
    print(data['quadrant'].value_counts())
    
    # Features and target
    features = [f'rssi_{i}' for i in range(1, 5)] + [f'phase_angle_{i}' for i in range(1, 5)] + [f'doppler_frequency_{i}' for i in range(1, 5)] + ['rssi_1_2_diff', 'rssi_3_4_diff', 'phase_angle_1_2_diff', 'phase_angle_3_4_diff']
    target = 'quadrant'
    
    # Verify no NaNs before splitting
    nan_counts = data[features].isna().sum()
    print(f"NaN counts in features before splitting: {nan_counts[nan_counts > 0].to_dict()}")
    if data[features].isna().any().any():
        print("Warning: Dropping rows with remaining NaNs before splitting.")
        data = data.dropna(subset=features)
    
    # Split data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    print("Applying SMOTE...")
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Sample counts after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with progress logging
    print("Training Random Forest...")
    rfc = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [200, 300],
        'max_depth': [15, 20],
        'min_samples_split': [5, 10],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }
    grid_search = GridSearchCV(rfc, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train_balanced)
    
    # Best model
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training time: {(time() - start_time) / 60:.2f} minutes")
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    total_accuracy = accuracy_score(y_test, y_pred)
    print(f"Total accuracy: {total_accuracy:.4f}")
    
    # Per-quadrant accuracy
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    quadrant_accuracies = {}
    results = []
    for q in quadrants:
        mask = y_test == q
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            quadrant_accuracies[q] = acc
            print(f"Accuracy for {q}: {acc:.4f}")
            results.append({'Quadrant': q, 'Accuracy': acc})
    
    # Per-file accuracy
    file_results = []
    for dataset, name in [(dynamic_data, 'dynamic'), (static_data, 'static')]:
        test_data = dataset[dataset.index.isin(X_test.index)]
        if not test_data.empty:
            test_data = test_data.copy()
            test_data['rssi_1_2_diff'] = test_data['rssi_1'] - test_data['rssi_2']
            test_data['rssi_3_4_diff'] = test_data['rssi_3'] - test_data['rssi_4']
            test_data['phase_angle_1_2_diff'] = test_data['phase_angle_1'] - test_data['phase_angle_2']
            test_data['phase_angle_3_4_diff'] = test_data['phase_angle_3'] - test_data['phase_angle_4']
            X_test_data = test_data[features]
            X_test_data = X_test_data.fillna(X_test_data.mean())
            X_test_data_scaled = scaler.transform(X_test_data)
            y_test_data = test_data[target]
            y_pred_data = model.predict(X_test_data_scaled)
            
            for filename in test_data['filename'].unique():
                mask = test_data['filename'] == filename
                file_acc = accuracy_score(y_test_data[mask], y_pred_data[mask])
                file_results.append({
                    'Dataset': name,
                    'Filename': filename,
                    'Total Accuracy': file_acc
                })
                for q in quadrants:
                    q_mask = (test_data['filename'] == filename) & (y_test_data == q)
                    if q_mask.sum() > 0:
                        file_results[-1][f'{q} Accuracy'] = accuracy_score(y_test_data[q_mask], y_pred_data[q_mask])
                    else:
                        file_results[-1][f'{q} Accuracy'] = np.nan
    
    # Save results
    results_df = pd.DataFrame(results + file_results)
    results_file = 'quadrant_results.csv'
    with open(results_file, 'w') as f:
        f.write(
            "# Quadrant and Per-File Results\n"
            "# This file contains quadrant accuracies and per-file accuracies for dynamic and static data.\n"
            "# - Quadrant: Q1 (x<7.5, y>=7.5), Q2 (x>=7.5, y>=7.5), Q3 (x<7.5, y<7.5), Q4 (x>=7.5, y<7.5).\n"
            "# - Accuracy: Prediction accuracy for each quadrant or file (0 to 1).\n"
            "# - Dataset: 'dynamic' (MoveTest2) or 'static' (Test8).\n"
            "# - Filename: Name of the data file (e.g., MoveTest2-1.csv, (12,3).csv).\n"
        )
    results_df.to_csv(results_file, mode='a', index=False)
    print(f"Results saved to: {os.path.abspath(results_file)}")
    
    # Feature importance
    importances = model.feature_importances_
    antenna_positions = {
        'rssi_1': 'A1=[0,0]', 'rssi_2': 'A2=[0,15]', 'rssi_3': 'A3=[15,15]', 'rssi_4': 'A4=[15,0]',
        'phase_angle_1': 'A1=[0,0]', 'phase_angle_2': 'A2=[0,15]', 'phase_angle_3': 'A3=[15,15]', 'phase_angle_4': 'A4=[15,0]',
        'doppler_frequency_1': 'A1=[0,0]', 'doppler_frequency_2': 'A2=[0,15]', 'doppler_frequency_3': 'A3=[15,15]', 'doppler_frequency_4': 'A4=[15,0]',
        'rssi_1_2_diff': 'A1-A2', 'rssi_3_4_diff': 'A3-A4', 'phase_angle_1_2_diff': 'A1-A2', 'phase_angle_3_4_diff': 'A3-A4'
    }
    feature_imp_df = pd.DataFrame({
        'Feature': features,
        'Importance': importances,
        'Antenna_Position': [antenna_positions[f] for f in features]
    })
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
    feature_imp_file = 'feature_importance.csv'
    with open(feature_imp_file, 'w') as f:
        f.write(
            "# Feature Importance\n"
            "# This file shows the importance of each feature in the Random Forest model.\n"
            "# - Feature: The signal feature (rssi, phase_angle, doppler_frequency) or pairwise difference for each antenna (1-4).\n"
            "# - Importance: The relative importance score (higher means more contribution to quadrant prediction).\n"
            "# - Antenna_Position: The antenna's coordinates in the 15x15 ft grid (A1=[0,0], A2=[0,15], A3=[15,15], A4=[15,0]) or difference pair (e.g., A1-A2).\n"
        )
    feature_imp_df.to_csv(feature_imp_file, mode='a', index=False)
    print(f"Feature importance saved to: {os.path.abspath(feature_imp_file)}")
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
    plt.title('Feature Importance (Antenna Positions: A1=[0,0], A2=[0,15], A3=[15,15], A4=[15,0])')
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=quadrants)
    cm_df = pd.DataFrame(cm, index=quadrants, columns=quadrants)
    cm_file = 'confusion_matrix.csv'
    with open(cm_file, 'w') as f:
        f.write(
            "# Confusion Matrix\n"
            "# This file shows the confusion matrix for quadrant predictions.\n"
            "# - Rows: Actual quadrants (Q1, Q2, Q3, Q4).\n"
            "# - Columns: Predicted quadrants (Q1, Q2, Q3, Q4).\n"
            "# - Values: Number of samples for each actual-predicted pair.\n"
        )
    cm_df.to_csv(cm_file, mode='a')
    print(f"Confusion matrix saved to: {os.path.abspath(cm_file)}")
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=quadrants, yticklabels=quadrants)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    # Actual vs. Predicted plot
    test_data = data.loc[X_test.index]
    test_data['predicted_quadrant'] = y_pred
    plt.figure(figsize=(10, 10))
    for q in quadrants:
        actual = test_data[test_data['quadrant'] == q]
        predicted = test_data[test_data['predicted_quadrant'] == q]
        plt.scatter(actual['x_true'], actual['y_true'], label=f'Actual {q}', alpha=0.5)
        plt.scatter(predicted['x_true'], predicted['y_true'], label=f'Predicted {q}', marker='x')
    plt.xlabel('X (ft)')
    plt.ylabel('Y (ft)')
    plt.title('Actual vs. Predicted Quadrants')
    plt.legend()
    plt.grid(True)
    plt.savefig('predictions.png')
    plt.close()
    
    # Save model
    joblib.dump(model, 'rfid_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')

if __name__ == '__main__':
    main()