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
from time import time

# Define quadrant mapping function
def coordinates_to_quadrant(x, y):
    """Map x, y coordinates to quadrants in a 15x15 ft grid."""
    if x < 7.5 and y < 7.5:
        return 'Q1'  # Bottom-left
    elif x < 7.5 and y >= 7.5:
        return 'Q2'  # Top-left
    elif x >= 7.5 and y >= 7.5:
        return 'Q3'  # Top-right
    else:
        return 'Q4'  # Bottom-right

# Parse quadrant from filename (e.g., Q1.csv, Q2.csv)
def parse_quadrant_from_filename(filename):
    """Extract quadrant from filename like Q1.csv."""
    quadrant = os.path.splitext(os.path.basename(filename))[0].upper()
    if quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        return quadrant
    else:
        print(f"Warning: Invalid filename {filename}, expecting Q1.csv, Q2.csv, etc.")
        return None

# Load and preprocess data
def load_and_preprocess_data(file_paths):
    """Load and preprocess RFID data from four CSV files, one per quadrant."""
    data_frames = []
    
    for file_path, quadrant in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            # Log and clean data
            if 'rssi' in df.columns:
                print(f"RSSI range in {file_path}: {df['rssi'].min()} to {df['rssi'].max()}")
                outliers_rssi = (df['rssi'] < -100) | (df['rssi'] > -40)
                print(f"Outliers in rssi (<-100 or >-40 dBm) in {file_path}: {outliers_rssi.sum()}")
                df.loc[outliers_rssi, 'rssi'] = np.nan
            if 'phase_angle' in df.columns:
                print(f"Phase angle range in {file_path}: {df['phase_angle'].min()} to {df['phase_angle'].max()}")
            if 'doppler_frequency' in df.columns:
                print(f"Doppler frequency range in {file_path}: {df['doppler_frequency'].min()} to {df['doppler_frequency'].max()}")
                outliers_doppler = df['doppler_frequency'].abs() > 1000
                print(f"Outliers in doppler_frequency (>1000 Hz) in {file_path}: {outliers_doppler.sum()}")
                df.loc[outliers_doppler, 'doppler_frequency'] = np.nan
            # Validate quadrant from filename
            parsed_quadrant = parse_quadrant_from_filename(file_path)
            if parsed_quadrant is None or parsed_quadrant != quadrant:
                print(f"Skipping {file_path}: Filename does not match expected quadrant {quadrant}.")
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
                if col.startswith('rssi_'):
                    pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
                # Uncomment to include phase_angle and doppler_frequency
                # if col.startswith(('phase_angle_', 'doppler_frequency_')):
                #     pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
            # Check for remaining NaNs
            nan_counts = pivoted.isna().sum()
            print(f"NaN counts in {file_path} after imputation: {nan_counts[nan_counts > 0].to_dict()}")
            if pivoted.isna().any().any():
                print(f"Warning: Remaining NaNs in {file_path} after imputation. Dropping rows.")
                pivoted = pivoted.dropna()
            pivoted['quadrant'] = quadrant
            pivoted['filename'] = os.path.basename(file_path)
            data_frames.append(pivoted)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    if not data_frames:
        print("Error: No valid data files loaded.")
        return None
    
    combined = pd.concat(data_frames, ignore_index=True)
    print(f"Combined dataset size: {len(combined)} rows")
    return combined

# Main processing
def main():
    """Main function to load data, train model, and generate outputs."""
    start_time = time()
    # File paths for each quadrant (UPDATE THESE PATHS)
    q1_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/QuadrantData/Q1.csv'
    q2_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/QuadrantData/Q2.csv'
    q3_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/QuadrantData/Q3.csv'
    q4_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/QuadrantData/Q4.csv'
    
    # List of file paths and their corresponding quadrants
    file_paths = [
        (q1_file, 'Q1'),
        (q2_file, 'Q2'),
        (q3_file, 'Q3'),
        (q4_file, 'Q4')
    ]
    
    # Load data
    print("Loading quadrant data...")
    data = load_and_preprocess_data(file_paths)
    if data is None or data.empty:
        print("Error: Failed to load data. Exiting.")
        return
    
    # Add pairwise feature differences
    data['rssi_1_2_diff'] = data['rssi_1'] - data['rssi_2']
    data['rssi_3_4_diff'] = data['rssi_3'] - data['rssi_4']
    # Uncomment to include phase_angle differences
    # data['phase_angle_1_2_diff'] = data['phase_angle_1'] - data['phase_angle_2']
    # data['phase_angle_3_4_diff'] = data['phase_angle_3'] - data['phase_angle_4']
    
    # Log sample counts per quadrant
    print("Sample counts per quadrant:")
    print(data['quadrant'].value_counts())
    
    # Features and target
    features = [f'rssi_{i}' for i in range(1, 5)] + ['rssi_1_2_diff', 'rssi_3_4_diff']
    # Uncomment to include phase_angle and doppler_frequency
    # features += [f'phase_angle_{i}' for i in range(1, 5)] + [f'doppler_frequency_{i}' for i in range(1, 5)] + ['phase_angle_1_2_diff', 'phase_angle_3_4_diff']
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
    
    # Train Random Forest
    print("Training Random Forest...")
    rfc = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [400, 500],
        'max_depth': [50],
        'min_samples_split': [2],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2', 0.5],
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
    results = []
    for q in quadrants:
        mask = y_test == q
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_pred[mask])
            print(f"Accuracy for {q}: {acc:.4f}")
            results.append({'Quadrant': q, 'Accuracy': acc})
    
    # Per-file accuracy
    file_results = []
    test_data = data.loc[X_test.index].copy()
    test_data['rssi_1_2_diff'] = test_data['rssi_1'] - test_data['rssi_2']
    test_data['rssi_3_4_diff'] = test_data['rssi_3'] - test_data['rssi_4']
    # Uncomment to include phase_angle differences
    # test_data['phase_angle_1_2_diff'] = test_data['phase_angle_1'] - test_data['phase_angle_2']
    # test_data['phase_angle_3_4_diff'] = test_data['phase_angle_3'] - test_data['phase_angle_4']
    X_test_data = test_data[features]
    X_test_data = X_test_data.fillna(X_test_data.mean())
    X_test_data_scaled = scaler.transform(X_test_data)
    y_test_data = test_data[target]
    y_pred_data = model.predict(X_test_data_scaled)
    
    for filename in test_data['filename'].unique():
        mask = test_data['filename'] == filename
        file_acc = accuracy_score(y_test_data[mask], y_pred_data[mask])
        file_results.append({
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
            "# This file contains quadrant accuracies and per-file accuracies.\n"
            "# - Quadrant: Q1 (x<7.5, y<7.5), Q2 (x<7.5, y>=7.5), Q3 (x>=7.5, y>=7.5), Q4 (x>=7.5, y<7.5).\n"
            "# - Accuracy: Prediction accuracy for each quadrant or file (0 to 1).\n"
            "# - Filename: Name of the data file (e.g., Q1.csv, Q2.csv, Q3.csv, Q4.csv).\n"
        )
    results_df.to_csv(results_file, mode='a', index=False)
    print(f"Results saved to: {os.path.abspath(results_file)}")
    
    # Feature importance
    importances = model.feature_importances_
    antenna_positions = {
        'rssi_1': 'A1=[0,0]', 'rssi_2': 'A2=[0,15]', 'rssi_3': 'A3=[15,15]', 'rssi_4': 'A4=[15,0]',
        # Uncomment to include phase_angle and doppler_frequency
        # 'phase_angle_1': 'A1=[0,0]', 'phase_angle_2': 'A2=[0,15]', 'phase_angle_3': 'A3=[15,15]', 'phase_angle_4': 'A4=[15,0]',
        # 'doppler_frequency_1': 'A1=[0,0]', 'doppler_frequency_2': 'A2=[0,15]', 'doppler_frequency_3': 'A3=[15,15]', 'doppler_frequency_4': 'A4=[15,0]',
        'rssi_1_2_diff': 'A1-A2', 'rssi_3_4_diff': 'A3-A4',
        # 'phase_angle_1_2_diff': 'A1-A2', 'phase_angle_3_4_diff': 'A3-A4'
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
            "# - Feature: The signal feature (rssi) or pairwise difference for each antenna (1-4).\n"
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
    
    # Save model and scaler
    joblib.dump(model, 'model_rssi.pkl')
    joblib.dump(scaler, 'scaler_rssi.pkl')

if __name__ == '__main__':
    main()