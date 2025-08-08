import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
from time import time
from datetime import datetime

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

# Parse quadrant from filename
def parse_quadrant_from_filename(filename):
    """Extract quadrant from filename like quad1.csv, quadz1_train.csv, or Test8-X.csv."""
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    if base.startswith('quad') or base.startswith('quadz'):
        quadrant_num = base.replace('quad', '').replace('z', '').replace('_train', '').replace('_test', '')
        if quadrant_num in ['1', '2', '3', '4']:
            return f'Q{quadrant_num}'
    elif base.startswith('test8-'):
        test_num = int(base.replace('test8-', ''))
        coordinates = {
            1: (2, 5), 2: (2, 8), 3: (2, 11), 4: (2, 14),
            5: (5, 5), 6: (5, 8), 7: (5, 11), 8: (5, 14),
            9: (8, 5), 10: (8, 8), 11: (8, 11), 12: (8, 14),
            13: (11, 5), 14: (11, 8), 15: (11, 11), 16: (11, 14)
        }
        if test_num in coordinates:
            x, y = coordinates[test_num]
            return coordinates_to_quadrant(x, y)
    print(f"Warning: Invalid filename {filename}, expecting quad1.csv, quadz1_train.csv, or Test8-X.csv.")
    return None

# Load and preprocess data
def load_and_preprocess_data(file_paths):
    """Load and preprocess RFID data from CSV files."""
    data_frames = []
    
    for file_path, quadrant in file_paths:
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            continue
        print(f"Processing file: {file_path}")
        try:
            df = pd.read_csv(file_path)
            # Convert ISO timestamp to Unix seconds if present
            if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
                try:
                    df['timestamp'] = pd.to_datetime(df['timestamp']).astype(int) / 10**9
                except Exception as e:
                    print(f"Error converting timestamp in {file_path}: {str(e)}")
                    continue
            # Log and clean data
            if 'rssi' in df.columns:
                print(f"RSSI range in {file_path}: {df['rssi'].min()} to {df['rssi'].max()}")
                outliers_rssi = (df['rssi'] < -100) | (df['rssi'] > -30)
                print(f"Outliers in rssi (<-100 or >-30 dBm) in {file_path}: {outliers_rssi.sum()}")
                df.loc[outliers_rssi, 'rssi'] = np.nan
            # Validate quadrant from filename
            parsed_quadrant = parse_quadrant_from_filename(file_path)
            if parsed_quadrant is None or parsed_quadrant != quadrant:
                print(f"Skipping {file_path}: Filename does not match expected quadrant {quadrant}.")
                continue
            # Pivot to include only rssi
            pivoted = df.pivot_table(
                index='timestamp',
                columns='antenna',
                values=['rssi'],
                aggfunc='mean'
            )
            pivoted.columns = [f'rssi_{col[1]}' for col in pivoted.columns]
            pivoted = pivoted.reset_index()
            # Fill NaNs with mean for RSSI features
            for col in pivoted.columns:
                if col.startswith('rssi_'):
                    pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
            # Compute RSSI difference features
            pivoted['rssi_diff_1_2'] = pivoted['rssi_1'] - pivoted['rssi_2']
            pivoted['rssi_diff_3_4'] = pivoted['rssi_3'] - pivoted['rssi_4']
            pivoted['rssi_diff_1_4'] = pivoted['rssi_1'] - pivoted['rssi_4']
            pivoted['rssi_diff_2_3'] = pivoted['rssi_2'] - pivoted['rssi_3']
            pivoted['rssi_diff_1_3'] = pivoted['rssi_1'] - pivoted['rssi_3']
            pivoted['rssi_diff_4_2'] = pivoted['rssi_4'] - pivoted['rssi_2']
            # Fill NaNs for difference features
            for col in ['rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3', 'rssi_diff_1_3', 'rssi_diff_4_2']:
                pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
            # Drop rows with excessive NaNs
            nan_threshold = len(pivoted.columns) * 0.1
            if pivoted.isna().sum().max() > nan_threshold:
                print(f"Warning: Dropping rows with excessive NaNs in {file_path}.")
                pivoted = pivoted.dropna()
            # Check for remaining NaNs
            nan_counts = pivoted.isna().sum()
            print(f"NaN counts in {file_path} after imputation: {nan_counts[nan_counts > 0].to_dict()}")
            if pivoted.empty:
                print(f"Error: No data remains in {file_path} after preprocessing.")
                continue
            if pivoted.isna().any().any():
                print(f"Warning: Remaining NaNs in {file_path} after imputation. Dropping rows.")
                pivoted = pivoted.dropna()
            if pivoted.empty:
                print(f"Error: All rows dropped in {file_path} due to NaNs.")
                continue
            pivoted['quadrant'] = quadrant
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
    """Main function to load data, train model, and evaluate on test data."""
    start_time = time()
    # Base path for files
    base_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/quadTesting/'
    test8_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
    # Training file paths (dynamic + static Test 8 data, exclude Test8-13.csv)
    train_files = [
        (f'{base_path}quad1_train.csv', 'Q1'),
        (f'{base_path}quad2_train.csv', 'Q2'),
        (f'{base_path}quad3_train.csv', 'Q3'),
        (f'{base_path}quad4_train.csv', 'Q4'),
        (f'{base_path}quadz1_train.csv', 'Q1'),
        (f'{base_path}quadz2_train.csv', 'Q2'),
        (f'{base_path}quadz3_train.csv', 'Q3'),
        (f'{base_path}quadz4_train.csv', 'Q4'),
        # Test 8 static data (12 files, exclude Test8-13.csv)
        (f'{test8_path}Test8-1.csv', 'Q1'),  # (2, 5)
        (f'{test8_path}Test8-2.csv', 'Q2'),  # (2, 8)
        (f'{test8_path}Test8-3.csv', 'Q2'),  # (2, 11)
        (f'{test8_path}Test8-6.csv', 'Q2'),  # (5, 8)
        (f'{test8_path}Test8-7.csv', 'Q2'),  # (5, 11)
        (f'{test8_path}Test8-10.csv', 'Q3'), # (8, 8)
        (f'{test8_path}Test8-11.csv', 'Q3'), # (8, 11)
        (f'{test8_path}Test8-12.csv', 'Q3'), # (8, 14)
        (f'{test8_path}Test8-14.csv', 'Q3'), # (11, 8)
        (f'{test8_path}Test8-15.csv', 'Q3'), # (11, 11)
        (f'{test8_path}Test8-16.csv', 'Q3'), # (11, 14)
        (f'{test8_path}Test8-9.csv', 'Q4'),  # (8, 5)
    ]
    # Test file paths (dynamic + static Test 8 data, include Test8-13.csv)
    test_files = [
        (f'{base_path}quad1_test.csv', 'Q1'),
        (f'{base_path}quad2_test.csv', 'Q2'),
        (f'{base_path}quad3_test.csv', 'Q3'),
        (f'{base_path}quad4_test.csv', 'Q4'),
        (f'{base_path}quadz1_test.csv', 'Q1'),
        (f'{base_path}quadz2_test.csv', 'Q2'),
        (f'{base_path}quadz3_test.csv', 'Q3'),
        (f'{base_path}quadz4_test.csv', 'Q4'),
        # Test 8 static data (4 files, include Test8-13.csv)
        (f'{test8_path}Test8-5.csv', 'Q1'),  # (5, 5)
        (f'{test8_path}Test8-4.csv', 'Q2'),  # (2, 14)
        (f'{test8_path}Test8-8.csv', 'Q2'),  # (5, 14)
        (f'{test8_path}Test8-13.csv', 'Q4'), # (11, 5)
    ]
    
    # Load training data
    print("Loading training data...")
    train_data = load_and_preprocess_data(train_files)
    if train_data is None or train_data.empty:
        print("Error: Failed to load training data. Exiting.")
        return
    
    # Load test data
    print("\nLoading test data...")
    test_data = load_and_preprocess_data(test_files)
    if test_data is None or test_data.empty:
        print("Error: Failed to load test data. Exiting.")
        return
    
    # Log sample counts per quadrant
    print("\nTraining sample counts per quadrant:")
    print(train_data['quadrant'].value_counts())
    print("\nTest sample counts per quadrant:")
    print(test_data['quadrant'].value_counts())
    
    # Define features (10 features)
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3', 'rssi_diff_1_3', 'rssi_diff_4_2'
    ]
    target = 'quadrant'
    
    # Verify no NaNs
    for dataset, name in [(train_data, 'training'), (test_data, 'test')]:
        nan_counts = dataset[features].isna().sum()
        print(f"NaN counts in {name} features: {nan_counts[nan_counts > 0].to_dict()}")
        if dataset[features].isna().any().any():
            print(f"Warning: Dropping rows with NaNs in {name} data.")
            dataset = dataset.dropna(subset=features)
    
    # Prepare training and test sets
    X_train = train_data[features]
    y_train = train_data[target]
    X_test = test_data[features]
    y_test = test_data[target]
    
    # Apply SMOTE to balance training data
    print("\nApplying SMOTE to balance training data...")
    smote = SMOTE(random_state=42, sampling_strategy={'Q1': 6000, 'Q2': 6000, 'Q3': 6000, 'Q4': 6000})
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Balanced training sample counts per quadrant:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with GridSearchCV
    print("\nTraining Random Forest...")
    rfc = RandomForestClassifier(random_state=42, class_weight='balanced')
    param_grid = {
        'n_estimators': [100, 150],
        'max_depth': [1],
        'min_samples_split': [250, 300],
        'min_samples_leaf': [200, 250, 300],
        'max_features': ['sqrt'],
        'max_samples': [0.2, 0.3]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1, verbose=2)
    grid_search.fit(X_train_scaled, y_train_balanced)
    
    # Best model
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Training time: {(time() - start_time) / 60:.2f} minutes")
    
    # Evaluate on training set
    y_train_pred = model.predict(X_train_scaled)
    train_accuracy = accuracy_score(y_train_balanced, y_train_pred)
    print(f"\nTraining accuracy: {train_accuracy:.4f}")
    print("\nTraining classification report:")
    print(classification_report(y_train_balanced, y_train_pred, target_names=['Q1', 'Q2', 'Q3', 'Q4']))
    
    # Evaluate on test set
    y_test_pred = model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, y_test_pred, target_names=['Q1', 'Q2', 'Q3', 'Q4']))
    
    # Save live predictions (for evaluation, exclude Actual_Quadrant)
    live_predictions = pd.DataFrame({
        'Timestamp': test_data['timestamp'],
        'Predicted_Quadrant': y_test_pred
    })
    live_predictions_file = 'live_predictions.csv'
    with open(live_predictions_file, 'w') as f:
        f.write(
            "# Live Predictions\n"
            "# This file contains predicted quadrants for test data.\n"
            "# - Timestamp: Time of the RFID reading.\n"
            "# - Predicted_Quadrant: Predicted quadrant by the model.\n"
        )
    live_predictions.to_csv(live_predictions_file, mode='w', index=False)
    print(f"Live predictions saved to: {os.path.abspath(live_predictions_file)}")
    
    # Per-quadrant accuracy
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    results = []
    for q in quadrants:
        mask = y_test == q
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_test_pred[mask])
            print(f"Accuracy for {q}: {acc:.4f}")
            results.append({'Quadrant': q, 'Accuracy': acc})
    
    # Save results
    results_df = pd.DataFrame(results)
    results_file = 'quadrant_results.csv'
    with open(results_file, 'w') as f:
        f.write(
            "# Quadrant Results\n"
            "# This file contains quadrant accuracies.\n"
            "# - Quadrant: Q1 (x<7.5, y<7.5), Q2 (x<7.5, y>=7.5), Q3 (x>=7.5, y>=7.5), Q4 (x>=7.5, y<7.5).\n"
            "# - Accuracy: Prediction accuracy for each quadrant (0 to 1).\n"
        )
    results_df.to_csv(results_file, mode='a', index=False)
    print(f"Results saved to: {os.path.abspath(results_file)}")
    
    # Feature importance
    importances = model.feature_importances_
    antenna_positions = {
        'rssi_1': 'A1=[0,0]', 'rssi_2': 'A2=[0,15]', 'rssi_3': 'A3=[15,15]', 'rssi_4': 'A4=[15,0]',
        'rssi_diff_1_2': 'A1-A2', 'rssi_diff_3_4': 'A3-A4', 'rssi_diff_1_4': 'A1-A4', 'rssi_diff_2_3': 'A2-A3', 
        'rssi_diff_1_3': 'A1-A3', 'rssi_diff_4_2': 'A4-A2'
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
            "# - Feature: The RSSI feature for each antenna (1-4) or RSSI difference.\n"
            "# - Importance: The relative importance score (higher means more contribution to quadrant prediction).\n"
            "# - Antenna_Position: The antenna's coordinates in the 15x15 ft grid (A1=[0,0], A2=[0,15], A3=[15,15], A4=[15,0]) or 'A1-A2'/'A3-A4'/'A1-A4'/'A2-A3'/'A1-A3'/'A4-A2' for differences.\n"
        )
    feature_imp_df.to_csv(feature_imp_file, mode='a', index=False)
    print(f"Feature importance saved to: {os.path.abspath(feature_imp_file)}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred, labels=quadrants)
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
    
    # Save model and scaler
    joblib.dump(model, 'model_full_features.pkl')
    joblib.dump(scaler, 'scaler_full_features.pkl')
    print(f"Model saved to: {os.path.abspath('model_full_features.pkl')}")
    print(f"Scaler saved to: {os.path.abspath('scaler_full_features.pkl')}")

if __name__ == '__main__':
    main()