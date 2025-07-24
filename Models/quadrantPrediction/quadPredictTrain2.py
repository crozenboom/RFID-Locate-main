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
    """Extract quadrant from filename like quad1.csv or quad1_test.csv."""
    base = os.path.splitext(os.path.basename(filename))[0].lower()
    if base.startswith('quad'):
        quadrant_num = base.replace('quad', '').replace('_test', '')
        if quadrant_num in ['1', '2', '3', '4']:
            return f'Q{quadrant_num}'
    print(f"Warning: Invalid filename {filename}, expecting quad1.csv, quad1_test.csv, etc.")
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
            # Pivot data to get one row per timestamp
            df['timestamp'] = df['timestamp'].round(3)
            pivoted = df.pivot_table(
                index='timestamp',
                columns='antenna',
                values=['rssi'],
                aggfunc='mean'
            )
            pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
            pivoted = pivoted.reset_index()
            # Fill NaNs with mean
            for col in pivoted.columns:
                if col.startswith('rssi_'):
                    pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
            # Add rssi_1_mean, rssi_2_mean, rssi_3_mean, rssi_4_mean
            for i in range(1, 5):
                if f'rssi_{i}' in pivoted.columns:
                    pivoted[f'rssi_{i}_mean'] = pivoted[f'rssi_{i}'].rolling(window=5, min_periods=1).mean()
                    pivoted[f'rssi_{i}_mean'] = pivoted[f'rssi_{i}_mean'].fillna(pivoted[f'rssi_{i}'].mean())
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
    # Training file paths
    base_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/quadTesting/'
    train_files = [
        (f'{base_path}quad1.csv', 'Q1'),
        (f'{base_path}quad2.csv', 'Q2'),
        (f'{base_path}quad3.csv', 'Q3'),
        (f'{base_path}quad4.csv', 'Q4')
    ]
    # Test file paths
    test_files = [
        (f'{base_path}quad1_test.csv', 'Q1'),
        (f'{base_path}quad2_test.csv', 'Q2'),
        (f'{base_path}quad3_test.csv', 'Q3'),
        (f'{base_path}quad4_test.csv', 'Q4')
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
    
    # Add pairwise feature differences
    train_data['rssi_1_2_diff'] = train_data['rssi_1'] - train_data['rssi_2']
    train_data['rssi_3_4_diff'] = train_data['rssi_3'] - train_data['rssi_4']
    test_data['rssi_1_2_diff'] = test_data['rssi_1'] - test_data['rssi_2']
    test_data['rssi_3_4_diff'] = test_data['rssi_3'] - test_data['rssi_4']
    
    # Log sample counts per quadrant
    print("\nTraining sample counts per quadrant:")
    print(train_data['quadrant'].value_counts())
    print("\nTest sample counts per quadrant:")
    print(test_data['quadrant'].value_counts())
    
    # Features and target
    features = (
        [f'rssi_{i}' for i in range(1, 5)] +
        ['rssi_1_2_diff', 'rssi_3_4_diff'] +
        [f'rssi_{i}_mean' for i in range(1, 5)]
    )
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
    
    # Apply SMOTE to training data only
    smote = SMOTE(sampling_strategy='auto', random_state=42, k_neighbors=3)
    print("\nApplying SMOTE to training data...")
    try:
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    except ValueError as e:
        print(f"Error applying SMOTE: {str(e)}")
        return
    print("Sample counts after SMOTE:")
    print(pd.Series(y_train_balanced).value_counts())
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest with GridSearchCV
    print("\nTraining Random Forest...")
    rfc = RandomForestClassifier(random_state=42, class_weight='balanced', max_samples=0.4)
    param_grid = {
        'n_estimators': [80, 120],
        'max_depth': [2, 3],
        'min_samples_split': [80, 100],
        'min_samples_leaf': [40, 50],
        'max_features': ['sqrt', 'log2']
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rfc, param_grid, cv=cv, scoring='accuracy', n_jobs=-1, verbose=2)
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
        'rssi_1_2_diff': 'A1-A2', 'rssi_3_4_diff': 'A3-A4',
        'rssi_1_mean': 'A1=[0,0]', 'rssi_2_mean': 'A2=[0,15]', 'rssi_3_mean': 'A3=[15,15]', 'rssi_4_mean': 'A4=[15,0]'
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