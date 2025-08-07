import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import joblib
import os
import time
from datetime import datetime
import logging
import re
import glob

# Configure logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def coordinates_to_quadrant(x, y):
    if x < 7.5 and y < 7.5:
        return 'Q1'
    elif x < 7.5 and y >= 7.5:
        return 'Q2'
    elif x >= 7.5 and y >= 7.5:
        return 'Q3'
    else:
        return 'Q4'

def parse_quadrant_from_filename(filename):
    fname = os.path.basename(filename).lower()
    coord_pattern = r'\((\d+),(\d+)\)\.csv'
    match = re.match(coord_pattern, fname)
    if match:
        x, y = map(int, match.groups())
        return coordinates_to_quadrant(x, y)
    if fname.startswith('quad') or fname.startswith('quadz'):
        try:
            quad_num = int(''.join(filter(str.isdigit, fname)))
            return f'Q{quad_num}' if 1 <= quad_num <= 4 else None
        except ValueError:
            logger.warning(f"Could not parse quadrant from filename: {filename}")
            return None
    logger.warning(f"Unknown filename format: {filename}")
    return None

def get_test8_files(test8_path):
    x_coords = [1, 3, 5, 7, 8, 10, 12, 14]
    y_coords = [1, 3, 5, 7, 8, 10, 12, 14]
    test_positions = {
        f'({x},{y}).csv': (x, y) for x in x_coords for y in y_coords
    }
    files = []
    found_files = glob.glob(os.path.join(test8_path, '(*,*).csv'))
    for fname in found_files:
        base_fname = os.path.basename(fname).lower()
        if base_fname in test_positions:
            x, y = test_positions[base_fname]
            quadrant = coordinates_to_quadrant(x, y)
            files.append((fname, quadrant))
        else:
            logger.warning(f"Skipping unrecognized file: {fname}")
    logger.info(f"Found {len(files)} Test8 files: {[os.path.basename(f[0]) for f in files]}")
    return files

def load_and_preprocess_data(file_paths):
    all_data = []
    rssi_stats = []
    for file_path, quadrant in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            logger.debug(f"Loaded {file_path}: {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            continue
        parsed_quadrant = parse_quadrant_from_filename(file_path)
        if parsed_quadrant != quadrant:
            logger.warning(f"Quadrant mismatch for {file_path}: expected {quadrant}, got {parsed_quadrant}")
            continue
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            try:
                df['timestamp'] = pd.to_datetime(df['timestamp']).astype(np.int64) / 10**9
            except Exception as e:
                logger.error(f"Error converting timestamps in {file_path}: {e}")
                continue
        if 'rssi' not in df.columns or 'antenna' not in df.columns:
            logger.warning(f"Missing required columns in {file_path}: {list(df.columns)}")
            continue
        invalid_rssi = (df['rssi'] < -100) | (df['rssi'] > -30)
        if invalid_rssi.any():
            logger.debug(f"Found {invalid_rssi.sum()} invalid RSSI values in {file_path}")
            df.loc[invalid_rssi, 'rssi'] = np.nan
        agg_funcs = ['mean', 'std']
        values = ['rssi']
        if 'doppler' in df.columns:
            agg_funcs.append('mean')
            values.append('doppler')
            logger.debug(f"Doppler data found in {file_path}")
        else:
            logger.warning(f"No Doppler data in {file_path}; setting doppler_rssi_1-4 to 0.0")
        pivoted = df.pivot_table(index='timestamp', columns='antenna', values=values, aggfunc=agg_funcs).reset_index()
        if pivoted.empty or len(pivoted) < 2:
            logger.warning(f"No valid data after pivoting in {file_path}: {len(pivoted)} rows")
            continue
        # Flatten multi-index columns
        flat_columns = ['timestamp']
        for col in pivoted.columns[1:]:
            stat, val, ant = col
            if val == 'rssi' and stat == 'mean':
                flat_columns.append(f'rssi_{ant}')
            elif val == 'rssi':
                flat_columns.append(f'{stat}_rssi_{ant}')
            elif val == 'doppler':
                flat_columns.append(f'doppler_rssi_{ant}')
        pivoted.columns = flat_columns
        logger.debug(f"Pivoted columns for {file_path}: {list(pivoted.columns)}")
        
        # Save mean and std for rssi_stats.csv
        for i in range(1, 5):
            mean_col = f'rssi_{i}'
            std_col = f'std_rssi_{i}'
            doppler_col = f'doppler_rssi_{i}'
            if mean_col in pivoted.columns:
                mean_rssi = pivoted[mean_col].mean()
                std_rssi = pivoted[std_col].std() if std_col in pivoted.columns else 0.0
                doppler_rssi = pivoted[doppler_col].mean() if doppler_col in pivoted.columns else 0.0
                rssi_stats.append({
                    'file': os.path.basename(file_path),
                    'antenna': i,
                    'mean_rssi': mean_rssi if not pd.isna(mean_rssi) else -80.0,
                    'std_rssi': std_rssi if not pd.isna(std_rssi) else 0.0,
                    'doppler_rssi': doppler_rssi if not pd.isna(doppler_rssi) else 0.0
                })
                logger.debug(f"{file_path} - Antenna {i}: mean_rssi={mean_rssi:.2f}, std_rssi={std_rssi:.2f}, doppler_rssi={doppler_rssi:.2f}")
        
        for i in range(1, 5):
            mean_col = f'rssi_{i}'
            std_col = f'std_rssi_{i}'
            doppler_col = f'doppler_rssi_{i}'
            if mean_col not in pivoted.columns:
                pivoted[mean_col] = -80.0
                pivoted[std_col] = 0.0
                pivoted[doppler_col] = 0.0
            pivoted[mean_col] = pivoted[mean_col].fillna(-80.0)
            pivoted[std_col] = pivoted[std_col].fillna(0.0)
            pivoted[doppler_col] = pivoted[doppler_col].fillna(0.0)
        pivoted['rssi_diff_1_2'] = pivoted['rssi_1'] - pivoted['rssi_2']
        pivoted['rssi_diff_3_4'] = pivoted['rssi_3'] - pivoted['rssi_4']
        pivoted['rssi_diff_1_4'] = pivoted['rssi_1'] - pivoted['rssi_4']
        pivoted['rssi_diff_2_3'] = pivoted['rssi_2'] - pivoted['rssi_3']
        pivoted['rssi_diff_1_3'] = pivoted['rssi_1'] - pivoted['rssi_3']
        pivoted['rssi_diff_4_2'] = pivoted['rssi_4'] - pivoted['rssi_2']
        for col in pivoted.columns:
            if col.startswith('rssi_diff'):
                pivoted[col] = pivoted[col].fillna(pivoted[col].mean())
        nan_threshold = int(0.1 * len(pivoted.columns))
        nan_counts = pivoted.isna().sum(axis=1)
        pivoted = pivoted[nan_counts <= nan_threshold]
        if pivoted.isna().any().any():
            logger.debug(f"Dropping {pivoted.isna().any(axis=1).sum()} rows with NaNs in {file_path}")
            pivoted = pivoted.dropna()
        if pivoted.empty:
            logger.warning(f"No valid data after preprocessing in {file_path}: {len(pivoted)} rows")
            continue
        pivoted['quadrant'] = quadrant
        all_data.append(pivoted)
    if not all_data:
        logger.error("No valid data loaded")
        return None
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    
    pd.DataFrame(rssi_stats).to_csv('rssi_stats.csv', index=False)
    logger.info("Saved RSSI and Doppler stats to rssi_stats.csv")
    
    return combined

def main():
    base_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/quadTesting/'
    test8_path = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'
    
    train_files = [
        (f'{base_path}quad1_train.csv', 'Q1'),
        (f'{base_path}quadz1_train.csv', 'Q1'),
        (f'{base_path}quad2_train.csv', 'Q2'),
        (f'{base_path}quadz2_train.csv', 'Q2'),
        (f'{base_path}quad3_train.csv', 'Q3'),
        (f'{base_path}quadz3_train.csv', 'Q3'),
        (f'{base_path}quad4_train.csv', 'Q4'),
        (f'{base_path}quadz4_train.csv', 'Q4')
    ]
    test_files = [
        (f'{base_path}quad1_test.csv', 'Q1'),
        (f'{base_path}quadz1_test.csv', 'Q1'),
        (f'{base_path}quad2_test.csv', 'Q2'),
        (f'{base_path}quadz2_test.csv', 'Q2'),
        (f'{base_path}quad3_test.csv', 'Q3'),
        (f'{base_path}quadz3_test.csv', 'Q3'),
        (f'{base_path}quad4_test.csv', 'Q4'),
        (f'{base_path}quadz4_test.csv', 'Q4')
    ]
    
    test8_files = get_test8_files(test8_path)
    train_files.extend([f for i, f in enumerate(test8_files) if i % 2 == 0])
    test_files.extend([f for i, f in enumerate(test8_files) if i % 2 == 1])
    
    logger.info(f"Training files: {[os.path.basename(f[0]) for f in train_files]}")
    logger.info(f"Test files: {[os.path.basename(f[0]) for f in test_files]}")
    
    train_data = load_and_preprocess_data(train_files)
    test_data = load_and_preprocess_data(test_files)
    if train_data is None or test_data is None:
        logger.error("Failed to load data")
        return
    logger.info("Train quadrant counts:\n" + train_data['quadrant'].value_counts().to_string())
    logger.info("Test quadrant counts:\n" + test_data['quadrant'].value_counts().to_string())
    features = [
        'rssi_1', 'rssi_2', 'rssi_3', 'rssi_4',
        'std_rssi_1', 'std_rssi_2', 'std_rssi_3', 'std_rssi_4',
        'doppler_rssi_1', 'doppler_rssi_2', 'doppler_rssi_3', 'doppler_rssi_4',
        'rssi_diff_1_2', 'rssi_diff_3_4', 'rssi_diff_1_4', 'rssi_diff_2_3',
        'rssi_diff_1_3', 'rssi_diff_4_2'
    ]
    X_train = train_data[features]
    y_train = train_data['quadrant']
    X_test = test_data[features]
    y_test = test_data['quadrant']
    if X_train.isna().any().any() or X_test.isna().any().any():
        logger.debug(f"Dropping rows with NaNs: Train={X_train.isna().any(axis=1).sum()}, Test={X_test.isna().any(axis=1).sum()}")
        X_train = X_train.dropna()
        y_train = y_train[X_train.index]
        X_test = X_test.dropna()
        y_test = y_test[X_test.index]
    smote = SMOTE(sampling_strategy={'Q1': 10000, 'Q2': 10000, 'Q3': 10000, 'Q4': 10000}, random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    logger.info("Balanced train quadrant counts:\n" + pd.Series(y_train_balanced).value_counts().to_string())
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test)
    param_grid = {
        'n_estimators': [100, 150], 'max_depth': [1], 'min_samples_split': [250, 300],
        'min_samples_leaf': [200, 250, 300], 'max_features': ['sqrt'], 'max_samples': [0.2, 0.3]
    }
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    start_time = time.time()
    grid_search.fit(X_train_scaled, y_train_balanced)
    logger.info(f"Training time: {time.time() - start_time:.2f} seconds")
    logger.info(f"Best parameters: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    logger.info("Train accuracy: " + str(accuracy_score(y_train_balanced, y_train_pred)))
    logger.info("Train classification report:\n" + classification_report(y_train_balanced, y_train_pred))
    logger.info("Test accuracy: " + str(accuracy_score(y_test, y_test_pred)))
    logger.info("Test classification report:\n" + classification_report(y_test, y_test_pred))
    
    with open('live_predictions.csv', 'w') as f:
        f.write('# Live Predictions\n')
        f.write('# This file contains predicted quadrants for test data.\n')
        f.write('# - Timestamp: Time of the RFID reading.\n')
        f.write('# - Predicted_Quadrant: Predicted quadrant by the model.\n')
    pd.DataFrame({'Timestamp': test_data['timestamp'], 'Predicted_Quadrant': y_test_pred}).to_csv(
        'live_predictions.csv', mode='a', index=False, header=['Timestamp', 'Predicted_Quadrant']
    )
    
    quadrant_accuracies = []
    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = y_test == quadrant
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_test_pred[mask])
            quadrant_accuracies.append({'Quadrant': quadrant, 'Accuracy': acc})
    with open('quadrant_results.csv', 'w') as f:
        f.write('# Quadrant Results\n')
        f.write('# This file contains quadrant accuracies.\n')
        f.write('# - Quadrant: Q1 (x<7.5, y<7.5), Q2 (x<7.5, y>=7.5), Q3 (x>=7.5, y>=7.5), Q4 (x>=7.5, y<7.5).\n')
        f.write('# - Accuracy: Prediction accuracy for each quadrant (0 to 1).\n')
    pd.DataFrame(quadrant_accuracies).to_csv(
        'quadrant_results.csv', mode='a', index=False, header=['Quadrant', 'Accuracy']
    )
    
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_,
        'Antenna_Position': [
            'A1=[0,0]', 'A2=[0,15]', 'A3=[15,15]', 'A4=[15,0]',
            'A1=[0,0]', 'A2=[0,15]', 'A3=[15,15]', 'A4=[15,0]',
            'A1=[0,0]', 'A2=[0,15]', 'A3=[15,15]', 'A4=[15,0]',
            'A1-A2', 'A3-A4', 'A1-A4', 'A2-A3', 'A1-A3', 'A4-A2'
        ]
    })
    with open('feature_importance.csv', 'w') as f:
        f.write('# Feature Importance\n')
        f.write('# This file shows the importance of each feature in the Random Forest model.\n')
        f.write('# - Feature: Raw/Std/Doppler RSSI for each antenna (1-4) or RSSI difference.\n')
        f.write('# - Importance: The relative importance score (higher means more contribution to quadrant prediction).\n')
        f.write('# - Antenna_Position: The antenna\'s coordinates in the 15x15 ft grid or difference pair.\n')
    feature_importance.to_csv(
        'feature_importance.csv', mode='a', index=False, header=['Feature', 'Importance', 'Antenna_Position']
    )
    
    cm = confusion_matrix(y_test, y_test_pred, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    with open('confusion_matrix.csv', 'w') as f:
        f.write('# Confusion Matrix\n')
        f.write('# This file shows the confusion matrix for quadrant predictions.\n')
        f.write('# - Rows: Actual quadrants (Q1, Q2, Q3, Q4).\n')
        f.write('# - Columns: Predicted quadrants (Q1, Q2, Q3, Q4).\n')
        f.write('# - Values: Number of samples for each actual-predicted pair.\n')
    pd.DataFrame(cm, index=['Q1', 'Q2', 'Q3', 'Q4'], columns=['Q1', 'Q2', 'Q3', 'Q4']).to_csv(
        'confusion_matrix.csv', mode='a', index=True, header=True
    )
    
    joblib.dump(model, 'model_full_features.pkl')
    joblib.dump(scaler, 'scaler_full_features.pkl')
    logger.info("Model and scaler saved to model_full_features.pkl and scaler_full_features.pkl")

if __name__ == '__main__':
    main()