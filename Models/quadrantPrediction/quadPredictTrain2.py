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
    test_positions = {
        'Test8-1.csv': (2, 5), 'Test8-2.csv': (5, 2), 'Test8-3.csv': (5, 5), 'Test8-4.csv': (2, 2),
        'Test8-5.csv': (2, 10), 'Test8-6.csv': (5, 13), 'Test8-7.csv': (5, 10), 'Test8-8.csv': (2, 13),
        'Test8-9.csv': (10, 10), 'Test8-10.csv': (13, 13), 'Test8-11.csv': (13, 10), 'Test8-12.csv': (10, 13),
        'Test8-13.csv': (10, 5), 'Test8-14.csv': (13, 2), 'Test8-15.csv': (13, 5), 'Test8-16.csv': (10, 2)
    }
    fname = os.path.basename(filename).lower()
    if fname.startswith('quad') or fname.startswith('quadz'):
        try:
            quad_num = int(''.join(filter(str.isdigit, fname)))
            return f'Q{quad_num}' if 1 <= quad_num <= 4 else None
        except ValueError:
            logger.warning(f"Could not parse quadrant from filename: {filename}")
            return None
    elif fname in test_positions:
        x, y = test_positions[fname]
        return coordinates_to_quadrant(x, y)
    logger.warning(f"Unknown filename format: {filename}")
    return None

def load_and_preprocess_data(file_paths):
    all_data = []
    rssi_stats = []
    for file_path, quadrant in file_paths:
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        df = pd.read_csv(file_path)
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
            logger.warning(f"Missing required columns in {file_path}")
            continue
        invalid_rssi = (df['rssi'] < -100) | (df['rssi'] > -30)
        if invalid_rssi.any():
            logger.debug(f"Found {invalid_rssi.sum()} invalid RSSI values in {file_path}")
            df.loc[invalid_rssi, 'rssi'] = np.nan
        pivoted = df.pivot_table(index='timestamp', columns='antenna', values='rssi', aggfunc='mean').reset_index()
        pivoted.columns = ['timestamp'] + [f'rssi_{int(col)}' for col in pivoted.columns[1:]]
        
        # Compute mean and std for each antenna
        for i in range(1, 5):
            rssi_col = f'rssi_{i}'
            if rssi_col in pivoted.columns:
                mean_rssi = pivoted[rssi_col].mean()
                std_rssi = pivoted[rssi_col].std()
                rssi_stats.append({
                    'file': os.path.basename(file_path),
                    'antenna': i,
                    'mean_rssi': mean_rssi if not pd.isna(mean_rssi) else -80.0,
                    'std_rssi': std_rssi if not pd.isna(std_rssi) else 0.0
                })
                logger.debug(f"{file_path} - Antenna {i}: mean_rssi={mean_rssi:.2f}, std_rssi={std_rssi:.2f}")
        
        for i in range(1, 5):
            rssi_col = f'rssi_{i}'
            if rssi_col not in pivoted.columns:
                pivoted[rssi_col] = np.nan
            pivoted[rssi_col] = pivoted[rssi_col].fillna(pivoted[rssi_col].mean())
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
            logger.warning(f"No valid data after preprocessing in {file_path}")
            continue
        pivoted['quadrant'] = quadrant
        all_data.append(pivoted)
    if not all_data:
        logger.error("No valid data loaded")
        return None
    combined = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined dataset shape: {combined.shape}")
    
    # Save RSSI stats to CSV
    pd.DataFrame(rssi_stats).to_csv('rssi_stats.csv', index=False)
    logger.info("Saved RSSI mean and std to rssi_stats.csv")
    
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
        (f'{base_path}quadz4_train.csv', 'Q4'),
        (f'{test8_path}Test8-2.csv', 'Q1'),
        (f'{test8_path}Test8-7.csv', 'Q3'),
        (f'{test8_path}Test8-10.csv', 'Q3'),
        (f'{test8_path}Test8-14.csv', 'Q4')
    ]
    test_files = [
        (f'{base_path}quad1_test.csv', 'Q1'),
        (f'{base_path}quadz1_test.csv', 'Q1'),
        (f'{base_path}quad2_test.csv', 'Q2'),
        (f'{base_path}quadz2_test.csv', 'Q2'),
        (f'{base_path}quad3_test.csv', 'Q3'),
        (f'{base_path}quadz3_test.csv', 'Q3'),
        (f'{base_path}quad4_test.csv', 'Q4'),
        (f'{base_path}quadz4_test.csv', 'Q4'),
        (f'{test8_path}Test8-1.csv', 'Q1'),
        (f'{test8_path}Test8-9.csv', 'Q3'),
        (f'{test8_path}Test8-13.csv', 'Q4'),
        (f'{test8_path}Test8-15.csv', 'Q4')
    ]
    train_data = load_and_preprocess_data(train_files)
    test_data = load_and_preprocess_data(test_files)
    if train_data is None or test_data is None:
        logger.error("Failed to load data")
        return
    logger.info("Train quadrant counts:\n" + train_data['quadrant'].value_counts().to_string())
    logger.info("Test quadrant counts:\n" + test_data['quadrant'].value_counts().to_string())
    features = ['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4', 'rssi_diff_1_2', 'rssi_diff_3_4',
                'rssi_diff_1_4', 'rssi_diff_2_3', 'rssi_diff_1_3', 'rssi_diff_4_2']
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
    smote = SMOTE(sampling_strategy={'Q1': 6000, 'Q2': 6000, 'Q3': 6000, 'Q4': 6000}, random_state=42)
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
    pd.DataFrame({'Timestamp': test_data['timestamp'], 'Predicted_Quadrant': y_test_pred}).to_csv(
        'live_predictions.csv', index=False,
        header=['# Live Predictions', '# This file contains predicted quadrants for test data.',
                '# - Timestamp: Time of the RFID reading.', '# - Predicted_Quadrant: Predicted quadrant by the model.']
    )
    quadrant_accuracies = []
    for quadrant in ['Q1', 'Q2', 'Q3', 'Q4']:
        mask = y_test == quadrant
        if mask.sum() > 0:
            acc = accuracy_score(y_test[mask], y_test_pred[mask])
            quadrant_accuracies.append({'Quadrant': quadrant, 'Accuracy': acc})
    pd.DataFrame(quadrant_accuracies).to_csv(
        'quadrant_results.csv', index=False,
        header=['# Quadrant Results', '# This file contains quadrant accuracies.',
                '# - Quadrant: Q1 (x<7.5, y<7.5), Q2 (x<7.5, y>=7.5), Q3 (x>=7.5, y>=7.5), Q4 (x>=7.5, y<7.5).',
                '# - Accuracy: Prediction accuracy for each quadrant (0 to 1).']
    )
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_,
        'Antenna_Position': ['A1=[0,0]', 'A2=[0,15]', 'A3=[15,15]', 'A4=[15,0]', 'A1-A2', 'A3-A4', 'A1-A4', 'A2-A3', 'A1-A3', 'A4-A2']
    })
    feature_importance.to_csv(
        'feature_importance.csv', index=False,
        header=['# Feature Importance', '# This file shows the importance of each feature in the Random Forest model.',
                '# - Feature: The RSSI feature for each antenna (1-4) or RSSI difference.',
                '# - Importance: The relative importance score (higher means more contribution to quadrant prediction).',
                '# - Antenna_Position: The antenna\'s coordinates in the 15x15 ft grid or difference pair.']
    )
    cm = confusion_matrix(y_test, y_test_pred, labels=['Q1', 'Q2', 'Q3', 'Q4'])
    pd.DataFrame(cm, index=['Q1', 'Q2', 'Q3', 'Q4'], columns=['Q1', 'Q2', 'Q3', 'Q4']).to_csv(
        'confusion_matrix.csv',
        header=['# Confusion Matrix', '# This file shows the confusion matrix for quadrant predictions.',
                '# - Rows: Actual quadrants (Q1, Q2, Q3, Q4).', '# - Columns: Predicted quadrants (Q1, Q2, Q3, Q4).',
                '# - Values: Number of samples for each actual-predicted pair.']
    )
    joblib.dump(model, 'model_full_features.pkl')
    joblib.dump(scaler, 'scaler_full_features.pkl')
    logger.info("Model and scaler saved to model_full_features.pkl and scaler_full_features.pkl")

if __name__ == '__main__':
    main()