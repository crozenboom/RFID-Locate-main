import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

# Define quadrant mapping function
def coordinates_to_quadrant(x, y):
    if x < 7.5 and y >= 7.5:
        return 'Q1'  # Top-left
    elif x >= 7.5 and y >= 7.5:
        return 'Q2'  # Top-right
    elif x < 7.5 and y < 7.5:
        return 'Q3'  # Bottom-left
    else:
        return 'Q4'  # Bottom-right

# Interpolate circular path coordinates
def interpolate_circle_path(timestamps, center_x, center_y, radius, direction, num_points=100):
    t = np.linspace(0, 1, num_points)
    timestamps = np.array(timestamps)
    t_normalized = (timestamps - timestamps.min()) / (timestamps.max() - timestamps.min())
    if direction == 'clockwise':
        angles = 2 * np.pi * (1 - t_normalized)
    else:
        angles = 2 * np.pi * t_normalized
    x = center_x + radius * np.cos(angles)
    y = center_y + radius * np.sin(angles)
    return x, y

# Load and preprocess data
def load_and_preprocess_data(metadata_file, data_dir, is_static=False):
    metadata = pd.read_csv(metadata_file)
    data_frames = []
    
    for _, row in metadata.iterrows():
        file_path = os.path.join(data_dir, f"{row['raw_CSV_filename']}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} not found, skipping.")
            continue
        
        df = pd.read_csv(file_path)
        # Pivot data to get one row per timestamp
        df['timestamp'] = df['timestamp'].round(3)  # Round to handle floating-point precision
        pivoted = df.pivot_table(
            index='timestamp',
            columns='antenna',
            values=['rssi', 'phase_angle'],
            aggfunc='mean'
        )
        pivoted.columns = [f'{col[0]}_{col[1]}' for col in pivoted.columns]
        pivoted = pivoted.reset_index()
        
        if is_static:
            # Assume static data has x_true, y_true columns
            # Replace with your actual column names
            if 'x_true' in df.columns and 'y_true' in df.columns:
                pivoted['x_true'] = df['x_true'].iloc[0]
                pivoted['y_true'] = df['y_true'].iloc[0]
                pivoted['quadrant'] = coordinates_to_quadrant(pivoted['x_true'], pivoted['y_true'])
        else:
            # Interpolate (x, y) for dynamic circular paths
            x, y = interpolate_circle_path(pivoted['timestamp'], row['center_x_true'], row['center_y_true'], row['radius_true'], row['direction'])
            pivoted['x_true'] = x
            pivoted['y_true'] = y
            pivoted['quadrant'] = pivoted.apply(lambda r: coordinates_to_quadrant(r['x_true'], r['y_true']), axis=1)
        
        pivoted['filename'] = row['raw_CSV_filename']
        data_frames.append(pivoted)
    
    if not data_frames:
        raise ValueError("No valid data files loaded.")
    
    return pd.concat(data_frames, ignore_index=True)

# Generate synthetic stationary data (replace with actual data if available)
def generate_synthetic_static_data(n_points_per_quadrant=100):
    np.random.seed(42)
    quadrants = ['Q1', 'Q2', 'Q3', 'Q4']
    x_ranges = {'Q1': (0, 7.5), 'Q2': (7.5, 15), 'Q3': (0, 7.5), 'Q4': (7.5, 15)}
    y_ranges = {'Q1': (7.5, 15), 'Q2': (7.5, 15), 'Q3': (0, 7.5), 'Q4': (0, 7.5)}
    data = []
    
    for q in quadrants:
        x = np.random.uniform(x_ranges[q][0], x_ranges[q][1], n_points_per_quadrant)
        y = np.random.uniform(y_ranges[q][0], y_ranges[q][1], n_points_per_quadrant)
        # Simulate RSSI and phase_angle (adjust ranges based on your data)
        for i in range(n_points_per_quadrant):
            row = {
                'timestamp': i,
                'x_true': x[i],
                'y_true': y[i],
                'quadrant': q
            }
            for ant in range(1, 5):
                # Simulate RSSI (-80 to -40 dBm) and phase_angle (0 to 4096)
                row[f'rssi_{ant}'] = np.random.uniform(-80, -40)
                row[f'phase_angle_{ant}'] = np.random.uniform(0, 4096)
            data.append(row)
    
    return pd.DataFrame(data)

# Main processing
def main():
    # File paths (update as needed)
    metadata_file = 'Testing/MovementTesting/CircleTests/CircleMetadata.csv'
    data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests'
    static_data_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8'
    
    # Load data
    print("Loading dynamic data...")
    dynamic_data = load_and_preprocess_data(metadata_file, data_dir, is_static=False)
    
    print("Loading/generating static data...")
    if os.path.exists(static_data_file):
        static_data = load_and_preprocess_data(static_data_file, data_dir, is_static=True)
    else:
        print("Warning: Static data file not found, generating synthetic data.")
        static_data = generate_synthetic_static_data()
    
    # Combine datasets
    data = pd.concat([static_data, dynamic_data], ignore_index=True)
    
    # Features and target
    features = [f'rssi_{i}' for i in range(1, 5)] + [f'phase_angle_{i}' for i in range(1, 5)]
    target = 'quadrant'
    
    # Handle missing values
    data[features] = data[features].fillna(data[features].mean())
    
    # Split data
    X = data[features]
    y = data[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    # Best model
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
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
    
    # Per-file accuracy (dynamic data only)
    dynamic_test = dynamic_data[dynamic_data['timestamp'].isin(X_test.index)]
    if not dynamic_test.empty:
        X_dynamic_test = dynamic_test[features]
        X_dynamic_test_scaled = scaler.transform(X_dynamic_test)
        y_dynamic_test = dynamic_test[target]
        y_dynamic_pred = model.predict(X_dynamic_test_scaled)
        
        for filename in dynamic_test['filename'].unique():
            mask = dynamic_test['filename'] == filename
            file_acc = accuracy_score(y_dynamic_test[mask], y_dynamic_pred[mask])
            file_results = {'Filename': filename, 'Total Accuracy': file_acc}
            for q in quadrants:
                q_mask = (dynamic_test['filename'] == filename) & (y_dynamic_test == q)
                if q_mask.sum() > 0:
                    q_acc = accuracy_score(y_dynamic_test[q_mask], y_dynamic_pred[q_mask])
                    file_results[f'{q} Accuracy'] = q_acc
                else:
                    file_results[f'{q} Accuracy'] = np.nan
            results.append(file_results)
    
    # Save results
    pd.DataFrame(results).to_csv('quadrant_results.csv', index=False)
    
    # Feature importance
    importances = model.feature_importances_
    feature_imp_df = pd.DataFrame({'Feature': features, 'Importance': importances})
    feature_imp_df = feature_imp_df.sort_values('Importance', ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_imp_df)
    plt.title('Feature Importance')
    plt.savefig('feature_importance.png')
    plt.close()
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=quadrants)
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