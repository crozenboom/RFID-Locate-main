import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import matplotlib.cm as cm  # Ensure correct import for colormaps
import os

# Suppress pandas FutureWarning
pd.set_option('future.no_silent_downcasting', True)

# Define file and directory paths
metadata_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/LineMetadata.csv'
static_data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8'
dynamic_data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/LineTests/MoveTest2'
script_dir = os.path.dirname(os.path.abspath(__file__))
plot_save_path = os.path.join(script_dir, 'quadrant_predictions_path3_gradient.png')
results_save_path = os.path.join(script_dir, 'quadrant_results.csv')

# Load metadata
try:
    metadata = pd.read_csv(metadata_file)
    metadata.columns = metadata.columns.str.strip()
    print("Metadata columns:", metadata.columns.tolist())
    metadata = metadata[metadata['raw_CSV_filename'].str.startswith('movetest2')]
except FileNotFoundError:
    print(f"Error: Metadata file {metadata_file} not found.")
    exit(1)
except Exception as e:
    print(f"Error reading metadata file: {e}")
    exit(1)

# Check metadata columns
required_cols = ['raw_CSV_filename', 'start_x_true', 'start_y_true', 'end_x_true', 'end_y_true', 'path_#']
missing_cols = [col for col in required_cols if col not in metadata.columns]
if missing_cols:
    print(f"Error: Missing metadata columns: {missing_cols}")
    exit(1)

# Function to assign quadrant based on coordinates
def get_quadrant(x, y):
    try:
        x, y = float(x), float(y)
        buffer = 0.5
        if abs(x - 7.5) < buffer or abs(y - 7.5) < buffer:
            return None  # Skip boundary points
        if x <= 7.5 and y <= 7.5:
            return 1  # Lower-left (Q1)
        elif x <= 7.5 and y > 7.5:
            return 2  # Upper-left (Q2)
        elif x > 7.5 and y > 7.5:
            return 3  # Upper-right (Q3)
        else:
            return 4  # Bottom-right (Q4)
    except (ValueError, TypeError) as e:
        print(f"Error in get_quadrant: {e}, x={x}, y={y}")
        return None

# Function to parse coordinates from filename
def parse_coordinates(filename):
    try:
        coords = filename.strip('().csv').split(',')
        x, y = float(coords[0]), float(coords[1])
        return x, y
    except (ValueError, IndexError) as e:
        print(f"Error parsing coordinates from {filename}: {e}")
        return None, None

# Function to add derived features
def add_derived_features(df, features):
    for i in range(1, 5):
        df[f'rssi_var_{i}'] = df[f'rssi_{i}'].rolling(window=5, min_periods=1).var()
        df[f'phase_diff_{i}'] = df[f'phase_{i}'].diff()
        df[f'doppler_diff_{i}'] = df[f'doppler_{i}'].diff()
        features.extend([f'rssi_var_{i}', f'phase_diff_{i}', f'doppler_diff_{i}'])
    return df, features

# Function to pivot data for features
def pivot_data(df, filename="unknown"):
    try:
        grouped = df.groupby(['epc', 'timestamp']).agg({
            'rssi': list,
            'phase_angle': list,
            'doppler_frequency': list,
            'antenna': list
        }).reset_index()
        
        features = ['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4', 
                    'phase_1', 'phase_2', 'phase_3', 'phase_4',
                    'doppler_1', 'doppler_2', 'doppler_3', 'doppler_4']
        feature_df = pd.DataFrame(index=grouped.index, columns=features)
        
        # Debugging: Count unique antennas per group and log antenna numbers
        antenna_counts = grouped['antenna'].apply(lambda x: len(set(x))).value_counts()
        antenna_numbers = df['antenna'].value_counts()
        print(f"Antenna counts (unique antennas per group) for {filename}:")
        print(antenna_counts)
        print(f"Antenna numbers for {filename}:")
        print(antenna_numbers)
        
        for idx, row in grouped.iterrows():
            antennas = row['antenna']
            rssi_values = row['rssi']
            phase_values = row['phase_angle']
            doppler_values = row['doppler_frequency']
            for ant, rssi, phase, doppler in zip(antennas, rssi_values, phase_values, doppler_values):
                ant = int(ant)
                if ant in [1, 2, 3, 4]:
                    feature_df.loc[idx, f'rssi_{ant}'] = rssi
                    feature_df.loc[idx, f'phase_{ant}'] = phase
                    feature_df.loc[idx, f'doppler_{ant}'] = doppler
        
        feature_df['epc'] = grouped['epc']
        feature_df['timestamp'] = grouped['timestamp']
        
        # Smooth features
        for col in features:
            feature_df[col] = pd.to_numeric(feature_df[col], errors='coerce')
            feature_df[col] = feature_df[col].rolling(window=5, min_periods=1).median()
        
        # Add derived features
        feature_df, features = add_derived_features(feature_df, features)
        
        return feature_df, features
    except Exception as e:
        print(f"Error in pivot_data for {filename}: {e}")
        return pd.DataFrame(), features

# Load static data
static_data = []
all_features = ['rssi_1', 'rssi_2', 'rssi_3', 'rssi_4', 
                'phase_1', 'phase_2', 'phase_3', 'phase_4',
                'doppler_1', 'doppler_2', 'doppler_3', 'doppler_4']
for filename in os.listdir(static_data_dir):
    if filename.endswith('.csv'):
        try:
            x, y = parse_coordinates(filename)
            if x is None or y is None:
                continue
            df = pd.read_csv(os.path.join(static_data_dir, filename))
            df_pivoted, features = pivot_data(df, filename)
            if df_pivoted.empty:
                print(f"Warning: Empty pivoted data for {filename}")
                continue
            df_pivoted['x'] = x
            df_pivoted['y'] = y
            df_pivoted['quadrant'] = df_pivoted.apply(lambda row: get_quadrant(row['x'], row['y']), axis=1)
            static_data.append(df_pivoted)
            all_features = features  # Update features list
        except Exception as e:
            print(f"Error processing static file {filename}: {e}")
            continue

if not static_data:
    print("Error: No valid static data files loaded.")
    exit(1)

static_data = pd.concat(static_data, ignore_index=True)

# Features and target for static data
try:
    print("\nMissing values in static data before imputation:")
    print(static_data[all_features].isna().sum())
    
    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    static_data[all_features] = imputer.fit_transform(static_data[all_features])
    
    print("\nMissing values in static data after imputation:")
    print(static_data[all_features].isna().sum())
    
    if static_data[all_features].isna().any().any():
        print("Error: Some features still contain NaN after imputation.")
        exit(1)
    
    X_static = static_data[all_features]
    y_static = static_data['quadrant']
except KeyError as e:
    print(f"Error: Missing feature columns in static data: {e}")
    exit(1)

# Print static data quadrant distribution
print("\nStatic data quadrant distribution:")
print(static_data['quadrant'].value_counts().sort_index())

# Scale features
scaler = StandardScaler()
X_static_scaled = scaler.fit_transform(X_static)

# Train Random Forest model with grid search
param_grid = {
    'n_estimators': [200, 300, 400],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}
rf = RandomForestClassifier(random_state=42, class_weight={1: 1.5, 2: 1.0, 3: 1.5, 4: 1.0})
grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_static_scaled, y_static)
rf_model = grid_search.best_estimator_
print("Best parameters:", grid_search.best_params_)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': all_features,
    'Importance': rf_model.feature_importances_
})
print("\nFeature Importance:")
print(feature_importance.sort_values(by='Importance', ascending=False))

# Load and process dynamic data
dynamic_data = []
true_paths = []
all_true_quadrants = []
all_pred_quadrants = []
path3_data = []
results = []
low_accuracy_files = []

for idx, row in metadata.iterrows():
    filename = f"{row['raw_CSV_filename']}.csv"
    filepath = os.path.join(dynamic_data_dir, filename)
    if not os.path.exists(filepath):
        print(f"Error: File {filename} not found in {dynamic_data_dir}")
        continue
    try:
        if not all(col in row and pd.notna(row[col]) for col in required_cols[:5]):
            print(f"Error: Missing or invalid metadata for {filename}")
            continue
        df = pd.read_csv(filepath)
        df_pivoted, _ = pivot_data(df, filename)
        if df_pivoted.empty:
            print(f"Error: Empty pivoted data for {filename}")
            continue
        
        print(f"\nMissing values in {filename} before imputation:")
        print(df_pivoted[all_features].isna().sum())
        df_pivoted[all_features] = imputer.transform(df_pivoted[all_features])
        print(f"Missing values in {filename} after imputation:")
        print(df_pivoted[all_features].isna().sum())
        
        if df_pivoted[all_features].isna().any().any():
            print(f"Warning: Some features in {filename} still contain NaN after imputation.")
            continue
        
        X_dynamic = df_pivoted[all_features]
        X_dynamic_scaled = scaler.transform(X_dynamic)
        
        # Predict quadrants
        y_pred = rf_model.predict(X_dynamic_scaled)
        df_pivoted['predicted_quadrant'] = y_pred
        
        # True path
        n_points = len(df_pivoted)
        true_x = np.linspace(row['start_x_true'], row['end_x_true'], n_points)
        true_y = np.linspace(row['start_y_true'], row['end_y_true'], n_points)
        
        if row['path_#'] == 3:
            path3_data.append({
                'true_x': true_x,
                'true_y': true_y,
                'pred_quadrants': y_pred,
                'label': row['raw_CSV_filename']
            })
        
        # Per-file accuracy
        true_quadrants = [get_quadrant(row['start_x_true'] + t*(row['end_x_true']-row['start_x_true']), 
                                       row['start_y_true'] + t*(row['end_y_true']-row['start_y_true'])) 
                          for t in np.linspace(0, 1, n_points)]
        valid_mask = [q is not None for q in true_quadrants]
        true_quadrants_valid = [q for q in true_quadrants if q is not None]
        y_pred_valid = y_pred[valid_mask]
        
        if true_quadrants_valid:
            accuracy = accuracy_score(true_quadrants_valid, y_pred_valid)
            print(f"Accuracy for {filename}: {accuracy:.4f}")
            results.append({'filename': filename, 'accuracy': accuracy})
            if accuracy < 0.5:
                low_accuracy_files.append((filename, accuracy, row['path_#']))
        
        all_true_quadrants.extend(true_quadrants_valid)
        all_pred_quadrants.extend(y_pred_valid.tolist())
        dynamic_data.append(df_pivoted)
    except Exception as e:
        print(f"Error processing dynamic file {filename}: {e}")
        continue

# Log low-accuracy files
if low_accuracy_files:
    print("\nLow-accuracy files (accuracy < 0.5):")
    for file, acc, path in low_accuracy_files:
        print(f"{file}: {acc:.4f} (Path #{path})")

# Calculate total accuracy and per-quadrant accuracy
if all_true_quadrants:
    total_accuracy = accuracy_score(all_true_quadrants, all_pred_quadrants)
    print(f"\nTotal Accuracy Across All Dynamic Files: {total_accuracy:.4f}")
    
    cm = confusion_matrix(all_true_quadrants, all_pred_quadrants, labels=[1, 2, 3, 4])
    per_quadrant_accuracy = cm.diagonal() / cm.sum(axis=1)
    quadrant_results = {
        'Quadrant': ['Q1 (Lower-Left)', 'Q2 (Upper-Left)', 'Q3 (Upper-Right)', 'Q4 (Bottom-Right)'],
        'Accuracy (%)': [acc * 100 for acc in per_quadrant_accuracy]
    }
    print("\nPer-Quadrant Accuracy:")
    print(pd.DataFrame(quadrant_results))
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    quadrant_df = pd.DataFrame(quadrant_results)
    results_df['total_accuracy'] = total_accuracy
    combined_results = pd.concat([
        results_df,
        quadrant_df.reset_index(drop=True)
    ], axis=0)
    combined_results.to_csv(results_save_path, index=False)
    print(f"Results saved to: {results_save_path}")
else:
    print("\nNo predictions made; cannot compute total accuracy.")

# Print dynamic data quadrant distribution
if all_true_quadrants:
    print("\nDynamic data true quadrant distribution:")
    print(pd.Series(all_true_quadrants).value_counts().sort_index())

# Plot true paths and predicted quadrants for path_# == 3 with color gradient
if path3_data:
    plt.figure(figsize=(10, 10))
    plt.grid(True)
    plt.xlim(0, 15)
    plt.ylim(0, 15)
    plt.axhline(y=7.5, color='k', linestyle='--', alpha=0.5)
    plt.axvline(x=7.5, color='k', linestyle='--', alpha=0.5)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('True Paths and Predicted Quadrants with Gradient (Path #3)')

    # Define distinct colormaps for each path
    colormaps = [cm.Blues, cm.Reds]  # Different colormaps for the two paths
    path_labels = set()  # To avoid duplicate legend entries

    # Plot true paths and predicted quadrants with gradient
    for idx, data in enumerate(path3_data):
        true_x = data['true_x']
        true_y = data['true_y']
        pred_quadrants = data['pred_quadrants']
        label = data['label']
        
        # Plot true path with color gradient
        points = np.array([true_x, true_y]).T
        segments = np.concatenate([points[:-1, None], points[1:, None]], axis=1)
        norm = plt.Normalize(0, len(true_x) - 1)
        lc = plt.cm.ScalarMappable(norm=norm, cmap=colormaps[idx % len(colormaps)])
        line = plt.plot(true_x, true_y, label=f'True Path {label}' if label not in path_labels else "", 
                        linewidth=2, alpha=0.75, zorder=1)
        path_labels.add(label)
        
        # Add gradient to the line
        for i in range(len(segments)):
            plt.plot(segments[i, :, 0], segments[i, :, 1], c=lc.to_rgba(i), linewidth=2, alpha=0.75, zorder=1)
        
        # Plot predicted quadrants as scatter points with same gradient
        colors = {1: 'red', 2: 'blue', 3: 'green', 4: 'purple'}
        for quad in [1, 2, 3, 4]:
            mask = pred_quadrants == quad
            if mask.any():
                scatter_x = true_x[mask]
                scatter_y = true_y[mask]
                indices = np.arange(len(true_x))[mask]
                plt.scatter(scatter_x, scatter_y, c=lc.to_rgba(indices), marker='x', s=100, alpha=0.6,
                           label=f'Predicted Q{quad} ({label})' if quad == 1 and label not in path_labels else "", zorder=2)
                path_labels.add(label)

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(plot_save_path, bbox_inches='tight')
    print(f"Plot saved to: {plot_save_path}")
    plt.show()
else:
    print("Warning: No data to plot for Path #3.")