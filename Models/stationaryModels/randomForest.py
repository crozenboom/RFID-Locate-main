import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

######################################################
#----------------- IMPORTS AND SETUP -----------------#
######################################################

# Directory for CSV files
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleTest0'

######################################################
#--------- DATA LOADING AND PER-LOCATION PROCESSING --#
######################################################

# Load all 64 CSV files, each representing a unique tag location
all_files = glob.glob(os.path.join(data_dir, '*.csv'))

data_list = []
for file in all_files:
    # Extract coordinates from filename, e.g., (1,1).csv -> x=1, y=1
    filename = os.path.basename(file)
    coords = filename.replace('.csv', '').strip('()').split(',')
    x_coord, y_coord = float(coords[0]), float(coords[1])
    
    # Read CSV for this location
    df = pd.read_csv(file)
    
    # Add coordinates to the dataframe
    df['x_coord'] = x_coord
    df['y_coord'] = y_coord
    data_list.append(df)

# Combine data, preserving location-specific structure
all_data = pd.concat(data_list, ignore_index=True)

######################################################
#--------------- FEATURE ENGINEERING ----------------#
######################################################

# Pivot data to create features per antenna for each location
#pivot_data = all_data.groupby(['x_coord', 'y_coord', 'antenna']).agg({
#    'rssi': 'mean',
#    'phase_angle': 'mean',
#    'channel_index': 'mean',
#    'doppler_frequency': 'mean'
#}).unstack()

# Alternative: Use only rssi and phase_angle (uncomment to try)
pivot_data = all_data.groupby(['x_coord', 'y_coord', 'antenna']).agg({
     'rssi': 'mean',
     'phase_angle': 'mean'
 }).unstack()

# Flatten column names, e.g., rssi_Ant1, rssi_Ant2, etc.
pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]

# Reset index to make x_coord and y_coord columns
pivot_data = pivot_data.reset_index()

# Add distance features to each antenna
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]
for i, (ax, ay) in enumerate(antennas, 1):
    pivot_data[f'distance_Ant{i}'] = np.sqrt((pivot_data['x_coord'] - ax)**2 + (pivot_data['y_coord'] - ay)**2)

# Handle any missing values (unlikely, as all antennas provide readings)
pivot_data = pivot_data.fillna(pivot_data.mean())

# Print pivot_data for inspection
#print(pivot_data)

######################################################
#----------------- MODEL TRAINING --------------------#
######################################################

# Prepare features (X) and targets (y)
X = pivot_data[[col for col in pivot_data.columns if col not in ['x_coord', 'y_coord']]]
y = pivot_data[['x_coord', 'y_coord']]  # Predict both x and y coordinates

# Scale features to improve model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

#print(X)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameter grid for tuning
param_grid = {
    'n_estimators': [800, 1000, 1500, 2000],
    'max_depth': [30, 35, 40, 45, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 3]
}

# Perform randomized search with cross-validation
grid_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    n_iter=60,  # Test n random combinations
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1  # Use all available CPU cores
)
grid_search.fit(X_train, y_train)

# Get best model and parameters
rf_model = grid_search.best_estimator_
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation MSE: {-grid_search.best_score_:.2f}")

# Evaluate model on test set
y_pred = rf_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
print(f"Test Set MSE (x, y): {mse[0]:.2f}, {mse[1]:.2f}")

######################################################
#------------------- PREDICTION ---------------------#
######################################################

# Predict coordinates for the test set
y_pred = rf_model.predict(X_test)
#print(f"Predicted coordinates for the test set: {y_pred}")

######################################################
#-------------------- PLOTTING -----------------------#
######################################################

# Create scatter plot of actual vs. predicted coordinates for all locations
plt.figure(figsize=(10, 10))  # Larger figure for clarity
plt.scatter(y['x_coord'], y['y_coord'], color='blue', label='Actual', alpha=0.5, s=100)
y_pred_all = rf_model.predict(X)  # Predict for all data
plt.scatter(y_pred_all[:, 0], y_pred_all[:, 1], color='red', label='Predicted', alpha=0.5, s=100)

# Add dotted lines between actual and predicted points
for actual, pred in zip(y.values, y_pred_all):
    plt.plot([actual[0], pred[0]], [actual[1], pred[1]], color='black', linestyle=':', linewidth=0.5)

# Label 10 random points with actual and predicted coordinates
np.random.seed(42)  # For reproducibility
random_indices = np.random.choice(len(y), size=10, replace=False)
for idx in random_indices:
    actual = y.values[idx]
    pred = y_pred_all[idx]
    # Label actual point
    plt.annotate(f'A: ({actual[0]:.1f}, {actual[1]:.1f})', 
                 (actual[0], actual[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=6, color='blue')
    # Label predicted point
    plt.annotate(f'P: ({pred[0]:.1f}, {pred[1]:.1f})', 
                 (pred[0], pred[1]), 
                 xytext=(5, 5), textcoords='offset points', fontsize=6, color='red')

# Plot antenna locations for context
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]
plt.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Actual vs. Predicted RFID Tag Coordinates (All 64 Locations in 8x8 Grid)')
plt.xlim(0, 15)  # Cover tags (1 to 14) and antennas
plt.ylim(0, 15)
plt.grid(True)
plt.legend()
plt.savefig('randomForest.png')
plt.close()

print("Scatter plot for all locations saved as 'randomForest.png'")