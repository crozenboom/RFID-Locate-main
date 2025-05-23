import pandas as pd
import numpy as np
import glob
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

######################################################
#----------------- IMPORTS AND SETUP -----------------#
######################################################

# Directory for CSV files
data_dir = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/Test8/'

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
pivot_data = all_data.groupby(['x_coord', 'y_coord', 'antenna']).agg({
    'rssi': 'mean',
    'phase_angle': 'mean',
    'channel_index': 'mean',
    'doppler_frequency': 'mean'
}).unstack()

# Flatten column names, e.g., rssi_Ant1, rssi_Ant2, etc.
pivot_data.columns = [f'{col[0]}_Ant{int(col[1])}' for col in pivot_data.columns]

# Reset index to make x_coord and y_coord columns
pivot_data = pivot_data.reset_index()

# Handle any missing values (unlikely, as all antennas provide readings)
pivot_data = pivot_data.fillna(pivot_data.mean())

print(pivot_data)

######################################################
#----------------- MODEL TRAINING --------------------#
######################################################

# Prepare features (X) and targets (y)
X = pivot_data[[col for col in pivot_data.columns if col not in ['x_coord', 'y_coord']]]
y = pivot_data[['x_coord', 'y_coord']]  # Predict both x and y coordinates

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model with 100 trees
rf_model = RandomForestRegressor(n_estimators=400, random_state=42)
rf_model.fit(X_train, y_train)

######################################################
#------------------- PREDICTION ---------------------#
######################################################

# Predict coordinates for the test set
y_pred = rf_model.predict(X_test)

######################################################
#-------------------- PLOTTING -----------------------#
######################################################

# Create scatter plot of actual vs. predicted coordinates
plt.figure(figsize=(8, 8))  # Square figure for equal scaling
plt.scatter(y_test['x_coord'], y_test['y_coord'], color='blue', label='Actual', alpha=0.5, s=100)
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted', alpha=0.5, s=100)

# Add dotted lines between actual and predicted points
for actual, pred in zip(y_test.values, y_pred):
    plt.plot([actual[0], pred[0]], [actual[1], pred[1]], color='black', linestyle=':', linewidth=1)

# Plot antenna locations for context
antennas = [(0, 0), (0, 15), (15, 15), (15, 0)]
plt.scatter([x for x, y in antennas], [y for x, y in antennas], color='green', marker='^', s=200, label='Antennas')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Actual vs. Predicted RFID Tag Coordinates (13x13 Grid)')
plt.xlim(-1, 16)  # Slightly beyond 0-13 for visibility
plt.ylim(-1, 16)
plt.grid(True)
plt.legend()
plt.savefig('coordinates_plot.png')
plt.close()

print("Scatter plot saved as 'coordinates_plot.png'")