import pandas as pd
import numpy as np

# Constants
SPEED_OF_LIGHT = 3 * 10**8  # Speed of light in meters per second (300,000,000 m/s)
SPEED_FACTOR = 0.01  # Assume the signal travels at 1% of the speed of light in the medium (adjust this factor)
METERS_TO_INCHES = 39.3701  # Conversion factor from meters to inches
TIME_SCALING_FACTOR = 1000  # Factor to scale time difference for more reasonable distances

# Function to calculate distance using Time of Flight
def calculate_distance(ToF):
    return (SPEED_OF_LIGHT * SPEED_FACTOR * ToF) / 2  # Adjust for the signal's slower speed

# Main function
def main():
    # Correct the file path and load the RFID data
    file_path = r"C:\Users\caleb\Documents\RFID_Location\data_2025-03-10_14-51-55.csv"
    
    # Define the correct column names
    column_names = ['Timestamp', 'EPC', 'TID', 'Antenna', 'RSSI', 'Frequency', 'Hostname', 'PhaseAngle', 'DopplerFrequency', 'CRHandle']
    
    # Load the CSV, skip comment lines, and use the correct header row
    df = pd.read_csv(file_path, skiprows=6, names=column_names)  # Skip the first 6 lines of comments
    
    # Strip leading/trailing spaces in column names
    df.columns = df.columns.str.strip()
    
    # Check the columns and first few rows of the dataframe
    print("Columns in the data:", df.columns)
    print("First few rows of the data:\n", df.head())
    
    # Ensure timestamp is in datetime format
    if 'Timestamp' not in df.columns:
        print("Error: 'Timestamp' column not found in the CSV.")
        return
    
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    # Get the unique antennas
    antennas = df['Antenna'].unique()
    
    # Dictionary to store distance calculations for each antenna
    distances = {}
    
    for antenna in antennas:
        # Filter data for each antenna
        antenna_data = df[df['Antenna'] == antenna]
        
        # Calculate Time of Flight (ToF) between consecutive readings
        for i in range(1, len(antenna_data)):
            time_diff = (antenna_data['Timestamp'].iloc[i] - antenna_data['Timestamp'].iloc[i - 1]).total_seconds()
            
            if time_diff > 0.00001:  # Ensure the time difference is reasonable
                # Scale the time difference
                time_diff_scaled = time_diff * TIME_SCALING_FACTOR
                distance = calculate_distance(time_diff_scaled)
                # Convert distance from meters to inches
                distance_in_inches = distance * METERS_TO_INCHES
                # Store the calculated distance for each antenna
                if antenna not in distances:
                    distances[antenna] = []
                distances[antenna].append(distance_in_inches)
    
    # Output only the average distance in inches for each antenna
    for antenna, dist_list in distances.items():
        avg_distance = np.mean(dist_list)
        print(f"Antenna {antenna} Average Distance: {avg_distance:.2f} inches")

if __name__ == "__main__":
    main()
