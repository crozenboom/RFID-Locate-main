import pandas as pd
import numpy as np

file_path = '/Users/meredithnye/Documents/CSV_files/'

# Load the data
df = pd.read_csv(file_path + 'CalTest1.csv')

# Show first few rows
print(df.head())

def rssi_to_distance(rssi):
    # Reference values for the power level used by the reader. In this case- Max power= 32.5 dBm
    d0 = 36 # 6 inches
    rssi0 = -65 # RSSI at 6 inches at max power
    n = 2 # N can be a value 2-4 which represents the density of the testing space

    # Calculating distance based on RSSI for all tag reads
    d = d0 * 10 ** ((rssi0 - rssi) / (10 * n))
    return d

# Convert RSSI to distance
df['Distance'] = df['RSSI'].apply(rssi_to_distance)

# Keep relevant columns
result = df[['RunTimestamp', 'AntennaPort', 'RSSI', 'Distance', 'PhaseAngle', 'DopplerFrequency']]

# Handle duplicates before saving:
# Remove exact duplicates (same Timestamp, AntennaPort, and RSSI)
result_no_duplicates = result.drop_duplicates(subset=['RunTimestamp', 'AntennaPort', 'RSSI'], keep='first')

# Sort the data by AntennaPort (1, then 2, then 3, then 4)
result_sorted = result_no_duplicates.sort_values(by='AntennaPort', ascending=True)

# Save to new CSV
output_path = file_path + 'CalTest1_NoDuplicates_Sorted.csv'
result_sorted.to_csv(output_path, index=False)

print(f"Processed and sorted CSV saved to {output_path}")
