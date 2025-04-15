import pandas as pd
import numpy as np
import os

file_path = '/Users/meredithnye/Documents/Test1/'

def rssi_to_distance(rssi):
    d0 = 36  # reference distance in inches
    rssi0 = -65  # RSSI at reference distance
    n = 2  # path-loss exponent

    d = d0 * 10 ** ((rssi0 - rssi) / (10 * n))
    return d

# To collect the summary of average distances
summary_data = []

# Loop through Test1-1.csv to Test1-20.csv
for i in range(1, 21):
    input_filename = f'Test1-{i}.csv'
    input_path = os.path.join(file_path, input_filename)

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"{input_filename} not found. Skipping.")
        continue

    print(f"Processing {input_filename}...")

    # Convert RSSI to distance
    df['Distance'] = df['RSSI'].apply(rssi_to_distance)

    # Keep relevant columns
    result = df[['RunTimestamp', 'AntennaPort', 'RSSI', 'Distance', 'PhaseAngle', 'DopplerFrequency']]

    # Remove duplicates
    result_no_duplicates = result.drop_duplicates(subset=['RunTimestamp', 'AntennaPort', 'RSSI'], keep='first')

    # Sort by AntennaPort
    result_sorted = result_no_duplicates.sort_values(by='AntennaPort', ascending=True)

    # Save output
    output_filename = f'1-{i}_Distances_NoDuplicates_Sorted.csv'
    output_path = os.path.join(file_path, output_filename)
    result_sorted.to_csv(output_path, index=False)

    print(f"Saved processed file to {output_path}")

    # Calculate average distance per antenna
    averages = result_sorted.groupby('AntennaPort')['Distance'].mean().reset_index()
    averages['Test'] = f'Test1-{i}'

    # Reorder columns
    averages = averages[['Test', 'AntennaPort', 'Distance']]
    summary_data.append(averages)

# Combine all average distance data into one DataFrame
summary_df = pd.concat(summary_data, ignore_index=True)

# Save to a summary CSV
summary_output_path = os.path.join(file_path, 'AntennaAverages.csv')
summary_df.to_csv(summary_output_path, index=False)

print(f"\n✅ Summary CSV saved to {summary_output_path}")