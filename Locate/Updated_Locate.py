import pandas as pd
import numpy as np

# Calculate the distance using the RSSI formula
def calculate_distance(d0, RSSI0, RSSI, n):
    return d0 * 10 ** ((RSSI0 - RSSI) / (10 * n))

# Main function
def main():
    # Load reference RSSI data from CSV
    reference_file = r"C:\Users\caleb\Documents\RFID_Location\Calibration.csv"
    ref_df = pd.read_csv(reference_file)
    ref_df.columns = ref_df.columns.str.strip()
    
    # User input for power level and path loss exponent
    power_level = float(input("Enter power level (dBm): "))
    n = float(input("Enter path loss exponent (n): "))
    default_EPC = "300833B2DDD9014000000000"
    EPC_input = input(f"Enter EPC (leave blank to use default {default_EPC}): ").strip()
    EPC = EPC_input if EPC_input else default_EPC
    
    # Extract reference distance and RSSI based on power level
    if power_level in ref_df["Power (dBm)"].values:
        row = ref_df[ref_df["Power (dBm)"] == power_level].iloc[0]
        d0 = row["Distance (in)"]
        RSSI0 = row["Avg RSSI (dBm)"]
    else:
        print("Error: Specified power level not found in reference file.")
        return
    
    # Load RFID data from CSV
    file_path = r"C:\Users\caleb\Documents\RFID_Location\data_2025-03-10_13-59-48.csv"
    df = pd.read_csv(file_path, skiprows=2)
    df.columns = df.columns.str.strip()
    
    #print("Columns found in CSV:", df.columns.tolist())
    
    # If EPC filter is provided, apply it
    if EPC_input:
        df = df[df['EPC'] == EPC_input]
    
    # Ensure required columns exist
    required_columns = {"Antenna", "RSSI"}
    if not required_columns.issubset(df.columns):
        print("Error: CSV file is missing required columns.")
        print("Columns found:", df.columns.tolist())
        return
    
    # Average RSSI per antenna
    avg_rssi = df.groupby("Antenna")["RSSI"].mean().reset_index()
    
    # Calculate the distance for each antenna
    distances = []
    for _, row in avg_rssi.iterrows():
        antenna = row["Antenna"]
        RSSI_avg = row["RSSI"]
        distance = calculate_distance(d0, RSSI0, RSSI_avg, n)
        distances.append(distance)
    
    # Print distances from each antenna
    for i, (antenna, distance) in enumerate(zip(avg_rssi["Antenna"], distances)):
        print(f"Antenna {antenna}: Distance = {distance:.2f} inches")
    
if __name__ == "__main__":
    main()
