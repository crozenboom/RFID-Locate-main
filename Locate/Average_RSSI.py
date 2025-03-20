import pandas as pd

# Load data from CSV
file_path = input("file path: ").strip('"').replace("\\", "/")
df = pd.read_csv(file_path, skiprows=2)

# Remove leading/trailing spaces from column names
df.columns = df.columns.str.strip()

# Display columns to check if required ones are present
print("Columns found in CSV:", df.columns.tolist())

# Ensure required columns exist
required_columns = {'EPC', 'Antenna', 'RSSI'}
if not required_columns.issubset(df.columns):
    print("Error: Missing one or more required columns in CSV file.")
else:
    # Filter for a specific EPC
    target_epc = "300833B2DDD9014000000000"
    df_filtered = df[df['EPC'] == target_epc]

    if df_filtered.empty:
        print(f"No data found for EPC: {target_epc}")
    else:
        # Group by 'Antenna' and calculate mean RSSI
        avg_rssi = df_filtered.groupby('Antenna')['RSSI'].mean().reset_index()

        # Print results
        print(f"Average RSSI per antenna for EPC {target_epc}:")
        print(avg_rssi)
