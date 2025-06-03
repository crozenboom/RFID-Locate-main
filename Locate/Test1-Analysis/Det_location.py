import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

# Load the CSV
file_path = "/Users/meredithnye/Documents/Test1/AntennaAverages.csv"
df = pd.read_csv(file_path)

# Pivot data so each test is a row and columns are antenna distances
pivot_df = df.pivot(index="Test", columns="AntennaPort", values="Distance")
pivot_df.columns = pivot_df.columns.astype(str)

# Antenna positions in feet, convert to inches (1 ft = 12 inches)
antennas_ft = {
    1: (0, 3),
    2: (3, 6),
    3: (6, 3),
    4: (3, 0)
}
antennas_in = {k: (x * 12, y * 12) for k, (x, y) in antennas_ft.items()}

# Multilateration function
def multilateration(distances, anchors):
    def residuals(p):
        x, y = p
        return [
            np.sqrt((x - anchors[int(k)][0])**2 + (y - anchors[int(k)][1])**2) - d
            for k, d in distances.items()
        ]
    
    x0 = np.array([3 * 12, 3 * 12])  # initial guess at center (in inches)
    result = least_squares(residuals, x0)
    return result.x

# Plotting
fig, ax = plt.subplots(figsize=(8, 8))
colors = plt.cm.plasma(np.linspace(0, 1, len(pivot_df)))

# Plot antennas
for k, (x, y) in antennas_in.items():
    ax.plot(x, y, 'ro')
    ax.text(x + 2, y + 2, f"Antenna {k}", color='red')

# Estimate and plot tag positions
for i, (test_name, row) in enumerate(pivot_df.iterrows()):
    distances = row.dropna().to_dict()
    est_x, est_y = multilateration(distances, antennas_in)
    ax.plot(est_x, est_y, 'x', color=colors[i], label=test_name)

# Format the plot
ax.set_title("Estimated RFID Tag Positions (inches)")
ax.set_xlabel("X (in)")
ax.set_ylabel("Y (in)")
ax.set_xlim(0, 6 * 12)  # 6 feet in inches
ax.set_ylim(0, 6 * 12)  # 6 feet in inches
ax.set_aspect('equal')

# Set grid to have 1ft x 1ft squares (12 inches)
ax.set_xticks(np.arange(0, 6 * 12 + 12, 12))  # 12-inch spacing for x-axis (1ft)
ax.set_yticks(np.arange(0, 6 * 12 + 12, 12))  # 12-inch spacing for y-axis (1ft)
ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)

# Update the labels to be in feet
ax.set_xticklabels(np.arange(0, 7, 1))  # Convert inches to feet (1ft increments)
ax.set_yticklabels(np.arange(0, 7, 1))  # Convert inches to feet (1ft increments)

# Add a legend and adjust layout
ax.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.25, 1))
plt.tight_layout()
plt.show()

