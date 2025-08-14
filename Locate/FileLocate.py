import sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
RSSI0 = {
    1: -50.877323,
    2: -54.708955,
    3: -56.277154,
    4: -57.565217
}
D0 = 10.60660172
N_values = {
    1: 1.959001199,
    2: 2.348866427,
    3: 2.013494858,
    4: 0.87256434
}
ANTENNA_POSITIONS = {
    1: np.array([0.0, 0.0]),
    2: np.array([15.0, 0.0]),
    3: np.array([15.0, 15.0]),
    4: np.array([0.0, 15.0])
}

# Get input file from command line or default to specified path
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/WalkingTest/CircleTest2.csv'

print(f"Starting processing of file: {input_file}")

# Read the CSV file
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from the CSV file.")

# Chunk size
chunk_size = 10

# List to hold output data
outputs = []

# Process the DataFrame in chunks of 10 rows
for idx in range(0, len(df), chunk_size):
    chunk = df.iloc[idx:idx + chunk_size]
    
    # Compute average RSSI per antenna in this chunk
    avg_rssi = chunk.groupby('antenna')['rssi'].mean().to_dict()
    
    # Available antennas with RSSI data
    avail_ants = [ant for ant in avg_rssi if ant in ANTENNA_POSITIONS]
    
    group_num = (idx // chunk_size) + 1
    
    print(f"\nProcessing group {group_num} (rows {idx+1} to {min(idx+chunk_size, len(df))})")
    print(f"Average RSSI values: {avg_rssi}")
    print(f"Available antennas: {avail_ants}")
    
    if len(avail_ants) < 3:
        print("Not enough info to calculate location.")
        outputs.append({
            'Group': group_num,
            'X': None,
            'Y': None,
            'Status': 'not enough info to calculate'
        })
        continue
    
    # Prepare positions and distances
    positions = []
    distances = []
    for ant in avail_ants:
        rssi = avg_rssi[ant]
        N = N_values[ant]  # use the antenna-specific N
        dist = D0 * (10 ** ((RSSI0[ant] - rssi) / (10 * N)))
        print(f"Antenna {ant}: RSSI={rssi:.2f}, N={N:.6f}, Calculated distance={dist:.2f}")
        positions.append(ANTENNA_POSITIONS[ant])
        distances.append(dist)
    
    positions = np.array(positions)
    distances = np.array(distances)
    
    # Initial guess: mean of antenna positions
    initial_guess = np.mean(positions, axis=0)
    print(f"Initial guess for location: {initial_guess}")
    
    # Define residual function
    def residual(xy):
        return np.linalg.norm(positions - xy, axis=1) - distances
    
    # Perform least squares optimization
    result = least_squares(residual, initial_guess)
    
    if result.success:
        x, y = result.x
        print(f"Predicted location: X={x:.2f}, Y={y:.2f}")
        outputs.append({
            'Group': group_num,
            'X': x,
            'Y': y,
            'Status': 'OK'
        })
    else:
        print("Failed to converge on a solution.")
        outputs.append({
            'Group': group_num,
            'X': None,
            'Y': None,
            'Status': 'failed to converge'
        })

# Create output DataFrame and save to CSV
output_df = pd.DataFrame(outputs)
output_df.to_csv('Coordinate Prediction.csv', index=False)
print("\nProcessing complete. Output saved to 'Coordinate Prediction.csv'.")

# Collect valid points for plotting
valid_points = [(d['X'], d['Y']) for d in outputs if d['Status'] == 'OK' and d['X'] is not None and d['Y'] is not None]
if not valid_points:
    print("No valid points to plot.")
else:
    print(f"Plotting {len(valid_points)} valid points.")

    # Static plot
    fig_static, ax_static = plt.subplots()
    ax_static.set_xlim(0, 15)
    ax_static.set_ylim(0, 15)
    ax_static.set_title('Predicted Locations')
    ax_static.set_xlabel('X (feet)')
    ax_static.set_ylabel('Y (feet)')

    # Mark antennas
    for ant, pos in ANTENNA_POSITIONS.items():
        ax_static.plot(pos[0], pos[1], 'ro', label=f'Ant {ant}')
    ax_static.legend()

    # Plot all points connected by lines
    if valid_points:
        xs, ys = zip(*valid_points)
        ax_static.plot(xs, ys, 'b-', marker='o', label='Path')
    plt.savefig('static_plot.png')
    print("Static plot saved to 'static_plot.png'.")

    # Animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_xlim(0, 15)
    ax_anim.set_ylim(0, 15)
    ax_anim.set_title('Animated Predicted Locations')
    ax_anim.set_xlabel('X (feet)')
    ax_anim.set_ylabel('Y (feet)')

    # Mark antennas
    for ant, pos in ANTENNA_POSITIONS.items():
        ax_anim.plot(pos[0], pos[1], 'ro', label=f'Ant {ant}')
    ax_anim.legend()

    line, = ax_anim.plot([], [], 'b-', marker='o', label='Path')

    def init():
        line.set_data([], [])
        return line,

    def animate(i):
        xs, ys = zip(*valid_points[:i+1])
        line.set_data(xs, ys)
        return line,

    ani = FuncAnimation(fig_anim, animate, init_func=init, frames=len(valid_points), interval=200, blit=True)
    ani.save('animation.gif', writer='pillow')
    print("Animation saved to 'animation.gif'.")