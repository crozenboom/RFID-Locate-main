import sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
RSSI0_PER_CHANNEL = {
    1: -53.44444444,
    2: -54.85,
    3: -53.73684211,
    4: -53.21052632,
    5: -54.9,
    6: -55,
    7: -53.0952381,
    8: -53.40909091,
    9: -57.28571429,
    10: -54,
    11: -56.9047619,
    12: -55.31818182,
    13: -55.15,
    14: -56.5,
    15: -55.47368421,
    16: -56.3,
    17: -57.21052632,
    18: -56.55555556,
    19: -57.21428571,
    20: -54.5,
    21: -53.69230769,
    22: -54,
    23: -53.92857143,
    24: -53.92307692,
    25: -52.21428571,
    26: -51.85714286,
    27: -52.71428571,
    28: -53.13636364,
    29: -53.19047619,
    30: -52.38095238,
    31: -53.27777778,
    32: -54.73684211,
    33: -54.95454545,
    34: -55.4,
    35: -57.35,
    36: -55.88888889,
    37: -55.71428571,
    38: -55.22727273,
    39: -53.76190476,
    40: -54.76190476,
    41: -54.19047619,
    42: -54,
    43: -52.45,
    44: -52.52631579,
    45: -53,
    46: -53.88888889,
    47: -54.45,
    48: -54.85,
    49: -53.1,
    50: -53.05263158
}
D0 = 10.60660172
N_values = {
    1: 0.982954949,
    2: 2.479294337,
    3: 2.52394818,
    4: 1.633127718
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
    chunk = df.iloc[idx:idx + chunk_size].copy()
    
    group_num = (idx // chunk_size) + 1
    
    print(f"\nProcessing group {group_num} (rows {idx+1} to {min(idx+chunk_size, len(df))})")
    
    # Compute RSSI0 and N for each row
    chunk['rssi0'] = chunk['channel_index'].map(RSSI0_PER_CHANNEL)
    chunk['n'] = chunk['antenna'].map(N_values)
    
    # Compute distance for each row
    chunk['dist'] = D0 * (10 ** ((chunk['rssi0'] - chunk['rssi']) / (10 * chunk['n'])))
    
    # Average distance per antenna
    avg_dist = chunk.groupby('antenna')['dist'].mean().to_dict()
    
    # Available antennas with data
    avail_ants = [ant for ant in avg_dist if ant in ANTENNA_POSITIONS]
    
    print(f"Average distances: { {k: v for k, v in avg_dist.items()} }")
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
        dist = avg_dist[ant]
        print(f"Antenna {ant}: Average Distance={dist:.2f}")
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