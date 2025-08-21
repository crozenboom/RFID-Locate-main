import sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import math

# Constants
C = 3e8 * 3.28084  # speed of light in feet/s ≈ 984251968.5 ft/s
ANTENNA_POSITIONS = {
    1: np.array([0.0, 0.0]),
    2: np.array([15.0, 0.0]),
    3: np.array([15.0, 15.0]),
    4: np.array([0.0, 15.0])
}

def get_frequency(channel_index):
    return (902.75 + (channel_index - 1) * 0.5) * 1e6  # Hz

# Input CSV
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/MovementTesting/CircleTests/CircleTest3/CircleTest3-1.csv'

print(f"Processing file: {input_file}")

df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows.")

# Chunk size
chunk_size = 10

# List to hold outputs
outputs = []

group_num = 0

# Process in chunks of 10 rows
for idx in range(0, len(df), chunk_size):
    chunk = df.iloc[idx:idx + chunk_size]
    group_num += 1
    positions = []
    distances = []
    print(f"\nProcessing Group {group_num} with {len(chunk)} reads")
    
    for ant in [1, 2, 3, 4]:
        ant_df = chunk[chunk['antenna'] == ant]
        if len(ant_df) < 2:  # Reduced to 2 for unwrapping
            print(f"Antenna {ant} skipped: less than 2 reads")
            continue
        
        # Get frequencies and phases
        freqs = [get_frequency(ch) for ch in ant_df['channel_index']]
        phases = (ant_df['phase_angle'] / 4096.0) * 2 * math.pi
        unwrapped = np.unwrap(phases)
        
        # Sort by frequency
        sorted_idx = np.argsort(freqs)
        freqs = np.array(freqs)[sorted_idx]
        unwrapped = unwrapped[sorted_idx]
        
        unique_freqs = len(set(freqs))
        if unique_freqs < 2:
            print(f"Antenna {ant} skipped: less than 2 unique frequencies")
            continue
        
        # Linear fit: unwrapped phase vs frequency
        slope, intercept = np.polyfit(freqs, unwrapped, 1)
        
        # Distance d = -slope * C / (4 * pi)
        dist = -slope * C / (4 * math.pi)
        
        if dist < 0 or dist > 50:  # Filter invalid distances
            print(f"Antenna {ant} skipped: invalid distance {dist:.2f} ft")
            continue
        
        print(f"Antenna {ant}: Distance={dist:.2f} ft")
        positions.append(ANTENNA_POSITIONS[ant])
        distances.append(dist)
    
    if len(positions) < 3:
        print("Not enough antennas with sufficient reads. Skipping this group.")
        outputs.append({
            'Group': group_num,
            'X': None,
            'Y': None,
            'Status': 'not enough data'
        })
        continue
    
    positions = np.array(positions)
    distances = np.array(distances)
    initial_guess = np.mean(positions, axis=0)
    
    # Residual function
    def residual(xy):
        return np.linalg.norm(positions - xy, axis=1) - distances
    
    # Least squares trilateration with bounds
    bounds = ([0, 0], [15, 15])
    result = least_squares(residual, initial_guess, bounds=bounds)
    
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
        print("Failed to converge")
        outputs.append({
            'Group': group_num,
            'X': None,
            'Y': None,
            'Status': 'failed'
        })

# Save CSV
output_df = pd.DataFrame(outputs)
output_df.to_csv('maybelocation.csv', index=False)
print("\nOutput saved to 'maybelocation.csv'.")

# Plotting
valid_points = [(d['X'], d['Y']) for d in outputs if d['Status'] == 'OK' and d['X'] is not None and d['Y'] is not None]

if valid_points:
    fig_static, ax_static = plt.subplots()
    ax_static.set_xlim(0, 15)
    ax_static.set_ylim(0, 15)
    ax_static.set_title('Predicted RFID Tag Locations')
    ax_static.set_xlabel('X (feet)')
    ax_static.set_ylabel('Y (feet)')
    
    for ant, pos in ANTENNA_POSITIONS.items():
        ax_static.plot(pos[0], pos[1], 'ro', label=f'Ant {ant}')
    ax_static.legend()
    
    xs, ys = zip(*valid_points)
    ax_static.plot(xs, ys, 'b-', marker='o', label='Path')
    plt.savefig('static_plot.png')
    
    # Animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_xlim(0, 15)
    ax_anim.set_ylim(0, 15)
    ax_anim.set_title('Animated Predicted RFID Tag Locations')
    ax_anim.set_xlabel('X (feet)')
    ax_anim.set_ylabel('Y (feet)')
    
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
else:
    print("No valid points to plot.")