import sys
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# ===================== Kalman Filter (1D) =====================
class Kalman1D:
    """
    Simple 1D Kalman filter for RSSI smoothing.
    x: state estimate (filtered RSSI)
    P: estimate covariance
    Q: process variance (how fast the 'true' RSSI may drift)
    R: measurement variance (how noisy the measurements are)
    """
    def __init__(self, initial_value: float, Q: float = 0.05, R: float = 2.0, initial_P: float = 1.0):
        self.x = float(initial_value)
        self.P = float(initial_P)
        self.Q = float(Q)
        self.R = float(R)

    def update(self, z: float) -> float:
        # Predict
        self.P = self.P + self.Q
        # Gain
        K = self.P / (self.P + self.R)
        # Correct
        self.x = self.x + K * (float(z) - self.x)
        # Update covariance
        self.P = (1.0 - K) * self.P
        return self.x

# ====== Tunables for Kalman (good starting points; adjust as needed) ======
KALMAN_Q = 0.05   # smaller for static tags, larger if tag moves
KALMAN_R = 2.0    # larger when your RSSI environment is noisier

# ===================== Your existing constants =====================
RSSI0 = {
    1: -54.76140196,
    2: -54.18993038,
    3: -54.24585557,
    4: -54.53863142
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

# ===================== IO =====================
# Get input file from command line or default to specified path
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/WalkingTest/CircleTest2.csv'

print(f"Starting processing of file: {input_file}")

# Read the CSV file
# Expected columns: timestamp,reader,antenna,rssi,epc,phase_angle,channel_index,doppler_frequency
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from the CSV file.")

# Make sure antenna is int-like and rssi is float
if df['antenna'].dtype != np.int64 and df['antenna'].dtype != np.int32:
    df['antenna'] = pd.to_numeric(df['antenna'], errors='coerce').astype('Int64')
df['rssi'] = pd.to_numeric(df['rssi'], errors='coerce')

# Drop rows with missing essentials
df = df.dropna(subset=['antenna', 'rssi']).copy()
df['antenna'] = df['antenna'].astype(int)

# ===================== Kalman filters per antenna =====================
# Persist filters across all chunks so smoothing carries over time
filters_by_antenna = {}

def get_filter_for_antenna(ant: int, first_value: float) -> Kalman1D:
    if ant not in filters_by_antenna:
        filters_by_antenna[ant] = Kalman1D(
            initial_value=first_value,
            Q=KALMAN_Q,
            R=KALMAN_R,
            initial_P=1.0
        )
    return filters_by_antenna[ant]

# ===================== Processing in chunks =====================
chunk_size = 10
outputs = []

for idx in range(0, len(df), chunk_size):
    chunk = df.iloc[idx:idx + chunk_size]

    # Apply Kalman per-row, per-antenna (order-preserving)
    # Collect filtered RSSI values per antenna for this chunk
    filtered_values = {}  # ant -> list of filtered rssi

    for _, row in chunk.iterrows():
        ant = int(row['antenna'])
        z = float(row['rssi'])

        # Initialize filter for antenna if needed using the first measurement
        kf = get_filter_for_antenna(ant, z)
        z_filt = kf.update(z)

        if ant not in filtered_values:
            filtered_values[ant] = []
        filtered_values[ant].append(z_filt)

    # Compute average FILTERED RSSI per antenna for this chunk
    avg_rssi = {ant: float(np.mean(vals)) for ant, vals in filtered_values.items()}

    # Available antennas with positions
    avail_ants = [ant for ant in avg_rssi if ant in ANTENNA_POSITIONS]

    group_num = (idx // chunk_size) + 1

    print(f"\nProcessing group {group_num} (rows {idx+1} to {min(idx+chunk_size, len(df))})")
    print(f"Average FILTERED RSSI values: {avg_rssi}")
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

    # Prepare positions and distances from FILTERED avg RSSI
    positions = []
    distances = []
    for ant in avail_ants:
        rssi = avg_rssi[ant]
        N = N_values[ant]  # antenna-specific path-loss exponent
        dist = D0 * (10 ** ((RSSI0[ant] - rssi) / (10 * N)))
        print(f"Antenna {ant}: filtered RSSI={rssi:.2f}, N={N:.6f}, Calculated distance={dist:.2f}")
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

# ===================== Save output =====================
output_df = pd.DataFrame(outputs)
output_df.to_csv('Coordinate Prediction.csv', index=False)
print("\nProcessing complete. Output saved to 'Coordinate Prediction.csv'.")

# ===================== Plotting =====================
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