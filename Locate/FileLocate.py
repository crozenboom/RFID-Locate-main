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
    Q: process variance
    R: measurement variance
    """
    def __init__(self, initial_value: float, Q: float = 0.05, R: float = 2.0, initial_P: float = 1.0):
        self.x = float(initial_value)
        self.P = float(initial_P)
        self.Q = float(Q)
        self.R = float(R)

    def update(self, z: float) -> float:
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (float(z) - self.x)
        self.P = (1.0 - K) * self.P
        return self.x

# ====== Kalman tunables ======
KALMAN_Q = 0.05
KALMAN_R = 2.0

# ===================== Constants =====================
RSSI0 = {1: -54.76140196, 2: -54.18993038, 3: -54.24585557, 4: -54.53863142}
D0 = 10.60660172
N_values = {1: 0.982954949, 2: 2.479294337, 3: 2.52394818, 4: 1.633127718}
ANTENNA_POSITIONS = {1: np.array([0.0, 0.0]), 2: np.array([15.0, 0.0]), 3: np.array([15.0, 15.0]), 4: np.array([0.0, 15.0])}

# ===================== IO =====================
if len(sys.argv) > 1:
    input_file = sys.argv[1]
else:
    input_file = '/Users/calebrozenboom/Documents/RFID_Project/RFID-Locate-main/Testing/WalkingTest/CircleTest2.csv'

print(f"Starting processing of file: {input_file}")
df = pd.read_csv(input_file)
print(f"Loaded {len(df)} rows from the CSV file.")

# Ensure correct types
df['antenna'] = pd.to_numeric(df['antenna'], errors='coerce').astype('Int64')
df['rssi'] = pd.to_numeric(df['rssi'], errors='coerce')
df = df.dropna(subset=['antenna','rssi']).copy()
df['antenna'] = df['antenna'].astype(int)

# ===================== Kalman filters per antenna =====================
filters_by_antenna = {}
def get_filter_for_antenna(ant: int, first_value: float) -> Kalman1D:
    if ant not in filters_by_antenna:
        filters_by_antenna[ant] = Kalman1D(initial_value=first_value, Q=KALMAN_Q, R=KALMAN_R)
    return filters_by_antenna[ant]

# ===================== Row-by-row processing =====================
outputs = []

for idx, row in df.iterrows():
    ant = int(row['antenna'])
    z = float(row['rssi'])

    # Update Kalman filter
    kf = get_filter_for_antenna(ant, z)
    z_filt = kf.update(z)

    # Latest filtered RSSI for all antennas
    avg_rssi = {a: filters_by_antenna[a].x for a in filters_by_antenna.keys()}
    avail_ants = [a for a in avg_rssi if a in ANTENNA_POSITIONS]

    group_num = idx + 1
    print(f"\nProcessing row {group_num}")
    print(f"Filtered RSSI values: {avg_rssi}")
    print(f"Available antennas: {avail_ants}")

    if len(avail_ants) < 3:
        outputs.append({'Group': group_num, 'X': None, 'Y': None, 'Status': 'not enough info to calculate'})
        continue

    # Compute distances
    positions = []
    distances = []
    for a in avail_ants:
        rssi_val = avg_rssi[a]
        N = N_values[a]
        dist = D0 * (10 ** ((RSSI0[a] - rssi_val)/(10*N)))
        positions.append(ANTENNA_POSITIONS[a])
        distances.append(dist)
        print(f"Antenna {a}: filtered RSSI={rssi_val:.2f}, N={N:.6f}, Distance={dist:.2f}")

    positions = np.array(positions)
    distances = np.array(distances)
    initial_guess = np.mean(positions, axis=0)

    def residual(xy):
        return np.linalg.norm(positions - xy, axis=1) - distances

    result = least_squares(residual, initial_guess)
    if result.success:
        x, y = result.x
        outputs.append({'Group': group_num, 'X': x, 'Y': y, 'Status': 'OK'})
        print(f"Predicted location: X={x:.2f}, Y={y:.2f}")
    else:
        outputs.append({'Group': group_num, 'X': None, 'Y': None, 'Status': 'failed to converge'})

# ===================== Save CSV =====================
output_df = pd.DataFrame(outputs)
output_df.to_csv('Coordinate Prediction.csv', index=False)
print("\nProcessing complete. Output saved to 'Coordinate Prediction.csv'.")

# ===================== Plotting =====================
valid_points = [(d['X'], d['Y']) for d in outputs if d['Status']=='OK' and d['X'] is not None and d['Y'] is not None]
if valid_points:
    # Static plot
    fig_static, ax_static = plt.subplots()
    ax_static.set_xlim(0,15)
    ax_static.set_ylim(0,15)
    ax_static.set_title('Predicted Locations')
    ax_static.set_xlabel('X (feet)')
    ax_static.set_ylabel('Y (feet)')
    for ant, pos in ANTENNA_POSITIONS.items():
        ax_static.plot(pos[0], pos[1], 'ro', label=f'Ant {ant}')
    ax_static.legend()
    xs, ys = zip(*valid_points)
    ax_static.plot(xs, ys, 'b-', marker='o', label='Path')
    plt.savefig('static_plot.png')
    print("Static plot saved to 'static_plot.png'.")

    # Animation
    fig_anim, ax_anim = plt.subplots()
    ax_anim.set_xlim(0,15)
    ax_anim.set_ylim(0,15)
    ax_anim.set_title('Animated Predicted Locations')
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
        xs_i, ys_i = zip(*valid_points[:i+1])
        line.set_data(xs_i, ys_i)
        return line,

    ani = FuncAnimation(fig_anim, animate, init_func=init, frames=len(valid_points), interval=200, blit=True)
    ani.save('animation.gif', writer='pillow')
    print("Animation saved to 'animation.gif'.")
else:
    print("No valid points to plot.")