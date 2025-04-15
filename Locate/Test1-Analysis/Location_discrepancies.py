import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

def plot_all_actual_locations(actual_locations_in, antennas_in):
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot antennas
    for k, (x, y) in antennas_in.items():
        ax.plot(x, y, '^', color='blue', markersize=10, label="Antenna" if k == 1 else "")
        ax.text(x + 2, y + 2, f"{k}", color='blue', fontsize=9)

    # Plot all actual tag locations
    for tag_number, (x, y) in actual_locations_in.items():
        ax.plot(x, y, 'go')
        ax.text(x + 2, y + 2, f"{tag_number}", color='green', fontsize=9)

    # Set limits and grid
    ax.set_xlim(0, 6 * 12 + 12)
    ax.set_ylim(0, 6 * 12 + 12)
    ax.set_xticks(np.arange(0, 6 * 12 + 24, 12))
    ax.set_yticks(np.arange(0, 6 * 12 + 24, 12))
    ax.set_aspect('equal')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

    ax.set_title("Actual Tag Locations with Antennas")
    ax.set_xlabel("X (inches)")
    ax.set_ylabel("Y (inches)")
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def multilateration(distances, anchors):
    def residuals(p):
        x, y = p
        return [
            np.sqrt((x - anchors[int(k)][0])**2 + (y - anchors[int(k)][1])**2) - d
            for k, d in distances.items()
        ]
    x0 = np.array([3 * 12, 3 * 12])
    result = least_squares(residuals, x0)
    return result.x

def load_data(file_path):
    df = pd.read_csv(file_path)
    pivot_df = df.pivot(index="Test", columns="AntennaPort", values="Distance")
    pivot_df.columns = pivot_df.columns.astype(str)
    return pivot_df

def get_antenna_positions():
    antennas_ft = {
        1: (0, 3),
        2: (3, 6),
        3: (6, 3),
        4: (3, 0)
    }
    return {k: (x * 12, y * 12) for k, (x, y) in antennas_ft.items()}

def get_actual_tag_locations():
    return {
        1: (1, 5),  2: (1, 1),  3: (2, 1),  4: (2, 3),  5: (6, 5),
        6: (5, 6),  7: (5, 3),  8: (5, 1),  9: (4, 4), 10: (4, 2),
        11: (1, 4), 12: (0, 5), 13: (2, 0), 14: (5, 0), 15: (3, 2),
        16: (4, 1), 17: (0, 0), 18: (3, 5), 19: (2, 4), 20: (4, 5)
    }

def convert_actual_locations_to_inches(actual_locations):
    return {k: (x * 12, y * 12) for k, (x, y) in actual_locations.items()}

def plot_results(pivot_df, antennas_in, actual_locations_in):
    tests_per_graph = 5
    num_graphs = (len(pivot_df) + tests_per_graph - 1) // tests_per_graph
    colors = plt.cm.Set1.colors

    for graph_num in range(num_graphs):
        fig, ax = plt.subplots(figsize=(8, 8))

        start_idx = graph_num * tests_per_graph
        end_idx = min((graph_num + 1) * tests_per_graph, len(pivot_df))

        # Antenna markers
        for k, (x, y) in antennas_in.items():
            ax.plot(x, y, '^', color='blue', markersize=10, label=f"Antenna {k}" if k == 1 else "")
            ax.text(x + 2, y + 2, f"{k}", color='blue', fontsize=9)

        # For dynamic axis limits
        all_x = []
        all_y = []

        for i, (test_name, row) in enumerate(pivot_df.iloc[start_idx:end_idx].iterrows()):
            distances = row.dropna().to_dict()
            est_x, est_y = multilateration(distances, antennas_in)
            tag_number = int(test_name.split()[-1]) if test_name.split()[-1].isdigit() else start_idx + i + 1
            actual_x, actual_y = actual_locations_in.get(tag_number, (None, None))

            if actual_x is None or actual_y is None:
                continue

            color = colors[i % len(colors)]

            ax.plot(est_x, est_y, 'x', color=color, label=f"Test {tag_number} Est")
            ax.plot(actual_x, actual_y, 'o', color=color, label=f"Test {tag_number} Actual")
            ax.plot([actual_x, est_x], [actual_y, est_y], color='gray', linewidth=1)

            all_x.extend([est_x, actual_x])
            all_y.extend([est_y, actual_y])

        # Axis limits with padding
        min_x = min(all_x + [0])
        max_x = max(all_x + [6 * 12])
        min_y = min(all_y + [0])
        max_y = max(all_y + [6 * 12])

        padding = 12  # 1ft padding
        ax.set_xlim(min_x - padding, max_x + padding)
        ax.set_ylim(min_y - padding, max_y + padding)

        ax.set_title(f"Estimated vs Actual Tag Positions (Tests {start_idx + 1}-{end_idx})")
        ax.set_xlabel("X (inches)")
        ax.set_ylabel("Y (inches)")
        ax.set_aspect('equal')

        # 1ft x 1ft grid squares
        ax.set_xticks(np.arange(0, max_x + padding * 2, 12))
        ax.set_yticks(np.arange(0, max_y + padding * 2, 12))
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        ax.legend(fontsize='small', loc='upper right', bbox_to_anchor=(1.25, 1))
        plt.tight_layout()
        plt.show()

def main():
    file_path = "/Users/meredithnye/Documents/Test1/AntennaAverages.csv"  # Update this path
    pivot_df = load_data(file_path)
    antennas_in = get_antenna_positions()
    actual_locations = get_actual_tag_locations()
    actual_locations_in = convert_actual_locations_to_inches(actual_locations)

    # First: plot 5-test groups with estimated vs actual
    plot_results(pivot_df, antennas_in, actual_locations_in)

    # Then: plot all actual tag locations together
    plot_all_actual_locations(actual_locations_in, antennas_in)
if __name__ == "__main__":
    main()
