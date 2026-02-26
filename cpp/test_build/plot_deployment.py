#!/usr/bin/env python3
"""Plot deployment test results from CSV files."""
import csv
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def parse_csv(filename):
    """Parse deployment CSV with trace data and reference path."""
    trace = {'step': [], 'x': [], 'y': [], 'cte': [], 'v': [], 'delta': [], 'progress': []}
    ref = {'x': [], 'y': []}
    in_ref = False

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'Reference' in line:
                    in_ref = True
                continue
            if line.startswith('ref_x'):
                in_ref = True
                continue
            if line.startswith('step'):
                continue

            parts = line.split(',')
            if in_ref and len(parts) >= 2:
                try:
                    ref['x'].append(float(parts[0]))
                    ref['y'].append(float(parts[1]))
                except ValueError:
                    pass
            elif not in_ref and len(parts) >= 7:
                try:
                    trace['step'].append(int(parts[0]))
                    trace['x'].append(float(parts[1]))
                    trace['y'].append(float(parts[2]))
                    trace['cte'].append(float(parts[3]))
                    trace['v'].append(float(parts[4]))
                    trace['delta'].append(float(parts[5]))
                    trace['progress'].append(float(parts[6]))
                except ValueError:
                    pass

    return trace, ref

def plot_leg(filename, title, outfile):
    """Create a 2x2 plot for a mission leg."""
    if not os.path.exists(filename):
        print(f"  Skip {filename} (not found)")
        return

    trace, ref = parse_csv(filename)
    if not trace['x']:
        print(f"  Skip {filename} (empty)")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(title, fontsize=14, fontweight='bold')

    # Top-left: XY path
    ax = axes[0, 0]
    ax.plot(ref['x'], ref['y'], 'b-', linewidth=1, alpha=0.5, label='Reference path')
    # Color trace by CTE
    cte = np.array(trace['cte'])
    sc = ax.scatter(trace['x'], trace['y'], c=cte, cmap='RdYlGn_r',
                    s=8, vmin=0, vmax=max(0.3, cte.max()), zorder=5)
    plt.colorbar(sc, ax=ax, label='CTE (m)')
    ax.set_xlabel('X (map frame, m)')
    ax.set_ylabel('Y (map frame, m)')
    ax.set_title('Planned vs Executed Path')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: CTE over time
    ax = axes[0, 1]
    ax.plot(trace['step'], trace['cte'], 'r-', linewidth=0.8)
    ax.axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='Avg target')
    ax.axhline(0.30, color='red', linestyle='--', alpha=0.5, label='Max target')
    ax.set_xlabel('Step')
    ax.set_ylabel('CTE (m)')
    ax.set_title(f'Cross-Track Error (max={cte.max():.3f}m, avg={cte.mean():.3f}m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-left: Velocity
    ax = axes[1, 0]
    ax.plot(trace['step'], trace['v'], 'g-', linewidth=0.8)
    ax.axhline(0.45, color='blue', linestyle='--', alpha=0.5, label='v_ref')
    ax.set_xlabel('Step')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Speed (avg={np.mean(trace["v"]):.2f} m/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Steering angle
    ax = axes[1, 1]
    delta_deg = np.array(trace['delta']) * 180 / np.pi
    ax.plot(trace['step'], delta_deg, 'm-', linewidth=0.8)
    max_steer = 0.45 * 180 / np.pi  # 25.8°
    ax.axhline(max_steer, color='red', linestyle='--', alpha=0.3)
    ax.axhline(-max_steer, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Step')
    ax.set_ylabel('Steering (deg)')
    ax.set_title('Steering Angle')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"  Saved {outfile}")

def main():
    legs = [
        ('deployment_hub_to_pickup.csv', 'Hub → Pickup', 'deployment_hub_to_pickup.png'),
        ('deployment_pickup_to_dropoff.csv', 'Pickup → Dropoff', 'deployment_pickup_to_dropoff.png'),
        ('deployment_dropoff_to_hub.csv', 'Dropoff → Hub', 'deployment_dropoff_to_hub.png'),
    ]

    for csv_file, title, png_file in legs:
        plot_leg(csv_file, title, png_file)

if __name__ == '__main__':
    main()
