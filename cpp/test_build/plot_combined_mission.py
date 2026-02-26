#!/usr/bin/env python3
"""Plot combined mission simulation results."""

import csv
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


def read_csv(path):
    """Read CSV file into list of dicts."""
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def read_ref_path(path):
    """Read reference path CSV."""
    x, y = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            x.append(float(row['x_qlabs']))
            y.append(float(row['y_qlabs']))
    return x, y


def main():
    csv_path = 'combined_mission.csv'
    if not os.path.exists(csv_path):
        print(f"ERROR: {csv_path} not found")
        sys.exit(1)

    data = read_csv(csv_path)
    if not data:
        print("ERROR: No data in CSV")
        sys.exit(1)

    # Parse fields
    t = [float(r['elapsed_s']) for r in data]
    x_ql = [float(r['x_qlabs']) for r in data]
    y_ql = [float(r['y_qlabs']) for r in data]
    cte = [float(r['cte']) for r in data]
    v = [float(r['v_meas']) for r in data]
    v_cmd = [float(r['v_cmd']) for r in data]
    delta = [float(r['delta_cmd']) * 180.0 / math.pi for r in data]  # to degrees
    heading_err = [float(r['heading_err']) * 180.0 / math.pi for r in data]
    legs = [r['leg'] for r in data]

    # Compute stats
    max_cte = max(cte)
    avg_cte = sum(cte) / len(cte)
    avg_speed = sum(v) / len(v)
    duration = t[-1] - t[0]
    max_heading = max(abs(h) for h in heading_err)

    # Assign colors per leg
    leg_colors = {
        'hub_to_pickup': '#2196F3',
        'pickup_to_dropoff': '#4CAF50',
        'dropoff_to_hub': '#FF9800',
    }

    # Create figure: 2x3 grid
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # -- Panel 1: Path in QLabs frame (large, spans 2 rows on left)
    ax_path = fig.add_subplot(gs[:, 0])
    ax_path.set_title('Combined Mission: Planned vs Simulated Path')

    # Load reference paths
    ref_files = [
        ('combined_ref_hub_to_pickup.csv', 'Hub→Pickup', '#90CAF9'),
        ('combined_ref_pickup_to_dropoff.csv', 'Pickup→Dropoff', '#A5D6A7'),
        ('combined_ref_dropoff_to_hub.csv', 'Dropoff→Hub', '#FFCC80'),
    ]
    for fname, label, color in ref_files:
        if os.path.exists(fname):
            rx, ry = read_ref_path(fname)
            ax_path.plot(rx, ry, color=color, linewidth=1, alpha=0.7, label=f'Planned: {label}')

    # Plot simulated path colored by leg
    prev_leg = None
    seg_x, seg_y = [], []
    for i in range(len(x_ql)):
        if legs[i] != prev_leg and seg_x:
            c = leg_colors.get(prev_leg, 'gray')
            ax_path.plot(seg_x, seg_y, color=c, linewidth=2)
            seg_x, seg_y = [seg_x[-1]], [seg_y[-1]]
        seg_x.append(x_ql[i])
        seg_y.append(y_ql[i])
        prev_leg = legs[i]
    if seg_x:
        c = leg_colors.get(prev_leg, 'gray')
        ax_path.plot(seg_x, seg_y, color=c, linewidth=2)

    # Legend entries for simulated paths
    for name, color in leg_colors.items():
        label = name.replace('_', ' ').replace('to', '→').title()
        ax_path.plot([], [], color=color, linewidth=2, label=f'Tracked: {label}')

    # Mark waypoints
    waypoints = {
        'Hub': (-1.205, -0.83),
        'Pickup': (0.125, 4.395),
        'Dropoff': (-0.905, 0.800),
    }
    for name, (wx, wy) in waypoints.items():
        ax_path.plot(wx, wy, '*', markersize=12, color='magenta', zorder=5)
        ax_path.annotate(name, (wx, wy), textcoords="offset points",
                        xytext=(8, 8), fontsize=9, color='magenta', fontweight='bold')

    # Stats box
    stats_text = (f"Duration: {duration:.1f}s\n"
                  f"Max CTE: {max_cte:.3f}m\n"
                  f"Avg CTE: {avg_cte:.3f}m\n"
                  f"Avg Speed: {avg_speed:.2f} m/s")
    ax_path.text(0.02, 0.98, stats_text, transform=ax_path.transAxes,
                verticalalignment='top', fontsize=9,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax_path.set_xlabel('X (QLabs frame, m)')
    ax_path.set_ylabel('Y (QLabs frame, m)')
    ax_path.legend(fontsize=7, loc='lower right')
    ax_path.set_aspect('equal')
    ax_path.grid(True, alpha=0.3)

    # -- Panel 2: CTE over time
    ax_cte = fig.add_subplot(gs[0, 1])
    ax_cte.set_title(f'Cross-Track Error (max={max_cte:.3f}m, avg={avg_cte:.3f}m)')
    ax_cte.plot(t, cte, 'r-', linewidth=0.8)
    ax_cte.axhline(y=0.15, color='orange', linestyle='--', alpha=0.7, label='Avg target')
    ax_cte.axhline(y=0.30, color='red', linestyle='--', alpha=0.7, label='Max target')
    ax_cte.set_xlabel('Time (s)')
    ax_cte.set_ylabel('CTE (m)')
    ax_cte.legend(fontsize=8)
    ax_cte.grid(True, alpha=0.3)
    ax_cte.set_ylim(bottom=0)

    # -- Panel 3: Speed over time
    ax_speed = fig.add_subplot(gs[0, 2])
    ax_speed.set_title(f'Speed (avg={avg_speed:.2f} m/s)')
    ax_speed.plot(t, v, 'g-', linewidth=0.8, label='v_meas')
    ax_speed.plot(t, v_cmd, 'c--', linewidth=0.5, alpha=0.5, label='v_cmd')
    ax_speed.set_xlabel('Time (s)')
    ax_speed.set_ylabel('Velocity (m/s)')
    ax_speed.legend(fontsize=8)
    ax_speed.grid(True, alpha=0.3)

    # -- Panel 4: Steering angle over time
    ax_steer = fig.add_subplot(gs[1, 1])
    ax_steer.set_title('Steering Angle')
    ax_steer.plot(t, delta, 'm-', linewidth=0.8)
    max_steer_deg = 30.0
    ax_steer.axhline(y=max_steer_deg, color='red', linestyle=':', alpha=0.5)
    ax_steer.axhline(y=-max_steer_deg, color='red', linestyle=':', alpha=0.5)
    ax_steer.set_xlabel('Time (s)')
    ax_steer.set_ylabel('Steering (deg)')
    ax_steer.grid(True, alpha=0.3)

    # -- Panel 5: Heading error over time
    ax_head = fig.add_subplot(gs[1, 2])
    ax_head.set_title(f'Heading Error (max={max_heading:.1f}°)')
    ax_head.plot(t, heading_err, 'b-', linewidth=0.8)
    ax_head.set_xlabel('Time (s)')
    ax_head.set_ylabel('Heading Error (deg)')
    ax_head.grid(True, alpha=0.3)

    fig.suptitle('Combined Mission Simulation Report', fontsize=14, fontweight='bold')

    out_path = 'combined_mission_report.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Report saved: {out_path}")


if __name__ == '__main__':
    main()
