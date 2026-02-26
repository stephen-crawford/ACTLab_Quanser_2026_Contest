#!/usr/bin/env python3
"""Plot map-frame startup test results showing deployment-realistic behavior."""

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
    """Read mapframe_startup.csv with trace data and reference path."""
    trace = {'elapsed_s': [], 'x': [], 'y': [], 'theta': [],
             'v_meas': [], 'v_cmd': [], 'delta_cmd': [],
             'cross_track_err': [], 'heading_err': [], 'progress_pct': []}
    ref = {'x': [], 'y': []}
    in_ref = False

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                if 'Reference' in line:
                    in_ref = True
                continue
            if line.startswith('ref_x'):
                in_ref = True
                continue
            if line.startswith('elapsed_s'):
                continue

            parts = line.split(',')
            if in_ref and len(parts) >= 2:
                try:
                    ref['x'].append(float(parts[0]))
                    ref['y'].append(float(parts[1]))
                except ValueError:
                    pass
            elif not in_ref and len(parts) >= 10:
                try:
                    trace['elapsed_s'].append(float(parts[0]))
                    trace['x'].append(float(parts[1]))
                    trace['y'].append(float(parts[2]))
                    trace['theta'].append(float(parts[3]))
                    trace['v_meas'].append(float(parts[4]))
                    trace['v_cmd'].append(float(parts[5]))
                    trace['delta_cmd'].append(float(parts[6]))
                    trace['cross_track_err'].append(float(parts[7]))
                    trace['heading_err'].append(float(parts[8]))
                    trace['progress_pct'].append(float(parts[9]))
                except ValueError:
                    pass

    for k in trace:
        trace[k] = np.array(trace[k])
    for k in ref:
        ref[k] = np.array(ref[k])
    return trace, ref


def main():
    csv_path = 'mapframe_startup.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.isfile(csv_path):
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    trace, ref = read_csv(csv_path)
    if len(trace['x']) == 0:
        print("Error: no trace data")
        sys.exit(1)

    t = trace['elapsed_s']
    cte = np.abs(trace['cross_track_err'])
    herr_deg = trace['heading_err'] * 180 / np.pi
    delta_deg = trace['delta_cmd'] * 180 / np.pi
    max_steer_deg = 0.45 * 180 / np.pi  # 25.8

    max_cte = np.max(cte)
    avg_cte = np.mean(cte)
    max_herr = np.max(np.abs(herr_deg))
    avg_speed = np.mean(trace['v_meas'])
    duration = t[-1] - t[0] if len(t) > 1 else 0

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
    fig.suptitle('Map-Frame Startup Test: Deployment-Realistic Pipeline',
                 fontsize=14, fontweight='bold')

    # -- Panel 1: XY path (large, spans 2 rows)
    ax = fig.add_subplot(gs[:, 0])
    if len(ref['x']) > 0:
        ax.plot(ref['x'], ref['y'], 'b-', linewidth=1, alpha=0.5,
                label='Blended reference path')
    # Color trace by CTE
    sc = ax.scatter(trace['x'], trace['y'], c=cte, cmap='RdYlGn_r',
                    s=10, vmin=0, vmax=max(0.15, max_cte), zorder=5)
    plt.colorbar(sc, ax=ax, shrink=0.6, label='CTE (m)')
    # Mark start/end
    ax.plot(trace['x'][0], trace['y'][0], 'go', markersize=10, zorder=10, label='Start (0,0,0)')
    ax.plot(trace['x'][-1], trace['y'][-1], 'rs', markersize=10, zorder=10, label='End')
    # Stats box
    stats = (f"Duration: {duration:.1f}s\n"
             f"Max CTE: {max_cte:.3f}m\n"
             f"Avg CTE: {avg_cte:.3f}m\n"
             f"Avg Speed: {avg_speed:.2f} m/s\n"
             f"Max |h_err|: {max_herr:.1f}\u00b0")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    ax.set_xlabel('X (map frame, m)')
    ax.set_ylabel('Y (map frame, m)')
    ax.set_title('Vehicle path vs blended reference')
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(True, alpha=0.3)

    # -- Panel 2: CTE over time
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, cte, 'r-', linewidth=0.8)
    ax.axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='Avg target')
    ax.axhline(0.25, color='red', linestyle='--', alpha=0.5, label='Max target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (m)')
    ax.set_title(f'Cross-Track Error (max={max_cte:.3f}m, avg={avg_cte:.3f}m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # -- Panel 3: Velocity
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(t, trace['v_meas'], 'g-', linewidth=0.8, label='v_meas')
    ax.plot(t, trace['v_cmd'], 'c--', linewidth=0.5, alpha=0.5, label='v_cmd')
    ax.axhline(0.45, color='blue', linestyle=':', alpha=0.3, label='v_ref')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Speed (avg={avg_speed:.2f} m/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # -- Panel 4: Steering
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, delta_deg, 'm-', linewidth=0.8)
    ax.axhline(max_steer_deg, color='red', linestyle='--', alpha=0.3)
    ax.axhline(-max_steer_deg, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (deg)')
    ax.set_title('Steering Angle (should NOT saturate)')
    ax.grid(True, alpha=0.3)

    # -- Panel 5: Heading error
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, herr_deg, 'b-', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (deg)')
    ax.set_title(f'Heading Error (max={max_herr:.1f}\u00b0)')
    ax.grid(True, alpha=0.3)

    out_path = 'mapframe_startup_report.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Report saved: {out_path}")


if __name__ == '__main__':
    main()
