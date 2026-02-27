#!/usr/bin/env python3
"""
Generate plots from MPCC test CSV output.

Usage:
    python3 plot_results.py <output_dir>

Reads CSV files from output_dir/ and generates PNG plots in the same directory.
Called automatically by run_tests.sh after test execution.
"""

import sys
import os
import glob

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path):
    """Load CSV with header row into dict of numpy arrays.

    Handles files with mixed row widths by reading only rows matching the header.
    """
    import io
    try:
        with open(path, 'r') as f:
            lines = f.readlines()
        if not lines:
            return None
        header = lines[0].strip()
        n_cols = len(header.split(','))
        valid_lines = [header + '\n']
        for line in lines[1:]:
            if len(line.strip().split(',')) == n_cols:
                valid_lines.append(line)
        if len(valid_lines) < 2:
            return None
        data = np.genfromtxt(io.StringIO(''.join(valid_lines)), delimiter=',',
                             names=True, dtype=None, encoding='utf-8')
        return data
    except Exception:
        return None


def plot_mission_path(output_dir):
    """Plot combined mission trajectory (bird's-eye view + telemetry)."""
    csv_path = os.path.join(output_dir, 'combined_mission.csv')
    if not os.path.exists(csv_path):
        return

    data = load_csv(csv_path)
    if data is None or len(data) == 0:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Combined Mission Simulation', fontsize=14)

    # Bird's-eye view (QLabs overhead: +X=UP, +Y=LEFT → plot Y vs X, invert horizontal)
    ax = axes[0, 0]
    legs = np.unique(data['leg'])
    colors = {'hub_to_pickup': '#2196F3', 'pickup_to_dropoff': '#FF9800', 'dropoff_to_hub': '#4CAF50'}
    for leg in legs:
        leg_str = str(leg)
        mask = data['leg'] == leg
        c = colors.get(leg_str, '#888888')
        ax.plot(data['y_qlabs'][mask], data['x_qlabs'][mask], '-', color=c, linewidth=1.5, label=leg_str)
    # Plot reference paths if available
    for leg_str, c in colors.items():
        ref_path = os.path.join(output_dir, f'combined_ref_{leg_str}.csv')
        if os.path.exists(ref_path):
            ref = load_csv(ref_path)
            if ref is not None:
                ax.plot(ref['y_qlabs'], ref['x_qlabs'], '--', color=c, alpha=0.4, linewidth=0.8)
    ax.invert_xaxis()  # +Y_qlabs = LEFT = screen left
    ax.set_xlabel('Y_QLabs (m)')
    ax.set_ylabel('X_QLabs (m)')
    ax.set_title('Trajectory (QLabs overhead view)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # CTE over time
    ax = axes[0, 1]
    ax.plot(data['elapsed_s'], data['cte'] * 1000, 'r-', linewidth=0.8)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (mm)')
    ax.set_title(f'Cross-Track Error (max={np.max(data["cte"])*1000:.1f}mm)')
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[1, 0]
    ax.plot(data['elapsed_s'], data['v_meas'], 'b-', linewidth=0.8, label='v_meas')
    ax.plot(data['elapsed_s'], data['v_cmd'], 'r--', linewidth=0.8, alpha=0.6, label='v_cmd')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Velocity')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Steering
    ax = axes[1, 1]
    ax.plot(data['elapsed_s'], np.degrees(data['delta_cmd']), 'g-', linewidth=0.8)
    ax.axhline(y=25.8, color='r', linestyle='--', alpha=0.3, label='max')
    ax.axhline(y=-25.8, color='r', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (deg)')
    ax.set_title('Steering Command')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_mission.png'), dpi=150)
    plt.close()
    print(f'  combined_mission.png')


def plot_deployment_legs(output_dir):
    """Plot individual deployment leg traces."""
    leg_files = sorted(glob.glob(os.path.join(output_dir, 'deployment_*.csv')))
    if not leg_files:
        return

    fig, axes = plt.subplots(len(leg_files), 3, figsize=(14, 4 * len(leg_files)))
    if len(leg_files) == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Deployment Leg Tests', fontsize=14)

    for i, csv_path in enumerate(leg_files):
        data = load_csv(csv_path)
        if data is None or len(data) == 0:
            continue

        leg_name = os.path.basename(csv_path).replace('deployment_', '').replace('.csv', '')

        # XY trajectory
        ax = axes[i, 0]
        ax.plot(data['x'], data['y'], 'b-', linewidth=1)
        ax.set_title(f'{leg_name} — trajectory')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # CTE
        ax = axes[i, 1]
        steps = np.arange(len(data))
        ax.plot(steps, data['cte'] * 1000, 'r-', linewidth=0.8)
        ax.set_ylabel('CTE (mm)')
        ax.set_title(f'CTE (max={np.max(data["cte"])*1000:.1f}mm)')
        ax.grid(True, alpha=0.3)

        # Steering + velocity
        ax = axes[i, 2]
        ax.plot(steps, np.degrees(data['delta']), 'g-', linewidth=0.8, label='steering')
        ax2 = ax.twinx()
        ax2.plot(steps, data['v'], 'b-', linewidth=0.8, alpha=0.6, label='velocity')
        ax.set_ylabel('Steering (deg)')
        ax2.set_ylabel('Velocity (m/s)')
        ax.set_title('Controls')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deployment_legs.png'), dpi=150)
    plt.close()
    print(f'  deployment_legs.png')


def plot_mapframe_startup(output_dir):
    """Plot mapframe startup test results."""
    csv_path = os.path.join(output_dir, 'mapframe_startup.csv')
    if not os.path.exists(csv_path):
        return

    data = load_csv(csv_path)
    if data is None or len(data) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Mapframe Startup Test', fontsize=14)

    # CTE
    ax = axes[0]
    ax.plot(data['elapsed_s'], data['cross_track_err'] * 1000, 'r-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (mm)')
    ax.set_title(f'CTE (max={np.max(data["cross_track_err"])*1000:.1f}mm)')
    ax.grid(True, alpha=0.3)

    # Heading error
    ax = axes[1]
    ax.plot(data['elapsed_s'], np.degrees(data['heading_err']), 'b-', linewidth=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (deg)')
    ax.set_title('Heading Error')
    ax.grid(True, alpha=0.3)

    # Velocity + steering
    ax = axes[2]
    ax.plot(data['elapsed_s'], data['v_meas'], 'b-', linewidth=1, label='v_meas')
    ax.plot(data['elapsed_s'], data['v_cmd'], 'r--', linewidth=0.8, alpha=0.6, label='v_cmd')
    ax2 = ax.twinx()
    ax2.plot(data['elapsed_s'], np.degrees(data['delta_cmd']), 'g-', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax2.set_ylabel('Steering (deg)')
    ax.set_title('Controls')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mapframe_startup.png'), dpi=150)
    plt.close()
    print(f'  mapframe_startup.png')


def plot_full_mission_sim(output_dir):
    """Plot full mission simulation with traffic events."""
    for suffix in ['', '_qlabs', '_traffic']:
        csv_path = os.path.join(output_dir, f'full_mission_sim{suffix}.csv')
        if not os.path.exists(csv_path):
            continue

        data = load_csv(csv_path)
        if data is None or len(data) == 0:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        title = 'Full Mission Sim' + (' (QLabs)' if suffix == '_qlabs' else
                                      ' (Traffic)' if suffix == '_traffic' else '')
        fig.suptitle(title, fontsize=14)

        # XY — use QLabs overhead view orientation (+X=UP, +Y=LEFT)
        ax = axes[0, 0]
        x_col = 'x_qlabs' if 'x_qlabs' in data.dtype.names else 'x'
        y_col = 'y_qlabs' if 'y_qlabs' in data.dtype.names else 'y'
        is_qlabs = 'x_qlabs' in data.dtype.names
        if x_col in data.dtype.names and y_col in data.dtype.names:
            if is_qlabs:
                # QLabs overhead: plot Y on horizontal (inverted), X on vertical
                ax.plot(data[y_col], data[x_col], 'b-', linewidth=1)
                ax.invert_xaxis()
                ax.set_xlabel('Y_QLabs (m)')
                ax.set_ylabel('X_QLabs (m)')
            else:
                ax.plot(data[x_col], data[y_col], 'b-', linewidth=1)
        frame = 'QLabs overhead' if is_qlabs else 'map'
        ax.set_title(f'Trajectory ({frame} frame)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # CTE
        ax = axes[0, 1]
        cte_col = 'cross_track_err' if 'cross_track_err' in data.dtype.names else 'cte'
        if cte_col in data.dtype.names:
            ax.plot(data['elapsed_s'], data[cte_col] * 1000, 'r-', linewidth=0.8)
            ax.set_ylabel('CTE (mm)')
            ax.set_title(f'CTE (max={np.max(data[cte_col])*1000:.1f}mm)')
        ax.set_xlabel('Time (s)')
        ax.grid(True, alpha=0.3)

        # Velocity
        ax = axes[1, 0]
        v_col = 'v_meas' if 'v_meas' in data.dtype.names else 'v'
        vcmd_col = 'v_cmd' if 'v_cmd' in data.dtype.names else None
        if v_col in data.dtype.names:
            ax.plot(data['elapsed_s'], data[v_col], 'b-', linewidth=0.8, label='velocity')
        if vcmd_col and vcmd_col in data.dtype.names:
            ax.plot(data['elapsed_s'], data[vcmd_col], 'r--', linewidth=0.8, alpha=0.6, label='v_cmd')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Velocity (m/s)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Steering + progress
        ax = axes[1, 1]
        delta_col = 'delta_cmd' if 'delta_cmd' in data.dtype.names else 'delta'
        if delta_col in data.dtype.names:
            ax.plot(data['elapsed_s'], np.degrees(data[delta_col]), 'g-', linewidth=0.8)
        if 'progress_pct' in data.dtype.names:
            ax2 = ax.twinx()
            ax2.plot(data['elapsed_s'], data['progress_pct'], 'b--', linewidth=0.8, alpha=0.5)
            ax2.set_ylabel('Progress (%)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Steering (deg)')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = f'full_mission_sim{suffix}.png'
        plt.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close()
        print(f'  {fname}')


def plot_swerving_comparison(output_dir):
    """Plot heading_weight comparison from swerving diagnostic."""
    hw0 = os.path.join(output_dir, 'swerving_hw0.csv')
    hw2 = os.path.join(output_dir, 'swerving_hw2.csv')
    if not os.path.exists(hw0) or not os.path.exists(hw2):
        return

    d0 = load_csv(hw0)
    d2 = load_csv(hw2)
    if d0 is None or d2 is None:
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    fig.suptitle('Heading Weight Comparison (heading_weight=0 vs 2)', fontsize=14)

    # CTE comparison
    ax = axes[0]
    ax.plot(d0['step'], d0['cte'] * 1000, 'b-', linewidth=1, label='hw=0 (ref)')
    ax.plot(d2['step'], d2['cte'] * 1000, 'r-', linewidth=1, alpha=0.7, label='hw=2')
    ax.set_xlabel('Step')
    ax.set_ylabel('CTE (mm)')
    ax.set_title('Cross-Track Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Steering comparison
    ax = axes[1]
    ax.plot(d0['step'], np.degrees(d0['delta_cmd']), 'b-', linewidth=1, label='hw=0')
    ax.plot(d2['step'], np.degrees(d2['delta_cmd']), 'r-', linewidth=1, alpha=0.7, label='hw=2')
    ax.set_xlabel('Step')
    ax.set_ylabel('Steering (deg)')
    ax.set_title('Steering Command')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Heading error
    ax = axes[2]
    ax.plot(d0['step'], np.degrees(d0['heading_err']), 'b-', linewidth=1, label='hw=0')
    ax.plot(d2['step'], np.degrees(d2['heading_err']), 'r-', linewidth=1, alpha=0.7, label='hw=2')
    ax.set_xlabel('Step')
    ax.set_ylabel('Heading Error (deg)')
    ax.set_title('Heading Error')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'swerving_comparison.png'), dpi=150)
    plt.close()
    print(f'  swerving_comparison.png')


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <output_dir>")
        sys.exit(1)

    output_dir = sys.argv[1]
    if not os.path.isdir(output_dir):
        print(f"Error: {output_dir} is not a directory")
        sys.exit(1)

    plot_mission_path(output_dir)
    plot_deployment_legs(output_dir)
    plot_mapframe_startup(output_dir)
    plot_full_mission_sim(output_dir)
    plot_swerving_comparison(output_dir)


if __name__ == '__main__':
    main()
