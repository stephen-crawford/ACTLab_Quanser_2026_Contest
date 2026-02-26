#!/usr/bin/env python3
"""
Generate mission report visualization matching the deployment report format.

Reads the full_mission_sim.csv output from test_full_mission_sim and produces
a QLabs-frame report figure showing:
  - SDCS road network (nodes and edges)
  - Planned paths for all 3 legs (distinct colors)
  - Simulated vehicle trajectory color-coded by CTE
  - Mission waypoints (Hub, Pickup, Dropoff)
  - Statistics (duration, max CTE, avg CTE, avg speed)

Also generates per-leg diagnostic plots (CTE, velocity, steering over time).
"""

import sys
import os
import math
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

# ---------------------------------------------------------------------------
# Coordinate transform
# ---------------------------------------------------------------------------
ORIGIN_X = -1.205
ORIGIN_Y = -0.83
THETA = 0.7177

def map_to_qlabs(x_map, y_map):
    cos_t, sin_t = math.cos(THETA), math.sin(THETA)
    x_rot = x_map * cos_t + y_map * sin_t
    y_rot = -x_map * sin_t + y_map * cos_t
    return x_rot + ORIGIN_X, y_rot + ORIGIN_Y

def qlabs_to_map(x_ql, y_ql):
    cos_t, sin_t = math.cos(THETA), math.sin(THETA)
    dx, dy = x_ql - ORIGIN_X, y_ql - ORIGIN_Y
    return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t

# ---------------------------------------------------------------------------
# SDCS Road Network
# ---------------------------------------------------------------------------
def get_sdcs_nodes():
    scale, xOff, yOff = 0.002035, 1134, 2363
    raw = [
        [1134,2299],[1266,2323],[1688,2896],[1688,2763],[2242,2323],
        [2109,2323],[1632,1822],[1741,1955],[766,1822],[766,1955],
        [504,2589],[1134,1300],[1134,1454],[1266,1454],[2242,905],
        [2109,1454],[1580,540],[1854.4,814.5],[1440,856],[1523,958],
        [1134,153],[1134,286],[159,905],[291,905],
    ]
    nodes = {}
    for i, (px, py) in enumerate(raw):
        nodes[i] = (scale*(px-xOff), scale*(yOff-py))
    nodes[24] = (-1.205, -0.83)
    return nodes

def get_sdcs_edges():
    return [
        (0,2),(1,7),(1,8),(2,4),(3,1),(4,6),(5,3),(6,0),(6,8),(7,5),
        (8,10),(9,0),(9,7),(10,1),(10,2),(1,13),(4,14),(6,13),(7,14),
        (8,23),(9,13),(11,12),(12,0),(12,7),(12,8),(13,19),(14,16),
        (14,20),(15,5),(15,6),(16,17),(16,18),(17,15),(17,16),(17,20),
        (18,11),(19,17),(20,22),(21,16),(22,9),(22,10),(23,21),
        (24,2),(10,24),(24,1),
    ]

# Mission waypoints (QLabs frame)
HUB = (-1.205, -0.83)
PICKUP = (0.125, 4.395)
DROPOFF = (-0.905, 0.800)

# Leg colors
LEG_COLORS = {
    'hub_to_pickup': '#00BFFF',
    'pickup_to_dropoff': '#FF6600',
    'dropoff_to_hub': '#00CC66',
}
LEG_LABELS = {
    'hub_to_pickup': 'Planned: Hub → Pickup',
    'pickup_to_dropoff': 'Planned: Pickup → Dropoff',
    'dropoff_to_hub': 'Planned: Dropoff → Hub',
}

# ---------------------------------------------------------------------------
# Generate planned paths using road_graph Python module
# ---------------------------------------------------------------------------
def generate_planned_paths():
    """Generate planned paths in QLabs frame."""
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
        from acc_stage1_mission.road_graph import SDCSRoadMap

        roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
        hub_heading = (-44.7) % (2 * np.pi)
        roadmap.add_node([-1.205, -0.83, hub_heading])
        roadmap.add_edge(24, 2, radius=0.0)
        roadmap.add_edge(10, 24, radius=0.0)
        roadmap.add_edge(24, 1, radius=0.866326)

        loop_path = roadmap.generate_path([24, 20, 9, 10],
                                          spacing=0.001,
                                          scale_factor=[1.01, 1.0])
        if loop_path is None:
            return {}

        loop = loop_path.T  # Nx2

        pickup_pt = np.array(PICKUP)
        dropoff_pt = np.array(DROPOFF)
        dists_p = np.linalg.norm(loop - pickup_pt, axis=1)
        dists_d = np.linalg.norm(loop - dropoff_pt, axis=1)

        def find_first_local_min(dists, thresh=0.5):
            in_region = False
            region_start = 0
            best = int(np.argmin(dists))
            for i in range(len(dists)):
                if dists[i] < thresh:
                    if not in_region:
                        in_region = True
                        region_start = i
                else:
                    if in_region:
                        return region_start + int(np.argmin(dists[region_start:i]))
            if in_region:
                return region_start + int(np.argmin(dists[region_start:]))
            return best

        pi = find_first_local_min(dists_p)
        di = pi + find_first_local_min(dists_d[pi:])

        return {
            'hub_to_pickup': loop[:pi+1],
            'pickup_to_dropoff': loop[pi:di+1],
            'dropoff_to_hub': loop[di:],
        }
    except Exception as e:
        print(f"Warning: Could not generate planned paths: {e}")
        import traceback
        traceback.print_exc()
        return {}

# ---------------------------------------------------------------------------
# Read CSV
# ---------------------------------------------------------------------------
def read_csv(path):
    data = {'elapsed_s': [], 'x': [], 'y': [], 'theta': [],
            'v_meas': [], 'v_cmd': [], 'cross_track_err': [],
            'delta_cmd': [], 'heading_err': []}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data['elapsed_s'].append(float(row['elapsed_s']))
                data['x'].append(float(row['x']))
                data['y'].append(float(row['y']))
                data['theta'].append(float(row['theta']))
                data['v_meas'].append(float(row['v_meas']))
                data['v_cmd'].append(float(row['v_cmd']))
                data['cross_track_err'].append(float(row['cross_track_err']))
                data['delta_cmd'].append(float(row.get('delta_cmd', '0')))
                data['heading_err'].append(float(row.get('heading_err', '0')))
            except (KeyError, ValueError):
                continue
    for k in data:
        data[k] = np.array(data[k])
    return data

# ---------------------------------------------------------------------------
# Generate mission report figure (matching deployment screenshot format)
# ---------------------------------------------------------------------------
def generate_report(data, planned_paths, output_path, blended_paths=None):
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw SDCS network
    nodes = get_sdcs_nodes()
    edges = get_sdcs_edges()
    for n1, n2 in edges:
        if n1 in nodes and n2 in nodes:
            ax.plot([nodes[n1][0], nodes[n2][0]],
                    [nodes[n1][1], nodes[n2][1]],
                    color='#cccccc', linewidth=0.5, zorder=1)
    for nid, (nx, ny) in nodes.items():
        ax.plot(nx, ny, 'o', color='#aaaaaa', markersize=3, zorder=2)
        ax.annotate(str(nid), (nx, ny), fontsize=5, color='#999999',
                    ha='center', va='bottom', xytext=(0, 3),
                    textcoords='offset points')

    # Draw planned paths (thin, background)
    for name, path in planned_paths.items():
        color = LEG_COLORS.get(name, 'cyan')
        label = LEG_LABELS.get(name, name)
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=1.5,
                alpha=0.4, zorder=3, label=label)

    # Draw blended paths (what the controller actually tracks) — thicker, foreground
    if blended_paths:
        BLEND_LABELS = {
            'hub_to_pickup': 'Tracked: Hub -> Pickup',
            'pickup_to_dropoff': 'Tracked: Pickup -> Dropoff',
            'dropoff_to_hub': 'Tracked: Dropoff -> Hub',
        }
        for name, path in blended_paths.items():
            color = LEG_COLORS.get(name, 'cyan')
            label = BLEND_LABELS.get(name, f'Tracked: {name}')
            ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.5,
                    alpha=0.9, zorder=4, label=label, linestyle='--')

    # Draw executed path (map→QLabs transform)
    x_ql = np.zeros(len(data['x']))
    y_ql = np.zeros(len(data['y']))
    for i in range(len(data['x'])):
        x_ql[i], y_ql[i] = map_to_qlabs(data['x'][i], data['y'][i])

    cte = np.abs(data['cross_track_err'])
    points = np.column_stack([x_ql, y_ql]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = mcolors.Normalize(vmin=0, vmax=0.5)
    seg_cte = (cte[:-1] + cte[1:]) / 2.0
    lc = LineCollection(segments, cmap='RdYlGn_r', norm=norm,
                        linewidths=2.5, zorder=5)
    lc.set_array(seg_cte)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.6, label='Cross-track error (m)')

    # Waypoints
    waypoints = {'Hub': HUB, 'Pickup': PICKUP, 'Dropoff': DROPOFF}
    colors = {'Hub': 'magenta', 'Pickup': 'blue', 'Dropoff': 'orange'}
    for name, (wx, wy) in waypoints.items():
        ax.plot(wx, wy, '*', color=colors[name], markersize=15, zorder=10,
                markeredgecolor='black', markeredgewidth=0.5)
        ax.annotate(name, (wx, wy), fontsize=10, fontweight='bold',
                    color=colors[name], ha='center', va='bottom',
                    xytext=(0, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                              edgecolor=colors[name], alpha=0.8))

    # Stats
    duration = data['elapsed_s'][-1] - data['elapsed_s'][0] if len(data['elapsed_s']) > 1 else 0
    max_cte = np.max(cte) if len(cte) > 0 else 0
    avg_cte = np.mean(cte) if len(cte) > 0 else 0
    avg_speed = np.mean(np.abs(data['v_meas'])) if len(data['v_meas']) > 0 else 0

    stats = (f"Duration: {duration:.1f}s\n"
             f"Max CTE: {max_cte:.3f}m\n"
             f"Avg CTE: {avg_cte:.3f}m\n"
             f"Avg speed: {avg_speed:.2f} m/s")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax.set_xlabel('X (QLabs frame, m)')
    ax.set_ylabel('Y (QLabs frame, m)')
    ax.set_title('Simulation Report: Planned vs Simulated Path')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    # Auto-scale
    all_x, all_y = list(x_ql), list(y_ql)
    for path in planned_paths.values():
        all_x.extend(path[:, 0])
        all_y.extend(path[:, 1])
    margin = 0.3
    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Report saved: {output_path}")

# ---------------------------------------------------------------------------
# Generate per-leg diagnostic plots
# ---------------------------------------------------------------------------
def generate_diagnostics(data, output_path):
    """Generate 2x2 diagnostic subplots (CTE, velocity, steering, heading error)."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Full Mission Simulation Diagnostics', fontsize=14, fontweight='bold')

    steps = np.arange(len(data['elapsed_s']))
    elapsed = data['elapsed_s']

    # CTE over time
    ax = axes[0, 0]
    cte = np.abs(data['cross_track_err'])
    ax.plot(elapsed, cte, 'r-', linewidth=0.8)
    ax.axhline(0.15, color='orange', linestyle='--', alpha=0.5, label='Avg target')
    ax.axhline(0.30, color='red', linestyle='--', alpha=0.5, label='Max target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CTE (m)')
    ax.set_title(f'Cross-Track Error (max={np.max(cte):.3f}m, avg={np.mean(cte):.3f}m)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Velocity
    ax = axes[0, 1]
    ax.plot(elapsed, data['v_meas'], 'g-', linewidth=0.8, label='v_meas')
    ax.plot(elapsed, data['v_cmd'], 'b--', linewidth=0.6, alpha=0.6, label='v_cmd')
    ax.axhline(0.45, color='blue', linestyle=':', alpha=0.3, label='v_ref')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title(f'Speed (avg={np.mean(data["v_meas"]):.2f} m/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Steering
    ax = axes[1, 0]
    delta_deg = data['delta_cmd'] * 180 / np.pi
    ax.plot(elapsed, delta_deg, 'm-', linewidth=0.8)
    max_steer = 0.45 * 180 / np.pi  # 25.8°
    ax.axhline(max_steer, color='red', linestyle='--', alpha=0.3)
    ax.axhline(-max_steer, color='red', linestyle='--', alpha=0.3)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Steering (deg)')
    ax.set_title('Steering Angle')
    ax.grid(True, alpha=0.3)

    # Heading error
    ax = axes[1, 1]
    heading_err_deg = data['heading_err'] * 180 / np.pi
    ax.plot(elapsed, heading_err_deg, 'c-', linewidth=0.8)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Heading Error (deg)')
    ax.set_title(f'Heading Error (max={np.max(np.abs(heading_err_deg)):.1f}°)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Diagnostics saved: {output_path}")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    csv_path = 'full_mission_sim.csv'
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]

    if not os.path.isfile(csv_path):
        print(f"Error: CSV not found: {csv_path}")
        sys.exit(1)

    print(f"Reading: {csv_path}")
    data = read_csv(csv_path)
    print(f"  {len(data['x'])} data points")

    print("Generating planned paths...")
    planned = generate_planned_paths()
    if planned:
        for name, path in planned.items():
            print(f"  {name}: {len(path)} waypoints")

    # Try to load blended paths (actual paths tracked by controller)
    blended = {}
    for name in ['hub_to_pickup', 'pickup_to_dropoff', 'dropoff_to_hub']:
        bp_file = os.path.join(os.path.dirname(csv_path), f'blended_{name}.csv')
        if os.path.isfile(bp_file):
            bp_data = np.loadtxt(bp_file, delimiter=',', skiprows=1)
            if len(bp_data) > 0:
                blended[name] = bp_data
                print(f"  Blended {name}: {len(bp_data)} waypoints")

    print("Generating report...")
    generate_report(data, planned, 'full_mission_sim_report.png', blended)

    print("Generating diagnostics...")
    generate_diagnostics(data, 'full_mission_sim_diagnostics.png')

if __name__ == '__main__':
    main()
