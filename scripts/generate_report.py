#!/usr/bin/env python3
"""
Generate a mission report diagram showing planned vs executed paths.

Reads the MPCC CSV log, transforms executed path coordinates from map->QLabs frame,
regenerates the planned path using the SDCSRoadMap (matching the C++ single-loop
approach), and produces a PNG figure with:
  - SDCS road network (node positions and edges as thin gray lines)
  - Planned path legs in distinct colors (hub->pickup, pickup->dropoff, dropoff->hub)
  - Executed path color-coded by cross-track error (green=low, red=high)
  - Key waypoints (Hub, Pickup, Dropoff)
  - Stats: max cross-track error, avg speed, mission duration

Usage:
    python3 scripts/generate_report.py <mpcc_csv_path>
    python3 scripts/generate_report.py logs/mpcc_20260224_153012.csv
"""

import sys
import os
import math
import csv
import subprocess

import numpy as np

# ---------------------------------------------------------------------------
# Coordinate transform: map frame <-> QLabs frame
# ---------------------------------------------------------------------------
ORIGIN_X = -1.205
ORIGIN_Y = -0.83
THETA = 0.7177  # radians


def map_to_qlabs(x_map, y_map):
    """Transform map-frame coordinates back to QLabs world frame."""
    cos_t = math.cos(THETA)
    sin_t = math.sin(THETA)
    # Inverse of R(+theta): R(-theta)
    x_rot = x_map * cos_t + y_map * sin_t
    y_rot = -x_map * sin_t + y_map * cos_t
    return x_rot + ORIGIN_X, y_rot + ORIGIN_Y


def qlabs_to_map(x_ql, y_ql):
    """Transform QLabs world-frame coordinates to map frame."""
    cos_t = math.cos(THETA)
    sin_t = math.sin(THETA)
    dx = x_ql - ORIGIN_X
    dy = y_ql - ORIGIN_Y
    return dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t


# ---------------------------------------------------------------------------
# SDCS Road Network (node positions in QLabs frame, from SDCSRoadMap)
# ---------------------------------------------------------------------------
def get_sdcs_nodes():
    """Return node positions in QLabs frame as {id: (x, y)}."""
    scale = 0.002035
    xOffset = 1134
    yOffset = 2363

    raw_poses = [
        [1134, 2299],  # 0
        [1266, 2323],  # 1
        [1688, 2896],  # 2
        [1688, 2763],  # 3
        [2242, 2323],  # 4
        [2109, 2323],  # 5
        [1632, 1822],  # 6
        [1741, 1955],  # 7
        [766, 1822],   # 8
        [766, 1955],   # 9
        [504, 2589],   # 10
        [1134, 1300],  # 11
        [1134, 1454],  # 12
        [1266, 1454],  # 13
        [2242, 905],   # 14
        [2109, 1454],  # 15
        [1580, 540],   # 16
        [1854.4, 814.5],  # 17
        [1440, 856],   # 18
        [1523, 958],   # 19
        [1134, 153],   # 20
        [1134, 286],   # 21
        [159, 905],    # 22
        [291, 905],    # 23
    ]

    nodes = {}
    for i, (px, py) in enumerate(raw_poses):
        nodes[i] = (scale * (px - xOffset), scale * (yOffset - py))

    # Node 24: hub spawn
    nodes[24] = (-1.205, -0.83)
    return nodes


def get_sdcs_edges():
    """Return edge list as [(from_id, to_id), ...]."""
    return [
        (0, 2), (1, 7), (1, 8), (2, 4), (3, 1), (4, 6), (5, 3),
        (6, 0), (6, 8), (7, 5), (8, 10), (9, 0), (9, 7), (10, 1),
        (10, 2), (1, 13), (4, 14), (6, 13), (7, 14), (8, 23),
        (9, 13), (11, 12), (12, 0), (12, 7), (12, 8), (13, 19),
        (14, 16), (14, 20), (15, 5), (15, 6), (16, 17), (16, 18),
        (17, 15), (17, 16), (17, 20), (18, 11), (19, 17),
        (20, 22), (21, 16), (22, 9), (22, 10), (23, 21),
        (24, 2), (10, 24), (24, 1),
    ]


# ---------------------------------------------------------------------------
# Mission waypoints (QLabs frame)
# ---------------------------------------------------------------------------
HUB = (-1.205, -0.83)
PICKUP = (0.125, 4.395)
DROPOFF = (-0.905, 0.800)


# ---------------------------------------------------------------------------
# Generate planned path using SDCSRoadMap — matches C++ single-loop approach
# ---------------------------------------------------------------------------
def _find_first_local_min(dists, threshold=0.5):
    """Find the index of the first local minimum below threshold.

    Scans forward until we enter a region within `threshold` of the target,
    then returns the argmin within that contiguous region.
    """
    in_region = False
    region_start = 0
    best_idx = int(np.argmin(dists))  # fallback to global min

    for i in range(len(dists)):
        if dists[i] < threshold:
            if not in_region:
                in_region = True
                region_start = i
        else:
            if in_region:
                # Exited the first close region — find min within it
                best_idx = region_start + int(
                    np.argmin(dists[region_start:i]))
                return best_idx
    # If we ended while still in region
    if in_region:
        best_idx = region_start + int(
            np.argmin(dists[region_start:]))
    return best_idx


def generate_planned_path():
    """Generate the planned path in QLabs frame matching the C++ RoadGraph.

    The C++ road_graph.cpp generates a SINGLE loop path through node sequence
    [24, 20, 9, 10], resamples it uniformly, applies scale factor [1.01, 1.0],
    then slices at the closest indices to Pickup and Dropoff to produce three
    route legs. This function replicates that exact approach.
    """
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from acc_stage1_mission.road_graph import SDCSRoadMap

        roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)

        # Same hub node and edges as C++ road_graph.cpp constructor
        hub_heading = (-44.7) % (2 * np.pi)
        roadmap.add_node([-1.205, -0.83, hub_heading])  # node 24
        roadmap.add_edge(24, 2, radius=0.0)
        roadmap.add_edge(10, 24, radius=0.0)
        roadmap.add_edge(24, 1, radius=0.866326)

        # Generate single loop path (same as C++)
        loop_sequence = [24, 20, 9, 10]
        loop_path_2xN = roadmap.generate_path(
            loop_sequence, spacing=0.001, scale_factor=[1.01, 1.0])

        if loop_path_2xN is None:
            print("Warning: Could not generate loop path")
            return {}

        loop_path = loop_path_2xN.T  # Nx2

        # Find waypoint indices for slicing.
        # The loop path passes near Pickup TWICE (once on the way out via
        # node 20, once on the return via inner track). We need the FIRST
        # pass for correct mission leg ordering: Hub -> Pickup -> Dropoff -> Hub.
        pickup_pt = np.array(PICKUP)
        dropoff_pt = np.array(DROPOFF)

        dists_pickup = np.linalg.norm(loop_path - pickup_pt, axis=1)
        dists_dropoff = np.linalg.norm(loop_path - dropoff_pt, axis=1)

        pickup_idx = _find_first_local_min(dists_pickup, threshold=0.5)
        # Search for dropoff only AFTER pickup (it must come second)
        dropoff_idx = pickup_idx + _find_first_local_min(
            dists_dropoff[pickup_idx:], threshold=0.5)

        print(f"  Loop: {len(loop_path)} pts, pickup_idx={pickup_idx}, "
              f"dropoff_idx={dropoff_idx}")

        # Slice into three legs
        legs = {}
        legs['hub_to_pickup'] = loop_path[:pickup_idx + 1]
        if pickup_idx < dropoff_idx:
            legs['pickup_to_dropoff'] = loop_path[pickup_idx:dropoff_idx + 1]
        legs['dropoff_to_hub'] = loop_path[dropoff_idx:]

        for name, path in legs.items():
            print(f"  {name}: {len(path)} waypoints")

        return legs
    except Exception as e:
        print(f"Warning: Could not generate planned path: {e}")
        import traceback
        traceback.print_exc()
        return {}


# ---------------------------------------------------------------------------
# Read MPCC CSV log
# ---------------------------------------------------------------------------
def read_mpcc_csv(csv_path):
    """Read MPCC CSV log and return structured data."""
    data = {
        'elapsed_s': [], 'x': [], 'y': [], 'theta': [],
        'v_meas': [], 'v_cmd': [], 'cross_track_err': [],
    }

    with open(csv_path, 'r') as f:
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
            except (KeyError, ValueError):
                continue

    for key in data:
        data[key] = np.array(data[key])

    return data


# ---------------------------------------------------------------------------
# Generate report figure
# ---------------------------------------------------------------------------

# Distinct colors for each mission leg
LEG_COLORS = {
    'hub_to_pickup': '#00BFFF',       # deep sky blue
    'pickup_to_dropoff': '#FF6600',   # orange
    'dropoff_to_hub': '#00CC66',      # green
}
LEG_LABELS = {
    'hub_to_pickup': 'Planned: Hub \u2192 Pickup',
    'pickup_to_dropoff': 'Planned: Pickup \u2192 Dropoff',
    'dropoff_to_hub': 'Planned: Dropoff \u2192 Hub',
}


def generate_figure(csv_path, data, planned_paths):
    """Generate and save the mission report figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # --- Draw SDCS road network (thin gray lines in QLabs frame) ---
    nodes = get_sdcs_nodes()
    edges = get_sdcs_edges()

    for (n1, n2) in edges:
        if n1 in nodes and n2 in nodes:
            x1, y1 = nodes[n1]
            x2, y2 = nodes[n2]
            ax.plot([x1, x2], [y1, y2], color='#cccccc', linewidth=0.5, zorder=1)

    # Draw node positions
    for nid, (nx, ny) in nodes.items():
        ax.plot(nx, ny, 'o', color='#aaaaaa', markersize=3, zorder=2)
        ax.annotate(str(nid), (nx, ny), fontsize=5, color='#999999',
                    ha='center', va='bottom', xytext=(0, 3),
                    textcoords='offset points')

    # --- Draw planned paths (each leg in a distinct color) ---
    for name, path in planned_paths.items():
        color = LEG_COLORS.get(name, 'cyan')
        label = LEG_LABELS.get(name, f'Planned: {name}')
        ax.plot(path[:, 0], path[:, 1], color=color, linewidth=2.0,
                alpha=0.8, zorder=3, label=label)

    # --- Draw executed path color-coded by cross-track error ---
    # Transform map -> QLabs
    x_ql = np.zeros(len(data['x']))
    y_ql = np.zeros(len(data['y']))
    for i in range(len(data['x'])):
        x_ql[i], y_ql[i] = map_to_qlabs(data['x'][i], data['y'][i])

    cte = np.abs(data['cross_track_err'])
    max_cte_color = 0.5  # saturate colormap at 0.5m

    # Create line segments for color-coded path
    points = np.column_stack([x_ql, y_ql]).reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    norm = mcolors.Normalize(vmin=0, vmax=max_cte_color)
    cmap = plt.cm.RdYlGn_r  # green (low error) -> red (high error)
    # Use average CTE of each segment pair
    seg_cte = (cte[:-1] + cte[1:]) / 2.0

    lc = LineCollection(segments, cmap=cmap, norm=norm, linewidths=2.5, zorder=5)
    lc.set_array(seg_cte)
    ax.add_collection(lc)
    fig.colorbar(lc, ax=ax, shrink=0.6, label='Cross-track error (m)')

    # --- Draw waypoints ---
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

    # --- Stats text ---
    duration = data['elapsed_s'][-1] - data['elapsed_s'][0] if len(data['elapsed_s']) > 1 else 0
    max_cte = np.max(cte) if len(cte) > 0 else 0
    avg_cte = np.mean(cte) if len(cte) > 0 else 0
    avg_speed = np.mean(np.abs(data['v_meas'])) if len(data['v_meas']) > 0 else 0

    stats_text = (
        f"Duration: {duration:.1f}s\n"
        f"Max CTE: {max_cte:.3f}m\n"
        f"Avg CTE: {avg_cte:.3f}m\n"
        f"Avg speed: {avg_speed:.2f} m/s"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    # --- Formatting ---
    ax.set_xlabel('X (QLabs frame, m)')
    ax.set_ylabel('Y (QLabs frame, m)')
    ax.set_title('Mission Report: Planned vs Executed Path')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', fontsize=8)

    # Auto-scale with some margin
    all_x = list(x_ql)
    all_y = list(y_ql)
    for path in planned_paths.values():
        all_x.extend(path[:, 0])
        all_y.extend(path[:, 1])
    for (wx, wy) in waypoints.values():
        all_x.append(wx)
        all_y.append(wy)

    if all_x and all_y:
        margin = 0.3
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    # Save
    csv_basename = os.path.splitext(os.path.basename(csv_path))[0]
    output_dir = os.path.dirname(csv_path)
    if not output_dir:
        output_dir = '.'
    output_path = os.path.join(output_dir, f"{csv_basename}_report.png")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"Report saved: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <mpcc_csv_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    if not os.path.isfile(csv_path):
        print(f"Error: CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading MPCC log: {csv_path}")
    data = read_mpcc_csv(csv_path)
    if len(data['x']) == 0:
        print("Error: No data points in CSV")
        sys.exit(1)
    print(f"  {len(data['x'])} data points, {data['elapsed_s'][-1]:.1f}s duration")

    print("Generating planned path...")
    planned_paths = generate_planned_path()
    if planned_paths:
        print(f"  {len(planned_paths)} route legs")
    else:
        print("  Warning: No planned paths generated (proceeding with executed only)")

    print("Generating figure...")
    output_path = generate_figure(csv_path, data, planned_paths)

    # Try to open the image
    try:
        subprocess.Popen(['xdg-open', output_path],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass  # xdg-open not available or no display

    return 0


if __name__ == '__main__':
    sys.exit(main())
