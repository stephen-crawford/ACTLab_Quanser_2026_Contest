#!/usr/bin/env python3
"""
Replicate the C++ road_graph.cpp SCSPath and path generation for the
"pickup_to_dropoff" route (nodes [21, 16, 18, 11, 12, 8]).

Faithfully implements:
  - SCSPath (radius=0 straight line + general case)
  - RoadMap::add_edge, find_shortest_path, generate_path
  - SDCSRoadMap node/edge definitions
  - RoadGraph endpoint attachment
"""

import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TWO_PI = 2.0 * math.pi

# --------------------------------------------------------------------------
# Math utilities (exact C++ ports)
# --------------------------------------------------------------------------

def wrap_to_2pi(th):
    th = math.fmod(math.fmod(th, TWO_PI) + TWO_PI, TWO_PI)
    return th

def wrap_to_pi(th):
    th = math.fmod(th, TWO_PI)
    th = math.fmod(th + TWO_PI, TWO_PI)
    if th > math.pi:
        th -= TWO_PI
    return th

def signed_angle_between(v1x, v1y, v2x, v2y):
    return wrap_to_pi(math.atan2(v2y, v2x) - math.atan2(v1y, v1x))

# --------------------------------------------------------------------------
# SCSPath - exact C++ port
# --------------------------------------------------------------------------

def SCSPath(start, end, radius, stepSize=0.01):
    """
    Returns dict with keys: x, y (lists), length (float), valid (bool).
    start/end: (x, y, theta)
    """
    result = {'x': [], 'y': [], 'length': 0.0, 'valid': False}

    p1x, p1y, th1 = start
    p2x, p2y, th2 = end

    if radius < 1e-6:
        # Straight line
        dx = p2x - p1x
        dy = p2y - p1y
        dist = math.hypot(dx, dy)
        if dist < 1e-6:
            result['valid'] = True
            result['length'] = 0.0
            return result
        n_pts = max(int(dist / stepSize), 2)
        for i in range(1, n_pts):
            t = i / n_pts
            result['x'].append(p1x + t * dx)
            result['y'].append(p1y + t * dy)
        result['length'] = dist
        result['valid'] = True
        return result

    t1x = math.cos(th1)
    t1y = math.sin(th1)
    t2x = math.cos(th2)
    t2y = math.sin(th2)

    # Direction of turn
    dp_x = p2x - p1x
    dp_y = p2y - p1y
    sa = signed_angle_between(t1x, t1y, dp_x, dp_y)
    if abs(sa) < 0.05:
        sa = wrap_to_pi(th2 - th1)
    direction = 1 if sa > 0 else -1

    n1x = radius * (-t1y) * direction
    n1y = radius * ( t1x) * direction
    n2x = radius * (-t2y) * direction
    n2y = radius * ( t2x) * direction

    tol = 0.01
    angle_diff = wrap_to_pi(th2 - th1)

    cx, cy = None, None

    if abs(angle_diff) < tol:
        # Nearly parallel headings
        vx = p2x - p1x
        vy = p2y - p1y
        v_norm = math.hypot(vx, vy)
        if v_norm < 1e-9:
            result['valid'] = True
            result['length'] = 0.0
            return result
        vux = vx / v_norm
        vuy = vy / v_norm
        dot = t1x * vux + t1y * vuy
        if 1.0 - abs(dot) < tol:
            cx = p2x + n1x
            cy = p2y + n1y
        else:
            return result  # invalid

    elif abs(wrap_to_pi(th2 - th1 + math.pi)) < tol:
        # Anti-parallel headings
        vx = (p2x + 2 * n2x) - p1x
        vy = (p2y + 2 * n2y) - p1y
        v_norm = math.hypot(vx, vy)
        if v_norm < 1e-9:
            result['valid'] = True
            result['length'] = 0.0
            return result
        vux = vx / v_norm
        vuy = vy / v_norm
        dot = t1x * vux + t1y * vuy
        if 1.0 - abs(dot) < tol:
            s = t1x * vx + t1y * vy
            if s < tol:
                cx = p1x + n1x
                cy = p1y + n1y
            else:
                cx = p2x + n2x
                cy = p2y + n2y
        else:
            return result  # invalid

    else:
        # General case: solve 2x2 linear system
        d1x = p1x + n1x
        d1y = p1y + n1y
        d2x = p2x + n2x
        d2y = p2y + n2y
        # A = [t1x, -t2x; t1y, -t2y], b = [d2x-d1x; d2y-d1y]
        det = t1x * (-t2y) - (-t2x) * t1y
        if abs(det) < 1e-10:
            return result  # invalid

        bx = d2x - d1x
        by = d2y - d1y
        alpha = ((-t2y) * bx - (-t2x) * by) / det
        beta  = (t1x * by - t1y * bx) / det

        if alpha >= -tol and beta <= tol:
            cx = d1x + alpha * t1x
            cy = d1y + alpha * t1y
        else:
            return result  # invalid

    if cx is None:
        return result

    # Tangent points on circle
    b1x = cx - n1x
    b1y = cy - n1y
    b2x = cx - n2x
    b2y = cy - n2y

    # Discretize line p1 -> b1
    line1_len = math.hypot(b1x - p1x, b1y - p1y)
    if line1_len > stepSize:
        ds = stepSize / line1_len
        s = ds
        while s < 1.0:
            result['x'].append(p1x + s * (b1x - p1x))
            result['y'].append(p1y + s * (b1y - p1y))
            s += ds

    # Discretize arc b1 -> b2
    av1x = b1x - cx
    av1y = b1y - cy
    av2x = b2x - cx
    av2y = b2y - cy
    ang_dist = wrap_to_2pi(direction *
        wrap_to_pi(math.atan2(av2y, av2x) - math.atan2(av1y, av1x)))
    arc_length = abs(ang_dist * radius)
    if arc_length > stepSize:
        start_angle = math.atan2(av1y, av1x)
        dth = stepSize / radius
        s = dth
        while s < ang_dist:
            th = start_angle + s * direction
            result['x'].append(cx + math.cos(th) * radius)
            result['y'].append(cy + math.sin(th) * radius)
            s += dth

    # Discretize line b2 -> p2
    line2_len = math.hypot(b2x - p2x, b2y - p2y)
    if line2_len > stepSize:
        ds = stepSize / line2_len
        s = ds
        while s < 1.0:
            result['x'].append(b2x + s * (p2x - b2x))
            result['y'].append(b2y + s * (p2y - b2y))
            s += ds

    result['length'] = line1_len + arc_length + line2_len
    result['valid'] = True
    return result

# --------------------------------------------------------------------------
# RoadMap classes
# --------------------------------------------------------------------------

class RoadMapNode:
    def __init__(self, x, y, theta, node_id):
        self.pose = (x, y, theta)
        self.id = node_id
        self.out_edges = []
        self.in_edges = []

class RoadMapEdge:
    def __init__(self, from_node, to_node):
        self.from_node = from_node
        self.to_node = to_node
        self.wp_x = []
        self.wp_y = []
        self.length = 0.0

class RoadMap:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, x, y, theta):
        node = RoadMapNode(x, y, theta, len(self.nodes))
        self.nodes.append(node)

    def add_edge(self, from_idx, to_idx, radius):
        edge_idx = len(self.edges)
        edge = RoadMapEdge(from_idx, to_idx)

        scs = SCSPath(self.nodes[from_idx].pose, self.nodes[to_idx].pose, radius)
        edge.wp_x = scs['x']
        edge.wp_y = scs['y']
        edge.length = scs['length']

        self.edges.append(edge)
        self.nodes[from_idx].out_edges.append(edge_idx)
        self.nodes[to_idx].in_edges.append(edge_idx)

        return scs  # return for diagnostics

    def find_shortest_path(self, start_idx, goal_idx):
        """A* shortest path. Returns (px, py) or None."""
        import heapq

        if start_idx == goal_idx:
            return None

        N = len(self.nodes)
        g_score = [1e18] * N
        g_score[start_idx] = 0.0

        came_from = [(-1, -1)] * N  # (prev_node, edge_idx)

        gx = self.nodes[goal_idx].pose[0]
        gy = self.nodes[goal_idx].pose[1]
        h0 = math.hypot(gx - self.nodes[start_idx].pose[0],
                         gy - self.nodes[start_idx].pose[1])

        open_set = [(h0, start_idx)]
        closed = set()

        while open_set:
            f, current = heapq.heappop(open_set)

            if current == goal_idx:
                # Reconstruct path
                px = [self.nodes[goal_idx].pose[0]]
                py = [self.nodes[goal_idx].pose[1]]

                node = goal_idx
                while came_from[node][0] >= 0:
                    prev = came_from[node][0]
                    eidx = came_from[node][1]
                    e = self.edges[eidx]

                    seg_x = [self.nodes[prev].pose[0]] + list(e.wp_x)
                    seg_y = [self.nodes[prev].pose[1]] + list(e.wp_y)
                    seg_x.extend(px)
                    seg_y.extend(py)
                    px = seg_x
                    py = seg_y

                    node = prev
                    if came_from[node][0] < 0:
                        break

                return (px, py)

            if current in closed:
                continue
            closed.add(current)

            for eidx in self.nodes[current].out_edges:
                edge = self.edges[eidx]
                neighbor = edge.to_node
                if neighbor in closed:
                    continue
                if edge.length <= 0:
                    continue

                tent_g = g_score[current] + edge.length
                if tent_g < g_score[neighbor]:
                    came_from[neighbor] = (current, eidx)
                    g_score[neighbor] = tent_g
                    h = math.hypot(gx - self.nodes[neighbor].pose[0],
                                   gy - self.nodes[neighbor].pose[1])
                    heapq.heappush(open_set, (tent_g + h, neighbor))

        return None

    def generate_path(self, node_sequence):
        """Generate path through a sequence of node indices."""
        px, py = [], []
        for i in range(len(node_sequence) - 1):
            seg = self.find_shortest_path(node_sequence[i], node_sequence[i + 1])
            if seg is None:
                return None
            sx, sy = seg
            # Append all but last point
            for j in range(len(sx) - 1):
                px.append(sx[j])
                py.append(sy[j])
        # Add final node
        last = node_sequence[-1]
        px.append(self.nodes[last].pose[0])
        py.append(self.nodes[last].pose[1])
        return (px, py)

# --------------------------------------------------------------------------
# Build SDCSRoadMap (full map, right-hand traffic)
# --------------------------------------------------------------------------

def build_sdcs_roadmap():
    scale = 0.002035
    xOffset = 1134
    yOffset = 2363

    innerR  = 305.5 * scale
    outerR  = 438.0 * scale
    circleR = 333.0 * scale
    onewayR = 350.0 * scale
    kinkR   = 375.0 * scale

    pi = math.pi
    hpi = math.pi / 2.0

    # Node poses: (px, py, th)
    nodePoses = [
        (1134, 2299, -hpi),        # 0
        (1266, 2323,  hpi),        # 1
        (1688, 2896,  0),          # 2
        (1688, 2763,  pi),         # 3
        (2242, 2323,  hpi),        # 4
        (2109, 2323, -hpi),        # 5
        (1632, 1822,  pi),         # 6
        (1741, 1955,  0),          # 7
        ( 766, 1822,  pi),         # 8
        ( 766, 1955,  0),          # 9
        ( 504, 2589, -42*pi/180),  # 10
        # Extra (full map)
        (1134, 1300, -hpi),              # 11
        (1134, 1454, -hpi),              # 12
        (1266, 1454,  hpi),              # 13
        (2242,  905,  hpi),              # 14
        (2109, 1454, -hpi),              # 15
        (1580,  540, -80.6*pi/180),      # 16
        (1854.4, 814.5, -9.4*pi/180),   # 17
        (1440,  856, -138*pi/180),       # 18
        (1523,  958,  42*pi/180),        # 19
        (1134,  153,  pi),               # 20
        (1134,  286,  0),                # 21
        ( 159,  905, -hpi),              # 22
        ( 291,  905,  hpi),              # 23
    ]

    rm = RoadMap()
    for np_ in nodePoses:
        x = scale * (np_[0] - xOffset)
        y = scale * (yOffset - np_[1])
        rm.add_node(x, y, np_[2])

    # Print node positions for the pickup_to_dropoff route
    print("=" * 70)
    print("NODE POSITIONS (QLabs world frame)")
    print("=" * 70)
    for nid in [21, 16, 18, 11, 12, 8]:
        n = rm.nodes[nid]
        print(f"  Node {nid:2d}: x={n.pose[0]:+.4f}, y={n.pose[1]:+.4f}, "
              f"th={math.degrees(n.pose[2]):+.1f} deg")
    print()

    # Edge configs (from C++ SDCSRoadMap constructor)
    edgeConfigs = [
        (0, 2, outerR),  (1, 7, innerR), (1, 8, outerR),
        (2, 4, outerR),  (3, 1, innerR), (4, 6, outerR),
        (5, 3, innerR),  (6, 0, outerR), (6, 8, 0),
        (7, 5, innerR),  (8, 10, onewayR), (9, 0, innerR),
        (9, 7, 0),       (10, 1, innerR), (10, 2, innerR),
        # Extra edges (full map)
        (1, 13, 0),      (4, 14, 0),       (6, 13, innerR),
        (7, 14, outerR), (8, 23, innerR),  (9, 13, outerR),
        (11, 12, 0),     (12, 0, 0),       (12, 7, outerR),
        (12, 8, innerR), (13, 19, innerR), (14, 16, circleR),
        (14, 20, circleR),(15, 5, outerR), (15, 6, innerR),
        (16, 17, circleR),(16, 18, innerR),(17, 15, innerR),
        (17, 16, circleR),(17, 20, circleR),(18, 11, kinkR),
        (19, 17, innerR),(20, 22, outerR), (21, 16, innerR),
        (22, 9, outerR), (22, 10, outerR), (23, 21, innerR),
    ]

    # Track SCSPath results for edges we care about
    route_edges = {(21, 16), (16, 18), (18, 11), (11, 12), (12, 8)}
    edge_scs_results = {}

    for (fr, to, r) in edgeConfigs:
        scs = rm.add_edge(fr, to, r)
        if (fr, to) in route_edges:
            edge_scs_results[(fr, to)] = (scs, r)

    # Spawn node (node 24) - matches reference: -44.7 % (2*pi) = 5.5655 rad
    hub_heading = math.fmod(-44.7, 2.0 * math.pi)
    if hub_heading < 0:
        hub_heading += 2.0 * math.pi
    rm.add_node(-1.205, -0.83, hub_heading)
    rm.add_edge(24, 2, 0.0)
    rm.add_edge(10, 24, 1.48202)   # reference radius
    rm.add_edge(24, 1, 0.866326)   # reference radius

    return rm, edge_scs_results

# --------------------------------------------------------------------------
# Main analysis
# --------------------------------------------------------------------------

def main():
    rm, edge_scs_results = build_sdcs_roadmap()

    # Report per-edge SCSPath results
    print("=" * 70)
    print("PER-EDGE SCSPath RESULTS for pickup_to_dropoff")
    print("=" * 70)

    edge_order = [(21, 16), (16, 18), (18, 11), (11, 12), (12, 8)]
    for (fr, to) in edge_order:
        scs, r = edge_scs_results[(fr, to)]
        n_pts = len(scs['x'])
        print(f"\n  Edge {fr}->{to}:  radius={r:.4f}  valid={scs['valid']}  "
              f"length={scs['length']:.4f}m  points={n_pts}")

        if n_pts >= 2:
            steps = []
            for i in range(1, n_pts):
                d = math.hypot(scs['x'][i] - scs['x'][i-1],
                               scs['y'][i] - scs['y'][i-1])
                steps.append(d)
            max_step = max(steps) if steps else 0
            min_step = min(steps) if steps else 0
            mean_step = np.mean(steps) if steps else 0
            print(f"    Step sizes: min={min_step:.5f}  max={max_step:.5f}  "
                  f"mean={mean_step:.5f}")
            jumps = [s for s in steps if s > 0.05]
            if jumps:
                print(f"    *** {len(jumps)} LARGE JUMPS (>0.05m): "
                      f"max={max(jumps):.5f}")
            else:
                print(f"    No large jumps (>0.05m)")

        # Check start/end vs node positions
        fr_node = rm.nodes[fr]
        to_node = rm.nodes[to]
        if n_pts > 0:
            d_start = math.hypot(scs['x'][0] - fr_node.pose[0],
                                 scs['y'][0] - fr_node.pose[1])
            d_end = math.hypot(scs['x'][-1] - to_node.pose[0],
                               scs['y'][-1] - to_node.pose[1])
            print(f"    Gap: start_to_from_node={d_start:.5f}  "
                  f"end_to_to_node={d_end:.5f}")

    # Generate full path using generate_path (which calls find_shortest_path)
    print("\n" + "=" * 70)
    print("FULL PATH GENERATION: generate_path([21, 16, 18, 11, 12, 8])")
    print("=" * 70)

    node_seq = [21, 16, 18, 11, 12, 8]
    path = rm.generate_path(node_seq)

    if path is None:
        print("  *** PATH GENERATION FAILED (returned None) ***")
        return

    px, py = path
    n_total = len(px)
    print(f"\n  Total points: {n_total}")

    # Compute total length
    total_length = 0.0
    steps = []
    for i in range(1, n_total):
        d = math.hypot(px[i] - px[i-1], py[i] - py[i-1])
        steps.append(d)
        total_length += d

    print(f"  Total path length: {total_length:.4f} m")

    if steps:
        steps_arr = np.array(steps)
        print(f"\n  Step size statistics:")
        print(f"    min:    {steps_arr.min():.6f} m")
        print(f"    max:    {steps_arr.max():.6f} m")
        print(f"    mean:   {steps_arr.mean():.6f} m")
        print(f"    median: {np.median(steps_arr):.6f} m")
        print(f"    std:    {steps_arr.std():.6f} m")

        # Large jumps
        jump_threshold = 0.05
        jump_indices = np.where(steps_arr > jump_threshold)[0]
        print(f"\n  Large step jumps (>{jump_threshold}m): {len(jump_indices)}")
        if len(jump_indices) > 0:
            for idx in jump_indices:
                print(f"    Index {idx}: step={steps_arr[idx]:.5f}m  "
                      f"from ({px[idx]:.4f},{py[idx]:.4f}) "
                      f"to ({px[idx+1]:.4f},{py[idx+1]:.4f})")

        # Also check for very small steps (potential duplicates)
        tiny_steps = np.where(steps_arr < 1e-6)[0]
        print(f"\n  Near-zero steps (<1e-6m): {len(tiny_steps)}")

    # Now do attach_endpoints as RoadGraph does
    print("\n" + "=" * 70)
    print("AFTER attach_endpoints (prepend PICKUP, append DROPOFF)")
    print("=" * 70)

    PICKUP_X, PICKUP_Y = 0.125, 4.395
    DROPOFF_X, DROPOFF_Y = -0.905, 0.800

    ds = 0.01
    rx, ry = list(px), list(py)

    def interpolate_gap(x1, y1, x2, y2, ds_):
        gx, gy = [], []
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist < ds_ * 1.5:
            return gx, gy
        n = max(int(dist / ds_), 2)
        for i in range(1, n):
            t = i / n
            gx.append(x1 + t * (x2 - x1))
            gy.append(y1 + t * (y2 - y1))
        return gx, gy

    # Prepend PICKUP
    if math.hypot(rx[0] - PICKUP_X, ry[0] - PICKUP_Y) > 0.02:
        gx, gy = interpolate_gap(PICKUP_X, PICKUP_Y, rx[0], ry[0], ds)
        nx = [PICKUP_X] + gx + rx
        ny = [PICKUP_Y] + gy + ry
        rx, ry = nx, ny

    # Append DROPOFF
    if math.hypot(rx[-1] - DROPOFF_X, ry[-1] - DROPOFF_Y) > 0.02:
        gx, gy = interpolate_gap(rx[-1], ry[-1], DROPOFF_X, DROPOFF_Y, ds)
        rx.extend(gx)
        ry.extend(gy)
        rx.append(DROPOFF_X)
        ry.append(DROPOFF_Y)

    n_final = len(rx)
    print(f"  Total points (with endpoints): {n_final}")

    steps_final = []
    total_len_final = 0.0
    for i in range(1, n_final):
        d = math.hypot(rx[i] - rx[i-1], ry[i] - ry[i-1])
        steps_final.append(d)
        total_len_final += d

    print(f"  Total path length: {total_len_final:.4f} m")

    if steps_final:
        sf = np.array(steps_final)
        print(f"\n  Step size statistics (with endpoints):")
        print(f"    min:    {sf.min():.6f} m")
        print(f"    max:    {sf.max():.6f} m")
        print(f"    mean:   {sf.mean():.6f} m")
        print(f"    median: {np.median(sf):.6f} m")
        print(f"    std:    {sf.std():.6f} m")

        jump_indices_final = np.where(sf > jump_threshold)[0]
        print(f"\n  Large step jumps (>{jump_threshold}m): {len(jump_indices_final)}")
        if len(jump_indices_final) > 0:
            for idx in jump_indices_final:
                print(f"    Index {idx}: step={sf[idx]:.5f}m  "
                      f"from ({rx[idx]:.4f},{ry[idx]:.4f}) "
                      f"to ({rx[idx+1]:.4f},{ry[idx+1]:.4f})")

    # ---------- Plotting ----------

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Full path with nodes
    ax = axes[0, 0]
    ax.plot(rx, ry, 'b-', linewidth=0.5, alpha=0.7, label='Path')
    ax.plot(rx[0], ry[0], 'go', markersize=10, label='Start (Pickup)')
    ax.plot(rx[-1], ry[-1], 'rs', markersize=10, label='End (Dropoff)')
    for nid in node_seq:
        n = rm.nodes[nid]
        ax.plot(n.pose[0], n.pose[1], 'k^', markersize=8)
        ax.annotate(f'N{nid}', (n.pose[0], n.pose[1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
        # Draw heading arrow
        dx = 0.05 * math.cos(n.pose[2])
        dy = 0.05 * math.sin(n.pose[2])
        ax.arrow(n.pose[0], n.pose[1], dx, dy, head_width=0.01,
                 head_length=0.005, fc='red', ec='red')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('pickup_to_dropoff Path (QLabs frame)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 2: Step sizes along path
    ax = axes[0, 1]
    if steps_final:
        ax.plot(range(len(steps_final)), steps_final, 'b-', linewidth=0.5)
        ax.axhline(y=0.05, color='r', linestyle='--', label='Jump threshold (0.05m)')
        ax.axhline(y=0.01, color='g', linestyle='--', alpha=0.5, label='Step size (0.01m)')
        # Mark large jumps
        if len(jump_indices_final) > 0:
            ax.plot(jump_indices_final, sf[jump_indices_final], 'ro', markersize=5,
                    label=f'{len(jump_indices_final)} jumps > 0.05m')
    ax.set_xlabel('Point index')
    ax.set_ylabel('Step size (m)')
    ax.set_title('Step Sizes Along Path')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-edge paths colored differently
    ax = axes[1, 0]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    for i, (fr, to) in enumerate(edge_order):
        scs, r = edge_scs_results[(fr, to)]
        if scs['x']:
            ax.plot(scs['x'], scs['y'], color=colors[i], linewidth=1.5,
                    label=f'{fr}->{to} (r={r:.3f})')
    for nid in node_seq:
        n = rm.nodes[nid]
        ax.plot(n.pose[0], n.pose[1], 'k^', markersize=8)
        ax.annotate(f'N{nid}', (n.pose[0], n.pose[1]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Per-Edge SCS Paths (colored)')
    ax.set_aspect('equal')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot 4: Step sizes per edge (stacked)
    ax = axes[1, 1]
    offset = 0
    for i, (fr, to) in enumerate(edge_order):
        scs, r = edge_scs_results[(fr, to)]
        if len(scs['x']) >= 2:
            edge_steps = []
            for j in range(1, len(scs['x'])):
                d = math.hypot(scs['x'][j] - scs['x'][j-1],
                               scs['y'][j] - scs['y'][j-1])
                edge_steps.append(d)
            indices = range(offset, offset + len(edge_steps))
            ax.plot(list(indices), edge_steps, color=colors[i], linewidth=0.8,
                    label=f'{fr}->{to}')
            offset += len(edge_steps)
    ax.axhline(y=0.05, color='r', linestyle='--', alpha=0.5, label='Jump threshold')
    ax.axhline(y=0.01, color='g', linestyle='--', alpha=0.3, label='Step size')
    ax.set_xlabel('Cumulative point index')
    ax.set_ylabel('Step size (m)')
    ax.set_title('Per-Edge Step Sizes')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/stephen/quanser-acc/scspath_analysis.png', dpi=150)
    print(f"\n  Plot saved to /home/stephen/quanser-acc/scspath_analysis.png")

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    invalid_edges = [(fr, to) for (fr, to) in edge_order
                     if not edge_scs_results[(fr, to)][0]['valid']]
    if invalid_edges:
        print(f"  INVALID edges: {invalid_edges}")
    else:
        print(f"  All {len(edge_order)} edges have valid=True")

    if steps_final:
        n_jumps = int(np.sum(sf > 0.05))
        print(f"  Total path points: {n_final}")
        print(f"  Total path length: {total_len_final:.4f} m")
        print(f"  Large jumps (>0.05m): {n_jumps}")
        if n_jumps == 0:
            print(f"  RESULT: Path is CLEAN - no chaotic waypoints detected")
        else:
            print(f"  RESULT: Path has {n_jumps} PROBLEMATIC jumps")

if __name__ == '__main__':
    main()
