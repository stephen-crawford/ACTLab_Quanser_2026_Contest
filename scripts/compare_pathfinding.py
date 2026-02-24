#!/usr/bin/env python3
"""
Pathfinding algorithm comparison for the SDCS road network.

Compares A*, D* with traffic weights, Dijkstra, and Bidirectional A*
on the competition road graph. Generates visualizations and metrics.

No ROS2 dependency — runs standalone.

Usage:
    python3 scripts/compare_pathfinding.py
"""

import sys
import os
import time
import heapq
import copy
import math
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Add parent directory so we can import road_graph
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'acc_stage1_mission'))
from road_graph import SDCSRoadMap, RoadMap, RoadMapNode, SCSPath

OUTPUT_DIR = Path(__file__).parent / 'pathfinding_results'
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Traffic penalty edge indices (ported from reference repo)
# These edges cross traffic lights or crosswalks.
# ---------------------------------------------------------------------------
TRAFFIC_PENALTY_EDGE_INDICES = [1, 2, 7, 8, 11, 12, 15, 17, 20, 22, 23, 24]
TRAFFIC_PENALTY = 20.0  # +10 crosswalk + +10 traffic light in reference


# ---------------------------------------------------------------------------
# Algorithm implementations
# ---------------------------------------------------------------------------

def find_astar_path(roadmap: RoadMap, start_idx: int, goal_idx: int):
    """Standard A* (already in RoadMap, re-implemented here for metrics)."""
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    open_set = []
    closed_set = set()
    h = np.linalg.norm(goal.pose[:2, :] - start.pose[:2, :])
    heapq.heappush(open_set, (h, id(start), start))

    g_score = {node: float('inf') for node in roadmap.nodes}
    g_score[start] = 0
    came_from = {node: None for node in roadmap.nodes}
    nodes_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            # Reconstruct path
            path = goal.pose[:2, :]
            node_seq = [roadmap.nodes.index(goal)]
            node = goal
            while True:
                prev_node, edge = came_from[node]
                if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                    path = np.hstack((prev_node.pose[:2, :], edge.waypoints, path))
                else:
                    path = np.hstack((prev_node.pose[:2, :], path))
                node_seq.insert(0, roadmap.nodes.index(prev_node))
                node = prev_node
                if came_from[node] is None:
                    break
            return path, nodes_expanded, node_seq

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        for edge in current.outEdges:
            neighbor = edge.toNode
            if neighbor in closed_set:
                continue
            if edge.length is None:
                continue
            tentative_g = g_score[current] + edge.length
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = (current, edge)
                g_score[neighbor] = tentative_g
                h_score = np.linalg.norm(goal.pose[:2, :] - neighbor.pose[:2, :])
                heapq.heappush(open_set, (tentative_g + h_score, id(neighbor), neighbor))

    return None, nodes_expanded, []


def find_dijkstra_path(roadmap: RoadMap, start_idx: int, goal_idx: int):
    """Dijkstra's algorithm (A* with heuristic=0). Optimal baseline."""
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    open_set = []
    closed_set = set()
    heapq.heappush(open_set, (0.0, id(start), start))

    g_score = {node: float('inf') for node in roadmap.nodes}
    g_score[start] = 0
    came_from = {node: None for node in roadmap.nodes}
    nodes_expanded = 0

    while open_set:
        cost, _, current = heapq.heappop(open_set)

        if current == goal:
            path = goal.pose[:2, :]
            node_seq = [roadmap.nodes.index(goal)]
            node = goal
            while True:
                prev_node, edge = came_from[node]
                if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                    path = np.hstack((prev_node.pose[:2, :], edge.waypoints, path))
                else:
                    path = np.hstack((prev_node.pose[:2, :], path))
                node_seq.insert(0, roadmap.nodes.index(prev_node))
                node = prev_node
                if came_from[node] is None:
                    break
            return path, nodes_expanded, node_seq

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        for edge in current.outEdges:
            neighbor = edge.toNode
            if neighbor in closed_set:
                continue
            if edge.length is None:
                continue
            tentative_g = g_score[current] + edge.length
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = (current, edge)
                g_score[neighbor] = tentative_g
                heapq.heappush(open_set, (tentative_g, id(neighbor), neighbor))

    return None, nodes_expanded, []


def find_dstar_weighted_path(roadmap: RoadMap, start_idx: int, goal_idx: int):
    """
    D* with traffic weights (ported from reference find_Dstar_path_weight).

    Backward search from goal to start. Adds penalties on traffic/crosswalk edges.
    """
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    # Build edge index lookup
    edge_to_idx = {}
    for i, edge in enumerate(roadmap.edges):
        edge_to_idx[id(edge)] = i

    # Save original lengths and add penalties
    original_lengths = {}
    for i, edge in enumerate(roadmap.edges):
        original_lengths[id(edge)] = edge.length
        if i in TRAFFIC_PENALTY_EDGE_INDICES and edge.length is not None:
            edge.length += TRAFFIC_PENALTY

    try:
        # Backward search: expand from goal, search toward start
        # Using incoming edges (reversed graph direction)
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0.0, id(goal), goal))

        g_score = {node: float('inf') for node in roadmap.nodes}
        g_score[goal] = 0
        rhs = {node: float('inf') for node in roadmap.nodes}
        rhs[goal] = 0
        came_from = {node: None for node in roadmap.nodes}
        nodes_expanded = 0

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == start:
                # Reconstruct path (forward direction: start -> goal)
                path = start.pose[:2, :]
                node_seq = [start_idx]
                node = start
                while node != goal:
                    next_node, edge = came_from[node]
                    if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                        path = np.hstack((path, edge.waypoints, next_node.pose[:2, :]))
                    else:
                        path = np.hstack((path, next_node.pose[:2, :]))
                    node_seq.append(roadmap.nodes.index(next_node))
                    node = next_node
                return path, nodes_expanded, node_seq

            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            # Backward expansion: iterate incoming edges
            for edge in current.inEdges:
                neighbor = edge.fromNode  # predecessor in original graph
                if neighbor in closed_set:
                    continue
                if edge.length is None:
                    continue

                tentative_g = g_score[current] + edge.length
                if tentative_g < g_score[neighbor]:
                    # Store forward direction: from neighbor, follow edge to current
                    came_from[neighbor] = (current, edge)
                    g_score[neighbor] = tentative_g
                    rhs[neighbor] = tentative_g
                    h_score = np.linalg.norm(start.pose[:2, :] - neighbor.pose[:2, :])
                    heapq.heappush(open_set, (tentative_g + h_score, id(neighbor), neighbor))

    finally:
        # Restore original edge lengths
        for edge in roadmap.edges:
            edge.length = original_lengths[id(edge)]

    return None, nodes_expanded, []


def find_bidirectional_astar_path(roadmap: RoadMap, start_idx: int, goal_idx: int):
    """Bidirectional A*: search from both ends, meet in the middle."""
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    # Forward search structures
    fwd_open = []
    fwd_closed = set()
    fwd_g = {node: float('inf') for node in roadmap.nodes}
    fwd_g[start] = 0
    fwd_came_from = {node: None for node in roadmap.nodes}
    h_fwd = np.linalg.norm(goal.pose[:2, :] - start.pose[:2, :])
    heapq.heappush(fwd_open, (h_fwd, id(start), start))

    # Backward search structures
    bwd_open = []
    bwd_closed = set()
    bwd_g = {node: float('inf') for node in roadmap.nodes}
    bwd_g[goal] = 0
    bwd_came_from = {node: None for node in roadmap.nodes}
    h_bwd = np.linalg.norm(start.pose[:2, :] - goal.pose[:2, :])
    heapq.heappush(bwd_open, (h_bwd, id(goal), goal))

    nodes_expanded = 0
    best_cost = float('inf')
    meeting_node = None

    while fwd_open or bwd_open:
        # Check termination: both frontiers' minimum f-values exceed best known cost
        fwd_min = fwd_open[0][0] if fwd_open else float('inf')
        bwd_min = bwd_open[0][0] if bwd_open else float('inf')
        if fwd_min >= best_cost and bwd_min >= best_cost:
            break

        # Expand forward
        if fwd_open and fwd_min <= bwd_min:
            _, _, current = heapq.heappop(fwd_open)
            if current not in fwd_closed:
                fwd_closed.add(current)
                nodes_expanded += 1
                for edge in current.outEdges:
                    neighbor = edge.toNode
                    if neighbor in fwd_closed or edge.length is None:
                        continue
                    tent_g = fwd_g[current] + edge.length
                    if tent_g < fwd_g[neighbor]:
                        fwd_came_from[neighbor] = (current, edge)
                        fwd_g[neighbor] = tent_g
                        h = np.linalg.norm(goal.pose[:2, :] - neighbor.pose[:2, :])
                        heapq.heappush(fwd_open, (tent_g + h, id(neighbor), neighbor))
                        # Check if backward has visited this node
                        if neighbor in bwd_closed:
                            total = tent_g + bwd_g[neighbor]
                            if total < best_cost:
                                best_cost = total
                                meeting_node = neighbor
        # Expand backward
        elif bwd_open:
            _, _, current = heapq.heappop(bwd_open)
            if current not in bwd_closed:
                bwd_closed.add(current)
                nodes_expanded += 1
                for edge in current.inEdges:
                    neighbor = edge.fromNode
                    if neighbor in bwd_closed or edge.length is None:
                        continue
                    tent_g = bwd_g[current] + edge.length
                    if tent_g < bwd_g[neighbor]:
                        bwd_came_from[neighbor] = (current, edge)
                        bwd_g[neighbor] = tent_g
                        h = np.linalg.norm(start.pose[:2, :] - neighbor.pose[:2, :])
                        heapq.heappush(bwd_open, (tent_g + h, id(neighbor), neighbor))
                        if neighbor in fwd_closed:
                            total = fwd_g[neighbor] + tent_g
                            if total < best_cost:
                                best_cost = total
                                meeting_node = neighbor

    if meeting_node is None:
        return None, nodes_expanded, []

    # Reconstruct forward half: start -> meeting_node
    fwd_path_nodes = []
    node = meeting_node
    while node != start:
        fwd_path_nodes.insert(0, node)
        if fwd_came_from[node] is None:
            return None, nodes_expanded, []
        node = fwd_came_from[node][0]
    fwd_path_nodes.insert(0, start)

    # Reconstruct backward half: meeting_node -> goal
    bwd_path_nodes = []
    node = meeting_node
    while node != goal:
        if bwd_came_from[node] is None:
            return None, nodes_expanded, []
        bwd_path_nodes.append(node)
        node = bwd_came_from[node][0]
    bwd_path_nodes.append(goal)

    # Full node sequence
    full_nodes = fwd_path_nodes + bwd_path_nodes[1:]
    node_seq = [roadmap.nodes.index(n) for n in full_nodes]

    # Generate waypoint path from node sequence
    path = full_nodes[0].pose[:2, :]
    for i in range(len(full_nodes) - 1):
        from_node = full_nodes[i]
        to_node = full_nodes[i + 1]
        # Find the edge connecting them
        edge = None
        for e in from_node.outEdges:
            if e.toNode == to_node:
                edge = e
                break
        if edge is None:
            return None, nodes_expanded, []
        if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
            path = np.hstack((path, edge.waypoints, to_node.pose[:2, :]))
        else:
            path = np.hstack((path, to_node.pose[:2, :]))

    return path, nodes_expanded, node_seq


# ---------------------------------------------------------------------------
# NEW ALGORITHM: Weighted A*
# ---------------------------------------------------------------------------

def find_weighted_astar_path(roadmap: RoadMap, start_idx: int, goal_idx: int,
                              epsilon: float = 1.5):
    """Weighted A*: f(n) = g(n) + epsilon * h(n).

    Bounded suboptimality: cost <= epsilon * optimal.
    """
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    open_set = []
    closed_set = set()
    h = np.linalg.norm(goal.pose[:2, :] - start.pose[:2, :])
    heapq.heappush(open_set, (epsilon * h, id(start), start))

    g_score = {node: float('inf') for node in roadmap.nodes}
    g_score[start] = 0
    came_from = {node: None for node in roadmap.nodes}
    nodes_expanded = 0

    while open_set:
        _, _, current = heapq.heappop(open_set)

        if current == goal:
            path = goal.pose[:2, :]
            node_seq = [roadmap.nodes.index(goal)]
            node = goal
            while True:
                prev_node, edge = came_from[node]
                if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                    path = np.hstack((prev_node.pose[:2, :], edge.waypoints, path))
                else:
                    path = np.hstack((prev_node.pose[:2, :], path))
                node_seq.insert(0, roadmap.nodes.index(prev_node))
                node = prev_node
                if came_from[node] is None:
                    break
            return path, nodes_expanded, node_seq

        if current in closed_set:
            continue
        closed_set.add(current)
        nodes_expanded += 1

        for edge in current.outEdges:
            neighbor = edge.toNode
            if neighbor in closed_set or edge.length is None:
                continue
            tentative_g = g_score[current] + edge.length
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = (current, edge)
                g_score[neighbor] = tentative_g
                h_score = np.linalg.norm(goal.pose[:2, :] - neighbor.pose[:2, :])
                heapq.heappush(open_set, (tentative_g + epsilon * h_score,
                                          id(neighbor), neighbor))

    return None, nodes_expanded, []


# ---------------------------------------------------------------------------
# NEW ALGORITHM: Multi-Heuristic A* (MHA*)
# ---------------------------------------------------------------------------

def _manhattan_heuristic(node, goal):
    """Manhattan distance heuristic (inadmissible on road graphs)."""
    return (abs(node.pose[0, 0] - goal.pose[0, 0]) +
            abs(node.pose[1, 0] - goal.pose[1, 0]))


def _angle_weighted_heuristic(node, goal):
    """Angle-weighted heuristic: Euclidean + heading difference penalty."""
    eucl = np.linalg.norm(goal.pose[:2, :] - node.pose[:2, :])
    # Heading difference penalty
    dth = abs(node.pose[2, 0] - goal.pose[2, 0])
    dth = min(dth, 2 * np.pi - dth)
    return eucl + 0.5 * dth


def find_mha_star_path(roadmap: RoadMap, start_idx: int, goal_idx: int,
                        w1: float = 1.5, w2: float = 1.5):
    """Multi-Heuristic A* (MHA*).

    Uses anchor search (admissible Euclidean) + 2 inadmissible heuristics
    (Manhattan, angle-weighted). Expands from inadmissible list when its
    min key <= w2 * anchor min key.

    Bound: cost <= w1 * w2 * optimal.
    """
    start = roadmap.nodes[start_idx]
    goal = roadmap.nodes[goal_idx]

    if start == goal:
        return None, 0, 0

    # Heuristics: anchor (admissible) + inadmissible
    def h_anchor(n):
        return np.linalg.norm(goal.pose[:2, :] - n.pose[:2, :])

    heuristics = [h_anchor, _manhattan_heuristic, _angle_weighted_heuristic]
    n_heuristics = len(heuristics)

    # Separate open lists for each heuristic
    open_lists = [[] for _ in range(n_heuristics)]
    g_score = {node: float('inf') for node in roadmap.nodes}
    g_score[start] = 0
    came_from = {node: None for node in roadmap.nodes}
    closed_anchor = set()
    closed_inad = set()
    nodes_expanded = 0

    # Initialize all open lists
    for i in range(n_heuristics):
        h_val = heuristics[i](start, goal) if i > 0 else heuristics[i](start)
        key = g_score[start] + (w1 if i == 0 else w1) * h_val
        heapq.heappush(open_lists[i], (key, id(start), start))

    def _key(node, i):
        h_val = heuristics[i](node, goal) if i > 0 else heuristics[i](node)
        return g_score[node] + (w1 if i == 0 else w1) * h_val

    def _expand(node):
        nonlocal nodes_expanded
        nodes_expanded += 1
        for edge in node.outEdges:
            neighbor = edge.toNode
            if edge.length is None:
                continue
            tentative_g = g_score[node] + edge.length
            if tentative_g < g_score[neighbor]:
                came_from[neighbor] = (node, edge)
                g_score[neighbor] = tentative_g
                if neighbor not in closed_anchor:
                    for i in range(n_heuristics):
                        h_val = heuristics[i](neighbor, goal) if i > 0 else heuristics[i](neighbor)
                        key = tentative_g + (w1 if i == 0 else w1) * h_val
                        heapq.heappush(open_lists[i], (key, id(neighbor), neighbor))

    def _reconstruct(node):
        path = node.pose[:2, :]
        node_seq = [roadmap.nodes.index(node)]
        while came_from[node] is not None:
            prev_node, edge = came_from[node]
            if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                path = np.hstack((prev_node.pose[:2, :], edge.waypoints, path))
            else:
                path = np.hstack((prev_node.pose[:2, :], path))
            node_seq.insert(0, roadmap.nodes.index(prev_node))
            node = prev_node
        return path, node_seq

    max_iterations = len(roadmap.nodes) * 20  # Safety limit

    for _ in range(max_iterations):
        if not open_lists[0]:
            break

        anchor_min = open_lists[0][0][0] if open_lists[0] else float('inf')

        expanded = False
        # Try inadmissible heuristics first
        for i in range(1, n_heuristics):
            while open_lists[i] and open_lists[i][0][2] in closed_inad:
                heapq.heappop(open_lists[i])

            if not open_lists[i]:
                continue

            if open_lists[i][0][0] <= w2 * anchor_min:
                # Check if goal reached
                if g_score[goal] <= open_lists[i][0][0]:
                    path, node_seq = _reconstruct(goal)
                    return path, nodes_expanded, node_seq

                _, _, s = heapq.heappop(open_lists[i])
                if s in closed_inad:
                    continue
                _expand(s)
                closed_inad.add(s)
                expanded = True
                break

        if not expanded:
            # Use anchor search
            while open_lists[0] and open_lists[0][0][2] in closed_anchor:
                heapq.heappop(open_lists[0])
            if not open_lists[0]:
                break

            if g_score[goal] <= open_lists[0][0][0]:
                path, node_seq = _reconstruct(goal)
                return path, nodes_expanded, node_seq

            _, _, s = heapq.heappop(open_lists[0])
            if s in closed_anchor:
                continue
            _expand(s)
            closed_anchor.add(s)

    # Check if goal was reached
    if g_score[goal] < float('inf'):
        path, node_seq = _reconstruct(goal)
        return path, nodes_expanded, node_seq

    return None, nodes_expanded, []


# ---------------------------------------------------------------------------
# NEW ALGORITHM: Experience-Based Planning
# ---------------------------------------------------------------------------

class ExperienceBasedPlanner:
    """Cache A* solutions; on repeat queries, validate and return instantly.

    Demonstrates the taxi mission's repetitive nature — once a route is found,
    subsequent calls return in O(n) path validation time.
    """

    def __init__(self):
        self._cache = {}  # (start_idx, goal_idx) -> (path, node_seq)
        self.cache_hits = 0
        self.cache_misses = 0

    def find_path(self, roadmap: RoadMap, start_idx: int, goal_idx: int):
        """Find path with experience caching."""
        key = (start_idx, goal_idx)

        if key in self._cache:
            cached_path, cached_seq = self._cache[key]
            # Validate cached path: check all edges still exist and have same length
            if self._validate_cached(roadmap, cached_seq):
                self.cache_hits += 1
                return cached_path.copy(), 0, cached_seq  # 0 nodes expanded
            else:
                del self._cache[key]

        # Cache miss — run standard A*
        self.cache_misses += 1
        path, nodes_expanded, node_seq = find_astar_path(roadmap, start_idx, goal_idx)

        if path is not None:
            self._cache[key] = (path.copy(), list(node_seq))

        return path, nodes_expanded, node_seq

    @staticmethod
    def _validate_cached(roadmap: RoadMap, node_seq: list) -> bool:
        """Validate that cached node sequence is still traversable."""
        for i in range(len(node_seq) - 1):
            from_node = roadmap.nodes[node_seq[i]]
            to_node = roadmap.nodes[node_seq[i + 1]]
            edge_found = False
            for edge in from_node.outEdges:
                if edge.toNode == to_node and edge.length is not None:
                    edge_found = True
                    break
            if not edge_found:
                return False
        return True


# ---------------------------------------------------------------------------
# Road Boundary Loader (for continuous-space algorithms)
# ---------------------------------------------------------------------------

class RoadBoundaryLoader:
    """Parse road_boundaries.yaml to provide spatial road queries.

    Provides:
        is_on_road(x, y) -> bool
        signed_distance_to_road(x, y) -> (float, normal_vec)
        sample_road_point() -> (x, y)
    """

    def __init__(self, yaml_path: str = None):
        self.segments = []
        self.circles = []
        self.default_width = 0.30
        self._bounds = None  # (x_min, x_max, y_min, y_max)

        if yaml_path is None:
            yaml_path = os.path.join(
                os.path.dirname(__file__), '..', 'config', 'road_boundaries.yaml')

        if HAS_YAML and os.path.isfile(yaml_path):
            self._load(yaml_path)
        # Always also add graph edges for full road coverage
        self._build_from_graph()

    def _load(self, yaml_path: str):
        """Load road boundaries from YAML config."""
        with open(yaml_path, 'r') as f:
            cfg = yaml.safe_load(f)

        self.default_width = cfg.get('road_width', 0.30)

        for seg in cfg.get('road_segments', []):
            if seg.get('type') == 'circular':
                center = seg['center']
                self.circles.append({
                    'cx': center['x'],
                    'cy': center['y'],
                    'radius': seg.get('radius', 0.5),
                    'width': seg.get('width', self.default_width),
                })
            else:
                points = []
                for pt in seg.get('centerline', []):
                    points.append({
                        'x': pt['x'], 'y': pt['y'],
                        'wl': pt.get('width_left', self.default_width),
                        'wr': pt.get('width_right', self.default_width),
                    })
                if points:
                    self.segments.append(points)

        self._compute_bounds()

    def _build_from_graph(self):
        """Build road boundaries from SDCSRoadMap edges.

        Uses slightly wider width (0.35m) than yaml segments to ensure
        all graph nodes are considered on-road for sampling algorithms.
        """
        roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
        graph_width = self.default_width + 0.05  # Slightly wider for coverage

        # Add each node as a small circular road zone to cover junction gaps
        for node in roadmap.nodes:
            self.circles.append({
                'cx': float(node.pose[0, 0]),
                'cy': float(node.pose[1, 0]),
                'radius': 0.05,
                'width': graph_width,
            })

        for edge in roadmap.edges:
            pts = []
            if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                # Sample every 10th point
                wp = edge.waypoints
                stride = max(1, wp.shape[1] // 20)
                for j in range(0, wp.shape[1], stride):
                    pts.append({
                        'x': float(wp[0, j]),
                        'y': float(wp[1, j]),
                        'wl': graph_width,
                        'wr': graph_width,
                    })
            else:
                pts.append({
                    'x': float(edge.fromNode.pose[0, 0]),
                    'y': float(edge.fromNode.pose[1, 0]),
                    'wl': graph_width,
                    'wr': graph_width,
                })
                pts.append({
                    'x': float(edge.toNode.pose[0, 0]),
                    'y': float(edge.toNode.pose[1, 0]),
                    'wl': graph_width,
                    'wr': graph_width,
                })
            if pts:
                self.segments.append(pts)

        self._compute_bounds()

    def _compute_bounds(self):
        """Compute bounding box of all road segments."""
        xs, ys = [], []
        for seg in self.segments:
            for pt in seg:
                xs.append(pt['x'])
                ys.append(pt['y'])
        for circ in self.circles:
            xs.extend([circ['cx'] - circ['radius'], circ['cx'] + circ['radius']])
            ys.extend([circ['cy'] - circ['radius'], circ['cy'] + circ['radius']])
        if xs and ys:
            margin = 0.5
            self._bounds = (min(xs) - margin, max(xs) + margin,
                            min(ys) - margin, max(ys) + margin)
        else:
            self._bounds = (-3, 3, -3, 6)

    def signed_distance_to_road(self, x: float, y: float) -> tuple:
        """Returns (signed_distance, normal_vec).

        Negative = inside road, positive = outside road.
        Returns the most negative (deepest inside) distance across all
        road regions, so a point inside ANY region returns negative.
        """
        min_dist = float('inf')
        best_normal = np.array([0.0, 0.0])

        # Check line segments
        for seg in self.segments:
            for i in range(len(seg) - 1):
                p1 = np.array([seg[i]['x'], seg[i]['y']])
                p2 = np.array([seg[i + 1]['x'], seg[i + 1]['y']])
                width = (seg[i]['wl'] + seg[i]['wr']) / 2

                # Project point onto segment
                v = p2 - p1
                v_len = np.linalg.norm(v)
                if v_len < 1e-9:
                    continue
                v_unit = v / v_len
                t = np.clip(np.dot(np.array([x, y]) - p1, v_unit) / v_len, 0, 1)
                proj = p1 + t * v

                # Distance from projection to point
                diff = np.array([x, y]) - proj
                dist_to_center = np.linalg.norm(diff)
                signed = dist_to_center - width

                # Keep the most negative (deepest inside any road region)
                if signed < min_dist:
                    min_dist = signed
                    if dist_to_center > 1e-9:
                        best_normal = diff / dist_to_center
                    else:
                        # On centerline
                        best_normal = np.array([-v_unit[1], v_unit[0]])

        # Check circles
        for circ in self.circles:
            dx = x - circ['cx']
            dy = y - circ['cy']
            dist_to_center = math.sqrt(dx * dx + dy * dy)
            road_radius = circ['radius'] + circ.get('width', self.default_width)
            signed = dist_to_center - road_radius

            if signed < min_dist:
                min_dist = signed
                if dist_to_center > 1e-9:
                    best_normal = np.array([dx, dy]) / dist_to_center
                else:
                    best_normal = np.array([1.0, 0.0])

        return min_dist, best_normal

    def is_on_road(self, x: float, y: float) -> bool:
        """Check if point (x, y) is on any road segment."""
        dist, _ = self.signed_distance_to_road(x, y)
        return dist <= 0

    def sample_road_point(self) -> tuple:
        """Sample a random point on the road (biased sampling for RRT*)."""
        # 80% chance: sample near a road segment
        if random.random() < 0.8 and self.segments:
            seg = random.choice(self.segments)
            pt = random.choice(seg)
            # Add small random offset within road width
            w = pt.get('wl', self.default_width)
            dx = random.gauss(0, w * 0.4)
            dy = random.gauss(0, w * 0.4)
            return (pt['x'] + dx, pt['y'] + dy)
        else:
            # Uniform sampling within bounds
            if self._bounds:
                x = random.uniform(self._bounds[0], self._bounds[1])
                y = random.uniform(self._bounds[2], self._bounds[3])
                return (x, y)
            return (random.uniform(-2, 2), random.uniform(-2, 5))

    def get_bounds(self):
        """Return (x_min, x_max, y_min, y_max)."""
        return self._bounds or (-3, 3, -3, 6)


# ---------------------------------------------------------------------------
# NEW ALGORITHM: RRT*
# ---------------------------------------------------------------------------

def find_rrt_star_path(roadmap: RoadMap, start_idx: int, goal_idx: int,
                       road_boundaries: RoadBoundaryLoader = None,
                       max_iter: int = 2000, step_size: float = 0.1,
                       goal_radius: float = 0.15):
    """RRT* — sampling-based in continuous 2D plane.

    80% road-biased sampling. Rewiring radius scales with log(n)/n.
    Returns (path_2xN, nodes_expanded, []) where path follows continuous space.
    Also returns tree for visualization.
    """
    start_pos = roadmap.nodes[start_idx].pose[:2, 0].copy()
    goal_pos = roadmap.nodes[goal_idx].pose[:2, 0].copy()

    if np.linalg.norm(start_pos - goal_pos) < goal_radius:
        path = np.hstack((start_pos.reshape(2, 1), goal_pos.reshape(2, 1)))
        return path, 0, []

    if road_boundaries is None:
        road_boundaries = RoadBoundaryLoader()

    # Tree storage: list of (position, parent_idx, cost)
    tree_pos = [start_pos]
    tree_parent = [-1]
    tree_cost = [0.0]
    tree_edges = []  # For visualization: (parent_idx, child_idx)

    goal_node_idx = None
    best_goal_cost = float('inf')

    for iteration in range(max_iter):
        # Sample point (biased toward road)
        if random.random() < 0.05:
            # Goal bias
            sample = goal_pos.copy()
        else:
            sx, sy = road_boundaries.sample_road_point()
            sample = np.array([sx, sy])

        # Find nearest node
        dists = [np.linalg.norm(sample - p) for p in tree_pos]
        nearest_idx = int(np.argmin(dists))
        nearest = tree_pos[nearest_idx]

        # Steer toward sample
        direction = sample - nearest
        dist = np.linalg.norm(direction)
        if dist < 1e-9:
            continue
        direction = direction / dist
        new_pos = nearest + min(step_size, dist) * direction

        # Check if new position is on road (collision-free)
        if not road_boundaries.is_on_road(new_pos[0], new_pos[1]):
            continue

        # Check if edge is collision-free (sample intermediate points)
        edge_clear = True
        n_checks = max(2, int(np.linalg.norm(new_pos - nearest) / 0.02))
        for j in range(1, n_checks):
            t = j / n_checks
            mid = nearest + t * (new_pos - nearest)
            if not road_boundaries.is_on_road(mid[0], mid[1]):
                edge_clear = False
                break
        if not edge_clear:
            continue

        new_cost = tree_cost[nearest_idx] + np.linalg.norm(new_pos - nearest)

        # RRT* rewiring: find nearby nodes
        n = len(tree_pos)
        rewire_radius = min(1.0, step_size * 3 * math.sqrt(math.log(n + 1) / (n + 1)))
        near_indices = [i for i, p in enumerate(tree_pos)
                        if np.linalg.norm(p - new_pos) < rewire_radius]

        # Choose best parent from near nodes
        best_parent = nearest_idx
        best_cost = new_cost
        for ni in near_indices:
            candidate_cost = tree_cost[ni] + np.linalg.norm(tree_pos[ni] - new_pos)
            if candidate_cost < best_cost:
                # Check edge feasibility
                mid_ok = True
                n_mid = max(2, int(np.linalg.norm(tree_pos[ni] - new_pos) / 0.02))
                for j in range(1, n_mid):
                    t = j / n_mid
                    mid = tree_pos[ni] + t * (new_pos - tree_pos[ni])
                    if not road_boundaries.is_on_road(mid[0], mid[1]):
                        mid_ok = False
                        break
                if mid_ok:
                    best_parent = ni
                    best_cost = candidate_cost

        # Add new node
        new_idx = len(tree_pos)
        tree_pos.append(new_pos.copy())
        tree_parent.append(best_parent)
        tree_cost.append(best_cost)
        tree_edges.append((best_parent, new_idx))

        # Rewire nearby nodes
        for ni in near_indices:
            rewire_cost = best_cost + np.linalg.norm(tree_pos[ni] - new_pos)
            if rewire_cost < tree_cost[ni]:
                mid_ok = True
                n_mid = max(2, int(np.linalg.norm(tree_pos[ni] - new_pos) / 0.02))
                for j in range(1, n_mid):
                    t = j / n_mid
                    mid = new_pos + t * (tree_pos[ni] - new_pos)
                    if not road_boundaries.is_on_road(mid[0], mid[1]):
                        mid_ok = False
                        break
                if mid_ok:
                    tree_parent[ni] = new_idx
                    tree_cost[ni] = rewire_cost

        # Check goal proximity
        if np.linalg.norm(new_pos - goal_pos) < goal_radius:
            if best_cost < best_goal_cost:
                best_goal_cost = best_cost
                goal_node_idx = new_idx

    if goal_node_idx is None:
        # Return None path but include tree data for visualization
        return None, max_iter, [], tree_pos, tree_edges

    # Reconstruct path
    path_points = [goal_pos]
    idx = goal_node_idx
    while idx != -1:
        path_points.append(tree_pos[idx])
        idx = tree_parent[idx]
    path_points.reverse()

    # Convert to 2xN
    path = np.array(path_points).T

    return path, max_iter, [], tree_pos, tree_edges


# ---------------------------------------------------------------------------
# NEW ALGORITHM: CHOMP
# ---------------------------------------------------------------------------

def find_chomp_path(roadmap: RoadMap, start_idx: int, goal_idx: int,
                    road_boundaries: RoadBoundaryLoader = None,
                    n_waypoints: int = 50, n_iterations: int = 200,
                    learning_rate: float = 0.02,
                    smoothness_weight: float = 1.0,
                    obstacle_weight: float = 10.0):
    """CHOMP — gradient-based trajectory optimization.

    Initialize with straight line, minimize smoothness + obstacle cost.
    Start and goal are fixed. Returns path and convergence history.
    """
    start_pos = roadmap.nodes[start_idx].pose[:2, 0].copy()
    goal_pos = roadmap.nodes[goal_idx].pose[:2, 0].copy()

    if road_boundaries is None:
        road_boundaries = RoadBoundaryLoader()

    # Initialize trajectory as straight line
    traj = np.zeros((n_waypoints, 2))
    for i in range(n_waypoints):
        t = i / (n_waypoints - 1)
        traj[i] = start_pos + t * (goal_pos - start_pos)

    # Pre-compute finite-difference matrix for smoothness
    # K is (n-2) x n tridiagonal: K[i, i] = -2, K[i, i+1] = 1, K[i, i-1] = 1
    n = n_waypoints
    K = np.zeros((n - 2, n))
    for i in range(n - 2):
        K[i, i] = 1
        K[i, i + 1] = -2
        K[i, i + 2] = 1

    convergence_history = []

    for iteration in range(n_iterations):
        # Compute smoothness cost: sum of squared second differences
        inner = traj[1:-1]  # Exclude fixed endpoints
        dd = np.diff(traj, n=2, axis=0)  # (n-2) x 2
        smoothness_cost = smoothness_weight * np.sum(dd ** 2)

        # Smoothness gradient for inner points
        smooth_grad = smoothness_weight * 2 * (K.T @ K @ traj)
        smooth_grad = smooth_grad[1:-1]  # Only update inner points

        # Compute obstacle cost and gradient
        obs_cost = 0.0
        obs_grad = np.zeros_like(inner)

        for i in range(len(inner)):
            x, y = inner[i]
            dist, normal = road_boundaries.signed_distance_to_road(x, y)

            if dist > 0:
                # Outside road — penalty
                obs_cost += obstacle_weight * dist * dist
                obs_grad[i] = obstacle_weight * 2 * dist * normal
            elif dist > -0.05:
                # Near road edge — small repulsive penalty
                penalty = 0.5 * obstacle_weight * (dist + 0.05) ** 2
                obs_cost += penalty
                obs_grad[i] = obstacle_weight * (dist + 0.05) * normal

        total_cost = smoothness_cost + obs_cost
        convergence_history.append(total_cost)

        # Gradient descent on inner points
        total_grad = smooth_grad + obs_grad
        grad_norm = np.linalg.norm(total_grad)
        if grad_norm > 1e-9:
            # Adaptive learning rate with gradient clipping
            effective_lr = learning_rate / max(1.0, grad_norm / 10.0)
            traj[1:-1] -= effective_lr * total_grad

    # Compute final path length
    diffs = np.diff(traj, axis=0)
    path_length = np.sum(np.sqrt(diffs[:, 0] ** 2 + diffs[:, 1] ** 2))

    # Convert to 2xN for consistency
    path = traj.T

    return path, n_iterations, [], convergence_history


# ---------------------------------------------------------------------------
# Path length computation from a node sequence (for forced routes)
# ---------------------------------------------------------------------------

def compute_forced_route_length(roadmap: RoadMap, node_seq: list):
    """Compute total length for a forced node sequence using generate_path."""
    path = roadmap.generate_path(node_seq)
    if path is None:
        return None, 0
    # Path is 2xN, compute arc length
    diffs = np.diff(path, axis=1)
    total_length = np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2))
    return path, total_length


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

ALGO_COLORS = {
    'A*': '#2196F3',
    'Dijkstra': '#4CAF50',
    'D* Weighted': '#FF9800',
    'Bidirectional A*': '#9C27B0',
    'Current (forced)': '#F44336',
    'Weighted A* (1.5)': '#00BCD4',
    'Weighted A* (2.0)': '#009688',
    'Weighted A* (3.0)': '#607D8B',
    'MHA*': '#E91E63',
    'Experience A*': '#795548',
    'RRT*': '#FF5722',
    'CHOMP': '#3F51B5',
}

ALGO_STYLES = {
    'A*': '-',
    'Dijkstra': '-',
    'D* Weighted': '-',
    'Bidirectional A*': '-',
    'Current (forced)': '--',
    'Weighted A* (1.5)': '-',
    'Weighted A* (2.0)': '-.',
    'Weighted A* (3.0)': ':',
    'MHA*': '-',
    'Experience A*': '-',
    'RRT*': '-',
    'CHOMP': '-',
}


def draw_road_network(ax, roadmap: RoadMap, alpha=0.3, show_labels=True):
    """Draw the road network on a matplotlib axes."""
    # Draw edges
    for edge in roadmap.edges:
        if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
            ax.plot(edge.waypoints[0], edge.waypoints[1],
                    color='#BDBDBD', linewidth=1.0, alpha=alpha, zorder=1)
        else:
            # Straight line fallback
            x = [edge.fromNode.pose[0, 0], edge.toNode.pose[0, 0]]
            y = [edge.fromNode.pose[1, 0], edge.toNode.pose[1, 0]]
            ax.plot(x, y, color='#BDBDBD', linewidth=1.0, alpha=alpha, zorder=1)

    # Draw nodes
    for i, node in enumerate(roadmap.nodes):
        x, y = node.pose[0, 0], node.pose[1, 0]
        ax.plot(x, y, 'o', color='#616161', markersize=5, zorder=3)
        if show_labels:
            ax.annotate(str(i), (x, y), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color='#212121', zorder=4)


def figure1_road_network(roadmap):
    """Figure 1: Full road network with labeled nodes."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    draw_road_network(ax, roadmap, alpha=0.6, show_labels=True)

    # Highlight special locations
    hub = roadmap.nodes[24].pose[:2, 0]
    ax.plot(hub[0], hub[1], 's', color='#F44336', markersize=12, zorder=5, label='Hub (node 24)')

    # Node 20 is near pickup
    n20 = roadmap.nodes[20].pose[:2, 0]
    ax.plot(n20[0], n20[1], '^', color='#2196F3', markersize=12, zorder=5, label='Node 20 (near pickup)')

    # Node 8 is near dropoff
    n8 = roadmap.nodes[8].pose[:2, 0]
    ax.plot(n8[0], n8[1], 'D', color='#4CAF50', markersize=12, zorder=5, label='Node 8 (near dropoff)')

    ax.set_title('SDCS Road Network — 25 Nodes, All Edges', fontsize=14)
    ax.set_xlabel('X (meters, QLabs frame)')
    ax.set_ylabel('Y (meters, QLabs frame)')
    ax.legend(loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig1_road_network.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig1_road_network.png'}")
    plt.close(fig)


def figure2_hub_to_pickup(roadmap, results, forced_path):
    """Figure 2: Hub→Pickup routes overlaid per algorithm."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    draw_road_network(ax, roadmap, alpha=0.2, show_labels=True)

    # Draw forced current route (dashed)
    if forced_path is not None:
        ax.plot(forced_path[0], forced_path[1],
                color=ALGO_COLORS['Current (forced)'],
                linestyle='--', linewidth=2.5, alpha=0.8,
                label='Current forced [24,1,13,19,17,20]', zorder=6)

    # Draw each algorithm's path
    for algo_name, data in results.items():
        scenario_data = data.get('24 → 20')
        if scenario_data and scenario_data['path'] is not None:
            path = scenario_data['path']
            ax.plot(path[0], path[1],
                    color=ALGO_COLORS.get(algo_name, '#000000'),
                    linestyle=ALGO_STYLES.get(algo_name, '-'),
                    linewidth=2.0, alpha=0.8,
                    label=f'{algo_name} ({scenario_data["length"]:.2f}m)',
                    zorder=7)

    ax.set_title('Hub → Pickup: Algorithm Comparison', fontsize=14)
    ax.set_xlabel('X (meters, QLabs frame)')
    ax.set_ylabel('Y (meters, QLabs frame)')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig2_hub_to_pickup.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig2_hub_to_pickup.png'}")
    plt.close(fig)


def figure3_bar_chart(results, forced_lengths):
    """Figure 3: Bar chart comparing path lengths."""
    scenarios = ['24 → 20', '21 → 8', '8 → 24']
    algos = list(results.keys()) + ['Current (forced)']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    x = np.arange(len(scenarios))
    bar_width = 0.15
    offset = -(len(algos) - 1) / 2 * bar_width

    for i, algo in enumerate(algos):
        lengths = []
        for scenario in scenarios:
            if algo == 'Current (forced)':
                lengths.append(forced_lengths.get(scenario, 0))
            else:
                data = results[algo].get(scenario, {})
                lengths.append(data.get('length', 0) if data.get('path') is not None else 0)

        color = ALGO_COLORS.get(algo, '#000000')
        bars = ax.bar(x + offset + i * bar_width, lengths, bar_width,
                      label=algo, color=color, alpha=0.85)
        # Add value labels on bars
        for bar, val in zip(bars, lengths):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.1f}', ha='center', va='bottom', fontsize=7)

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Path Length (meters)')
    ax.set_title('Path Length Comparison Across Algorithms and Scenarios', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig3_length_comparison.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig3_length_comparison.png'}")
    plt.close(fig)


def figure4_all_legs(roadmap, results):
    """Figure 4: All three mission legs per algorithm."""
    legs = [
        ('24 → 20', 'Hub → Pickup'),
        ('21 → 8', 'Pickup → Dropoff'),
        ('8 → 24', 'Dropoff → Hub'),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))

    for ax, (scenario, title) in zip(axes, legs):
        draw_road_network(ax, roadmap, alpha=0.2, show_labels=False)

        for algo_name, data in results.items():
            scenario_data = data.get(scenario)
            if scenario_data and scenario_data['path'] is not None:
                path = scenario_data['path']
                ax.plot(path[0], path[1],
                        color=ALGO_COLORS.get(algo_name, '#000000'),
                        linestyle=ALGO_STYLES.get(algo_name, '-'),
                        linewidth=2.0, alpha=0.7,
                        label=f'{algo_name}')

        ax.set_title(title, fontsize=12)
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle('All Mission Legs — Algorithm Routes', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig4_all_legs.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_DIR / 'fig4_all_legs.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# NEW Figures 5-10
# ---------------------------------------------------------------------------

def figure5_weighted_astar_tradeoff(roadmap, scenario='24 → 20'):
    """Figure 5: Weighted A* epsilon tradeoff — path length + nodes expanded vs epsilon."""
    start, goal = 24, 20
    epsilons = [1.0, 1.5, 2.0, 3.0, 5.0]
    lengths = []
    expanded = []

    for eps in epsilons:
        path, n_exp, _ = find_weighted_astar_path(roadmap, start, goal, epsilon=eps)
        if path is not None:
            diffs = np.diff(path, axis=1)
            length = float(np.sum(np.sqrt(diffs[0] ** 2 + diffs[1] ** 2)))
        else:
            length = 0
        lengths.append(length)
        expanded.append(n_exp)

    fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    color1 = '#2196F3'
    color2 = '#FF9800'

    ax1.set_xlabel('Epsilon (inflation factor)')
    ax1.set_ylabel('Path Length (m)', color=color1)
    line1 = ax1.plot(epsilons, lengths, 'o-', color=color1, linewidth=2,
                     markersize=8, label='Path Length')
    ax1.tick_params(axis='y', labelcolor=color1)

    # Optimal reference line
    if lengths and lengths[0] > 0:
        ax1.axhline(y=lengths[0], color=color1, linestyle=':', alpha=0.3,
                     label=f'Optimal (eps=1.0): {lengths[0]:.3f}m')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Nodes Expanded', color=color2)
    line2 = ax2.plot(epsilons, expanded, 's--', color=color2, linewidth=2,
                     markersize=8, label='Nodes Expanded')
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='center right')

    # Annotate bounds
    for i, eps in enumerate(epsilons):
        if lengths[i] > 0 and lengths[0] > 0:
            ratio = lengths[i] / lengths[0]
            ax1.annotate(f'{ratio:.2f}x', (eps, lengths[i]),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=8, ha='center')

    ax1.set_title(f'Weighted A* Tradeoff ({scenario}): Quality vs Speed', fontsize=14)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig5_weighted_astar_tradeoff.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig5_weighted_astar_tradeoff.png'}")
    plt.close(fig)


def figure6_experience_speedup(roadmap, scenarios):
    """Figure 6: Experience-based speedup — first-call vs cached-call time bars."""
    planner = ExperienceBasedPlanner()

    first_times = {}
    cached_times = {}

    for scenario_name, (start, goal) in scenarios.items():
        # First call (cache miss — full A*)
        t0 = time.perf_counter()
        planner.find_path(roadmap, start, goal)
        first_times[scenario_name] = (time.perf_counter() - t0) * 1000

        # Second call (cache hit — O(n) validation only)
        t0 = time.perf_counter()
        planner.find_path(roadmap, start, goal)
        cached_times[scenario_name] = (time.perf_counter() - t0) * 1000

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x = np.arange(len(scenarios))
    bar_width = 0.3

    first_vals = [first_times[s] for s in scenarios]
    cached_vals = [cached_times[s] for s in scenarios]

    bars1 = ax.bar(x - bar_width / 2, first_vals, bar_width,
                   label='First Call (A*)', color='#F44336', alpha=0.85)
    bars2 = ax.bar(x + bar_width / 2, cached_vals, bar_width,
                   label='Cached Call', color='#4CAF50', alpha=0.85)

    # Add speedup labels
    for i in range(len(scenarios)):
        if cached_vals[i] > 0:
            speedup = first_vals[i] / cached_vals[i]
            ax.text(x[i] + bar_width / 2, cached_vals[i] + 0.01,
                    f'{speedup:.0f}x', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#4CAF50')

    ax.set_xlabel('Scenario')
    ax.set_ylabel('Time (ms)')
    ax.set_title('Experience-Based Planning: First Call vs Cached Call', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(list(scenarios.keys()))
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig6_experience_speedup.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig6_experience_speedup.png'}")
    plt.close(fig)


def figure7_continuous_vs_graph(roadmap, astar_path, rrt_result, chomp_result,
                                road_boundaries):
    """Figure 7: Continuous vs graph paths — A*, RRT*, CHOMP overlaid with road shading."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw road boundary shading
    bounds = road_boundaries.get_bounds()
    if bounds:
        x_range = np.linspace(bounds[0], bounds[1], 100)
        y_range = np.linspace(bounds[2], bounds[3], 100)
        road_mask = np.zeros((len(y_range), len(x_range)))
        for yi, yv in enumerate(y_range):
            for xi, xv in enumerate(x_range):
                if road_boundaries.is_on_road(xv, yv):
                    road_mask[yi, xi] = 1.0
        ax.contourf(x_range, y_range, road_mask, levels=[0.5, 1.5],
                    colors=['#E8F5E9'], alpha=0.4, zorder=0)
        ax.contour(x_range, y_range, road_mask, levels=[0.5],
                   colors=['#81C784'], linewidths=0.5, alpha=0.5, zorder=0)

    # Draw road network
    draw_road_network(ax, roadmap, alpha=0.2, show_labels=True)

    # A* path (graph-based)
    if astar_path is not None:
        ax.plot(astar_path[0], astar_path[1], '-', color='#2196F3',
                linewidth=2.5, alpha=0.9, label='A* (graph)', zorder=5)

    # RRT* path
    rrt_path = rrt_result[0] if rrt_result and rrt_result[0] is not None else None
    if rrt_path is not None:
        ax.plot(rrt_path[0], rrt_path[1], '-', color='#FF5722',
                linewidth=2.5, alpha=0.9, label='RRT* (continuous)', zorder=6)

    # CHOMP path
    chomp_path = chomp_result[0] if chomp_result and chomp_result[0] is not None else None
    if chomp_path is not None:
        ax.plot(chomp_path[0], chomp_path[1], '-', color='#3F51B5',
                linewidth=2.5, alpha=0.9, label='CHOMP (continuous)', zorder=7)

    # Mark start/goal
    start_pos = roadmap.nodes[24].pose[:2, 0]
    goal_pos = roadmap.nodes[20].pose[:2, 0]
    ax.plot(start_pos[0], start_pos[1], 's', color='#F44336', markersize=15,
            zorder=10, label='Start (Hub)')
    ax.plot(goal_pos[0], goal_pos[1], '*', color='#4CAF50', markersize=18,
            zorder=10, label='Goal (Pickup)')

    ax.set_title('Graph-Based vs Continuous-Space Paths (Hub → Pickup)', fontsize=14)
    ax.set_xlabel('X (meters, QLabs frame)')
    ax.set_ylabel('Y (meters, QLabs frame)')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig7_continuous_vs_graph.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig7_continuous_vs_graph.png'}")
    plt.close(fig)


def figure8_rrt_tree(roadmap, rrt_result, road_boundaries):
    """Figure 8: RRT* tree visualization — full tree gray, solution bold."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Draw road boundary shading
    bounds = road_boundaries.get_bounds()
    if bounds:
        x_range = np.linspace(bounds[0], bounds[1], 80)
        y_range = np.linspace(bounds[2], bounds[3], 80)
        road_mask = np.zeros((len(y_range), len(x_range)))
        for yi, yv in enumerate(y_range):
            for xi, xv in enumerate(x_range):
                if road_boundaries.is_on_road(xv, yv):
                    road_mask[yi, xi] = 1.0
        ax.contourf(x_range, y_range, road_mask, levels=[0.5, 1.5],
                    colors=['#E8F5E9'], alpha=0.3, zorder=0)

    draw_road_network(ax, roadmap, alpha=0.15, show_labels=False)

    rrt_path = rrt_result[0] if rrt_result else None
    tree_pos = rrt_result[3] if rrt_result and len(rrt_result) > 3 else []
    tree_edges = rrt_result[4] if rrt_result and len(rrt_result) > 4 else []

    # Draw tree edges (gray, thin)
    for parent_idx, child_idx in tree_edges:
        if parent_idx < len(tree_pos) and child_idx < len(tree_pos):
            p1 = tree_pos[parent_idx]
            p2 = tree_pos[child_idx]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    '-', color='#BDBDBD', linewidth=0.3, alpha=0.5, zorder=1)

    # Draw tree nodes (small gray dots)
    if tree_pos:
        tree_arr = np.array(tree_pos)
        ax.scatter(tree_arr[:, 0], tree_arr[:, 1], s=1, c='#9E9E9E',
                   alpha=0.4, zorder=2)

    # Draw solution path (bold)
    if rrt_path is not None:
        ax.plot(rrt_path[0], rrt_path[1], '-', color='#FF5722',
                linewidth=3.0, alpha=0.95, label='RRT* Solution', zorder=8)

    # Start/goal markers
    start_pos = roadmap.nodes[24].pose[:2, 0]
    goal_pos = roadmap.nodes[20].pose[:2, 0]
    ax.plot(start_pos[0], start_pos[1], 's', color='#F44336', markersize=12, zorder=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], '*', color='#4CAF50', markersize=15, zorder=10, label='Goal')

    n_tree = len(tree_pos) if tree_pos else 0
    ax.set_title(f'RRT* Tree Visualization ({n_tree} nodes, {len(tree_edges)} edges)', fontsize=14)
    ax.set_xlabel('X (meters, QLabs frame)')
    ax.set_ylabel('Y (meters, QLabs frame)')
    ax.legend(loc='best')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig8_rrt_tree.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig8_rrt_tree.png'}")
    plt.close(fig)


def figure9_chomp_convergence(convergence_history):
    """Figure 9: CHOMP convergence — total cost vs iteration."""
    if not convergence_history:
        print("  Skipping fig9 (no CHOMP convergence data)")
        return

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    iterations = range(len(convergence_history))
    ax.plot(iterations, convergence_history, '-', color='#3F51B5', linewidth=1.5)

    # Mark key points
    min_cost = min(convergence_history)
    min_idx = convergence_history.index(min_cost)
    ax.axhline(y=min_cost, color='#4CAF50', linestyle=':', alpha=0.5,
               label=f'Min cost: {min_cost:.2f} (iter {min_idx})')

    # Initial cost
    ax.axhline(y=convergence_history[0], color='#F44336', linestyle=':', alpha=0.5,
               label=f'Initial cost: {convergence_history[0]:.2f}')

    # Convergence rate: mark 90% of improvement
    initial = convergence_history[0]
    improvement = initial - min_cost
    if improvement > 0:
        threshold = initial - 0.9 * improvement
        for i, c in enumerate(convergence_history):
            if c <= threshold:
                ax.axvline(x=i, color='#FF9800', linestyle='--', alpha=0.3,
                           label=f'90% improvement at iter {i}')
                break

    ax.set_xlabel('Iteration')
    ax.set_ylabel('Total Cost (smoothness + obstacle)')
    ax.set_title('CHOMP Trajectory Optimization Convergence', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'fig9_chomp_convergence.png', dpi=150)
    print(f"  Saved {OUTPUT_DIR / 'fig9_chomp_convergence.png'}")
    plt.close(fig)


def figure10_comprehensive_dashboard(results, extended_results, forced_lengths,
                                      scenarios, astar_optimal):
    """Figure 10: Comprehensive dashboard — 2x3 grid summary."""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(2, 3, figure=fig, hspace=0.35, wspace=0.35)

    # Panel 1: Path lengths comparison
    ax1 = fig.add_subplot(gs[0, 0])
    all_algos = list(results.keys()) + list(extended_results.keys())
    scenario_name = '24 → 20'
    algo_lengths = {}
    for algo in results:
        d = results[algo].get(scenario_name, {})
        if d.get('path') is not None:
            algo_lengths[algo] = d['length']
    for algo in extended_results:
        d = extended_results[algo]
        if d.get('length', 0) > 0:
            algo_lengths[algo] = d['length']

    if algo_lengths:
        names = list(algo_lengths.keys())
        vals = [algo_lengths[n] for n in names]
        colors = [ALGO_COLORS.get(n, '#333333') for n in names]
        bars = ax1.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax1.set_yticks(range(len(names)))
        ax1.set_yticklabels(names, fontsize=7)
        ax1.set_xlabel('Path Length (m)')
        ax1.set_title(f'Path Lengths ({scenario_name})', fontsize=11)
        for i, (bar, val) in enumerate(zip(bars, vals)):
            ax1.text(val + 0.05, i, f'{val:.2f}m', va='center', fontsize=7)

    # Panel 2: Nodes expanded
    ax2 = fig.add_subplot(gs[0, 1])
    algo_expanded = {}
    for algo in results:
        d = results[algo].get(scenario_name, {})
        if d.get('nodes_expanded', 0) > 0:
            algo_expanded[algo] = d['nodes_expanded']
    for algo in extended_results:
        d = extended_results[algo]
        if d.get('nodes_expanded', 0) > 0:
            algo_expanded[algo] = d['nodes_expanded']

    if algo_expanded:
        names = list(algo_expanded.keys())
        vals = [algo_expanded[n] for n in names]
        colors = [ALGO_COLORS.get(n, '#333333') for n in names]
        ax2.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax2.set_yticks(range(len(names)))
        ax2.set_yticklabels(names, fontsize=7)
        ax2.set_xlabel('Nodes Expanded')
        ax2.set_title('Search Effort', fontsize=11)

    # Panel 3: Computation time
    ax3 = fig.add_subplot(gs[0, 2])
    algo_times = {}
    for algo in results:
        d = results[algo].get(scenario_name, {})
        if d.get('time_ms', 0) > 0:
            algo_times[algo] = d['time_ms']
    for algo in extended_results:
        d = extended_results[algo]
        if d.get('time_ms', 0) > 0:
            algo_times[algo] = d['time_ms']

    if algo_times:
        names = list(algo_times.keys())
        vals = [algo_times[n] for n in names]
        colors = [ALGO_COLORS.get(n, '#333333') for n in names]
        ax3.barh(range(len(names)), vals, color=colors, alpha=0.85)
        ax3.set_yticks(range(len(names)))
        ax3.set_yticklabels(names, fontsize=7)
        ax3.set_xlabel('Time (ms)')
        ax3.set_title('Computation Time', fontsize=11)
        ax3.set_xscale('log')

    # Panel 4: Optimality ratios
    ax4 = fig.add_subplot(gs[1, 0])
    if astar_optimal > 0 and algo_lengths:
        names = list(algo_lengths.keys())
        ratios = [algo_lengths[n] / astar_optimal for n in names]
        colors = [ALGO_COLORS.get(n, '#333333') for n in names]
        bars = ax4.barh(range(len(names)), ratios, color=colors, alpha=0.85)
        ax4.axvline(x=1.0, color='#F44336', linestyle='--', alpha=0.5, label='Optimal')
        ax4.set_yticks(range(len(names)))
        ax4.set_yticklabels(names, fontsize=7)
        ax4.set_xlabel('Path Length / Optimal')
        ax4.set_title('Optimality Ratio', fontsize=11)
        ax4.legend(fontsize=8)

    # Panel 5: Algorithm properties table
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.axis('off')
    table_data = [
        ['Algorithm', 'Category', 'Optimal?', 'Bound', 'Complete?'],
        ['A*', 'Graph', 'Yes', '1.0x', 'Yes'],
        ['Dijkstra', 'Graph', 'Yes', '1.0x', 'Yes'],
        ['D* Weighted', 'Graph', 'No', 'N/A', 'Yes'],
        ['Bidir A*', 'Graph', 'Yes', '1.0x', 'Yes'],
        ['Weighted A*', 'Graph', 'No', 'eps*x', 'Yes'],
        ['MHA*', 'Graph', 'No', 'w1*w2*x', 'Yes'],
        ['Experience', 'Cached', 'Yes*', '1.0x', 'Yes'],
        ['RRT*', 'Continuous', 'Asymp.', 'N/A', 'Prob.'],
        ['CHOMP', 'Continuous', 'Local', 'N/A', 'No'],
    ]
    table = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                      loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    ax5.set_title('Algorithm Properties', fontsize=11, pad=20)

    # Panel 6: Recommendation text
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    rec_text = (
        "RECOMMENDATION\n"
        "─────────────────\n\n"
        "For the SDCS taxi mission:\n\n"
        "  PRIMARY: A* on road graph\n"
        "  (optimal, fast, complete)\n\n"
        "  RUNTIME: Experience-Based A*\n"
        "  (cache routes for O(n) replay)\n\n"
        "  FALLBACK: Weighted A* (eps=1.5)\n"
        "  (faster search, bounded quality)\n\n"
        "  NOT RECOMMENDED:\n"
        "  - RRT*/CHOMP on this graph\n"
        "    (narrow lanes need precise\n"
        "     road-following geometry)"
    )
    ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes, fontsize=9,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#E3F2FD', alpha=0.8))
    ax6.set_title('Recommendation', fontsize=11, pad=20)

    fig.suptitle('Pathfinding Algorithm Comparison — Comprehensive Dashboard', fontsize=16, y=0.98)
    fig.savefig(OUTPUT_DIR / 'fig10_comprehensive_dashboard.png', dpi=150, bbox_inches='tight')
    print(f"  Saved {OUTPUT_DIR / 'fig10_comprehensive_dashboard.png'}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("Pathfinding Algorithm Comparison — SDCS Road Network")
    print("=" * 72)

    # Build road graph
    print("\nBuilding road graph...")
    roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
    print(f"  Nodes: {len(roadmap.nodes)}")
    print(f"  Edges: {len(roadmap.edges)}")

    # Algorithm registry (original 4)
    algorithms = {
        'A*': find_astar_path,
        'Dijkstra': find_dijkstra_path,
        'D* Weighted': find_dstar_weighted_path,
        'Bidirectional A*': find_bidirectional_astar_path,
    }

    # Test scenarios (matching actual mission legs)
    scenarios = {
        '24 → 20': (24, 20),   # hub_to_pickup
        '21 → 8':  (21, 8),    # pickup_to_dropoff (node 21 near pickup)
        '8 → 24':  (8, 24),    # dropoff_to_hub
    }

    # Current forced sequences for comparison
    forced_sequences = {
        '24 → 20': [24, 1, 13, 19, 17, 20],
        '21 → 8':  [21, 16, 18, 11, 12, 8],
        '8 → 24':  [8, 10, 24],
    }

    # Compute forced route lengths
    forced_results = {}
    forced_lengths = {}
    print("\n--- Current forced routes ---")
    for scenario, seq in forced_sequences.items():
        path, length = compute_forced_route_length(roadmap, seq)
        forced_results[scenario] = path
        forced_lengths[scenario] = length
        if path is not None:
            print(f"  {scenario} (seq {seq}): {length:.3f} m, {path.shape[1]} waypoints")
        else:
            print(f"  {scenario} (seq {seq}): FAILED")

    # Run original algorithms on all scenarios
    results = defaultdict(dict)
    print("\n--- Algorithm Results (Original 4) ---")

    for algo_name, algo_func in algorithms.items():
        print(f"\n  [{algo_name}]")
        for scenario_name, (start, goal) in scenarios.items():
            t0 = time.perf_counter()
            path, nodes_expanded, node_seq = algo_func(roadmap, start, goal)
            elapsed = time.perf_counter() - t0

            if path is not None:
                diffs = np.diff(path, axis=1)
                length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
                n_waypoints = path.shape[1]
            else:
                length = 0
                n_waypoints = 0

            results[algo_name][scenario_name] = {
                'path': path,
                'length': length,
                'nodes_expanded': nodes_expanded,
                'n_waypoints': n_waypoints,
                'time_ms': elapsed * 1000,
                'node_seq': node_seq,
            }

            seq_str = ' → '.join(str(n) for n in node_seq) if node_seq else 'N/A'
            status = f"{length:.3f} m" if path is not None else "NO PATH"
            print(f"    {scenario_name}: {status}, "
                  f"{nodes_expanded} expanded, {n_waypoints} wpts, "
                  f"{elapsed*1000:.2f} ms")
            print(f"      Nodes: [{seq_str}]")

    # -----------------------------------------------------------------------
    # NEW ALGORITHMS (graph-based)
    # -----------------------------------------------------------------------
    print("\n--- New Graph-Based Algorithms ---")

    # Weighted A* at various epsilon values
    for eps in [1.5, 2.0, 3.0]:
        algo_name = f'Weighted A* ({eps})'
        print(f"\n  [{algo_name}]")
        for scenario_name, (start, goal) in scenarios.items():
            t0 = time.perf_counter()
            path, nodes_expanded, node_seq = find_weighted_astar_path(
                roadmap, start, goal, epsilon=eps)
            elapsed = time.perf_counter() - t0

            if path is not None:
                diffs = np.diff(path, axis=1)
                length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
                n_waypoints = path.shape[1]
            else:
                length = 0
                n_waypoints = 0

            results[algo_name][scenario_name] = {
                'path': path,
                'length': length,
                'nodes_expanded': nodes_expanded,
                'n_waypoints': n_waypoints,
                'time_ms': elapsed * 1000,
                'node_seq': node_seq,
            }
            status = f"{length:.3f} m" if path is not None else "NO PATH"
            print(f"    {scenario_name}: {status}, "
                  f"{nodes_expanded} expanded, {elapsed*1000:.2f} ms")

    # MHA*
    algo_name = 'MHA*'
    print(f"\n  [{algo_name}]")
    for scenario_name, (start, goal) in scenarios.items():
        t0 = time.perf_counter()
        path, nodes_expanded, node_seq = find_mha_star_path(roadmap, start, goal)
        elapsed = time.perf_counter() - t0

        if path is not None:
            diffs = np.diff(path, axis=1)
            length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
            n_waypoints = path.shape[1]
        else:
            length = 0
            n_waypoints = 0

        results[algo_name][scenario_name] = {
            'path': path,
            'length': length,
            'nodes_expanded': nodes_expanded,
            'n_waypoints': n_waypoints,
            'time_ms': elapsed * 1000,
            'node_seq': node_seq,
        }
        status = f"{length:.3f} m" if path is not None else "NO PATH"
        print(f"    {scenario_name}: {status}, "
              f"{nodes_expanded} expanded, {elapsed*1000:.2f} ms")

    # Experience-Based A*
    algo_name = 'Experience A*'
    print(f"\n  [{algo_name}]")
    exp_planner = ExperienceBasedPlanner()
    for scenario_name, (start, goal) in scenarios.items():
        # First call (cache miss)
        t0 = time.perf_counter()
        path, nodes_expanded, node_seq = exp_planner.find_path(roadmap, start, goal)
        elapsed1 = time.perf_counter() - t0

        # Second call (cache hit)
        t0 = time.perf_counter()
        path2, nodes_expanded2, node_seq2 = exp_planner.find_path(roadmap, start, goal)
        elapsed2 = time.perf_counter() - t0

        if path is not None:
            diffs = np.diff(path, axis=1)
            length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
            n_waypoints = path.shape[1]
        else:
            length = 0
            n_waypoints = 0

        results[algo_name][scenario_name] = {
            'path': path,
            'length': length,
            'nodes_expanded': nodes_expanded,
            'n_waypoints': n_waypoints,
            'time_ms': elapsed1 * 1000,
            'node_seq': node_seq,
        }
        status = f"{length:.3f} m" if path is not None else "NO PATH"
        print(f"    {scenario_name}: {status}, "
              f"first={elapsed1*1000:.2f}ms, cached={elapsed2*1000:.4f}ms, "
              f"{nodes_expanded} expanded")

    print(f"  Cache stats: hits={exp_planner.cache_hits}, misses={exp_planner.cache_misses}")

    # -----------------------------------------------------------------------
    # NEW ALGORITHMS (continuous-space)
    # -----------------------------------------------------------------------
    print("\n--- Continuous-Space Algorithms ---")
    print("  Loading road boundaries...")
    road_boundaries = RoadBoundaryLoader()
    print(f"  Loaded {len(road_boundaries.segments)} segments, "
          f"{len(road_boundaries.circles)} circles")

    # RRT* (Hub → Pickup only — most interesting scenario)
    extended_results = {}

    print("\n  [RRT*] (Hub → Pickup, 24 → 20)")
    random.seed(42)  # Reproducibility
    t0 = time.perf_counter()
    rrt_result = find_rrt_star_path(roadmap, 24, 20,
                                     road_boundaries=road_boundaries,
                                     max_iter=2000, step_size=0.1,
                                     goal_radius=0.15)
    rrt_elapsed = time.perf_counter() - t0

    rrt_path = rrt_result[0] if rrt_result else None
    if rrt_path is not None:
        diffs = np.diff(rrt_path, axis=1)
        rrt_length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
        n_tree = len(rrt_result[3]) if len(rrt_result) > 3 else 0
        print(f"    Path: {rrt_length:.3f} m, {rrt_path.shape[1]} wpts, "
              f"{n_tree} tree nodes, {rrt_elapsed*1000:.1f} ms")
    else:
        rrt_length = 0
        n_tree = len(rrt_result[3]) if rrt_result and len(rrt_result) > 3 else 0
        print(f"    NO PATH FOUND ({n_tree} tree nodes, {rrt_elapsed*1000:.1f} ms)")

    extended_results['RRT*'] = {
        'path': rrt_path,
        'length': rrt_length,
        'nodes_expanded': n_tree,
        'time_ms': rrt_elapsed * 1000,
    }

    # CHOMP (Hub → Pickup only)
    print("\n  [CHOMP] (Hub → Pickup, 24 → 20)")
    t0 = time.perf_counter()
    chomp_result = find_chomp_path(roadmap, 24, 20,
                                    road_boundaries=road_boundaries,
                                    n_waypoints=50, n_iterations=200)
    chomp_elapsed = time.perf_counter() - t0

    chomp_path = chomp_result[0] if chomp_result else None
    chomp_history = chomp_result[3] if chomp_result and len(chomp_result) > 3 else []
    if chomp_path is not None:
        diffs = np.diff(chomp_path, axis=1)
        chomp_length = float(np.sum(np.sqrt(diffs[0]**2 + diffs[1]**2)))
        print(f"    Path: {chomp_length:.3f} m, {chomp_path.shape[1]} wpts, "
              f"{chomp_elapsed*1000:.1f} ms")
        if chomp_history:
            print(f"    Cost: {chomp_history[0]:.2f} -> {chomp_history[-1]:.2f} "
                  f"({(1 - chomp_history[-1]/chomp_history[0])*100:.1f}% reduction)")
    else:
        chomp_length = 0
        print(f"    NO PATH ({chomp_elapsed*1000:.1f} ms)")

    extended_results['CHOMP'] = {
        'path': chomp_path,
        'length': chomp_length,
        'nodes_expanded': 200,
        'time_ms': chomp_elapsed * 1000,
    }

    # -----------------------------------------------------------------------
    # SUMMARY TABLE
    # -----------------------------------------------------------------------
    all_algorithms = list(algorithms.keys()) + [
        'Weighted A* (1.5)', 'Weighted A* (2.0)', 'Weighted A* (3.0)',
        'MHA*', 'Experience A*',
    ]

    print("\n" + "=" * 100)
    print("SUMMARY TABLE (All Algorithms)")
    print("=" * 100)
    header = (f"{'Algorithm':<22} {'Category':<10} {'Scenario':<12} "
              f"{'Length (m)':>10} {'Expanded':>9} {'Waypts':>7} "
              f"{'Time (ms)':>10} {'Opt. Bound':<12} {'Node Sequence'}")
    print(header)
    print("-" * len(header))

    # Optimality bounds
    opt_bounds = {
        'A*': '1.0x (opt)',
        'Dijkstra': '1.0x (opt)',
        'D* Weighted': 'N/A',
        'Bidirectional A*': '1.0x (opt)',
        'Weighted A* (1.5)': '1.5x',
        'Weighted A* (2.0)': '2.0x',
        'Weighted A* (3.0)': '3.0x',
        'MHA*': '2.25x (w1*w2)',
        'Experience A*': '1.0x (cached)',
    }

    categories = {
        'A*': 'graph', 'Dijkstra': 'graph', 'D* Weighted': 'graph',
        'Bidirectional A*': 'graph',
        'Weighted A* (1.5)': 'graph', 'Weighted A* (2.0)': 'graph',
        'Weighted A* (3.0)': 'graph',
        'MHA*': 'graph', 'Experience A*': 'cached',
    }

    for algo_name in all_algorithms:
        for scenario_name in scenarios:
            d = results[algo_name].get(scenario_name, {})
            if not d:
                continue
            seq_str = '→'.join(str(n) for n in d.get('node_seq', [])) if d.get('node_seq') else 'N/A'
            length_str = f"{d['length']:.3f}" if d.get('path') is not None else "FAIL"
            cat = categories.get(algo_name, '?')
            bound = opt_bounds.get(algo_name, 'N/A')
            print(f"{algo_name:<22} {cat:<10} {scenario_name:<12} {length_str:>10} "
                  f"{d.get('nodes_expanded', 0):>9} {d.get('n_waypoints', 0):>7} "
                  f"{d.get('time_ms', 0):>10.2f} {bound:<12} {seq_str}")

    # Continuous-space results
    print("\n" + "-" * 100)
    print("CONTINUOUS-SPACE ALGORITHMS (Hub → Pickup only)")
    print("-" * 100)
    for algo_name, d in extended_results.items():
        length_str = f"{d['length']:.3f}" if d.get('path') is not None else "FAIL"
        print(f"  {algo_name:<20} {length_str:>10} m, "
              f"{d.get('nodes_expanded', 0):>6} nodes/iters, "
              f"{d.get('time_ms', 0):>10.1f} ms")

    # Forced route comparison
    print("\n" + "-" * 100)
    print("FORCED ROUTE COMPARISON (Current vs Best Algorithm)")
    print("-" * 100)
    for scenario_name in scenarios:
        forced_len = forced_lengths.get(scenario_name, 0)
        best_algo = None
        best_len = float('inf')
        for algo_name in all_algorithms:
            d = results[algo_name].get(scenario_name, {})
            if d.get('path') is not None and d.get('length', float('inf')) < best_len:
                best_len = d['length']
                best_algo = algo_name

        if best_algo and forced_len > 0:
            savings = forced_len - best_len
            pct = (savings / forced_len) * 100 if forced_len > 0 else 0
            print(f"  {scenario_name}: Forced={forced_len:.3f}m, "
                  f"Best ({best_algo})={best_len:.3f}m, "
                  f"Savings={savings:.3f}m ({pct:.1f}%)")
        elif best_algo:
            print(f"  {scenario_name}: No forced route, "
                  f"Best ({best_algo})={best_len:.3f}m")

    # Recommended route sequences
    print("\n" + "-" * 100)
    print("RECOMMENDED _ROUTE_NODE_SEQUENCES")
    print("-" * 100)
    route_mapping = {
        'hub_to_pickup': '24 → 20',
        'pickup_to_dropoff': '21 → 8',
        'dropoff_to_hub': '8 → 24',
    }
    for route_name, scenario in route_mapping.items():
        best_algo = None
        best_len = float('inf')
        best_seq = []
        for algo_name in all_algorithms:
            d = results[algo_name].get(scenario, {})
            if d.get('path') is not None and d.get('length', float('inf')) < best_len:
                best_len = d['length']
                best_algo = algo_name
                best_seq = d.get('node_seq', [])
        print(f"  '{route_name}': {best_seq}  ({best_algo}, {best_len:.3f}m)")

    # -----------------------------------------------------------------------
    # GENERATE FIGURES
    # -----------------------------------------------------------------------
    print("\n--- Generating figures (1-4: original) ---")
    figure1_road_network(roadmap)
    figure2_hub_to_pickup(roadmap, results, forced_results.get('24 → 20'))
    figure3_bar_chart(results, forced_lengths)
    figure4_all_legs(roadmap, results)

    print("\n--- Generating figures (5-10: new) ---")

    # Figure 5: Weighted A* epsilon tradeoff
    figure5_weighted_astar_tradeoff(roadmap)

    # Figure 6: Experience-based speedup
    figure6_experience_speedup(roadmap, scenarios)

    # Get A* path for comparison
    astar_path_24_20 = results['A*'].get('24 → 20', {}).get('path')
    astar_length = results['A*'].get('24 → 20', {}).get('length', 0)

    # Figure 7: Continuous vs graph paths
    figure7_continuous_vs_graph(roadmap, astar_path_24_20, rrt_result,
                                chomp_result, road_boundaries)

    # Figure 8: RRT* tree
    figure8_rrt_tree(roadmap, rrt_result, road_boundaries)

    # Figure 9: CHOMP convergence
    figure9_chomp_convergence(chomp_history)

    # Figure 10: Comprehensive dashboard
    figure10_comprehensive_dashboard(results, extended_results, forced_lengths,
                                      scenarios, astar_length)

    print(f"\nAll figures saved to: {OUTPUT_DIR}/")
    print("Done!")


if __name__ == '__main__':
    main()
