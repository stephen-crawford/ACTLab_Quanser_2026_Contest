"""
Road-network-based path planner for SDCS competition track.

Ported from the reference repo's SDCSRoadMap (mats.py), SCSPath and
RoadMap (path_planning.py) classes. Uses Straight-Curve-Straight (SCS)
path generation with A* graph search for proper road-following geometry.

All coordinates are in QLabs world frame (meters).

Note: The primary path planning runs in C++ (road_graph.cpp). This Python
version is used by visualization tools (path_overlay, generate_report).
"""

import math
import heapq
import numpy as np
from typing import List, Optional
from scipy.interpolate import splprep, splev

# ---------------------------------------------------------------------------
# Math utilities (ported from pal.utilities.math)
# ---------------------------------------------------------------------------
TWO_PI = 2.0 * np.pi


def _wrap_to_2pi(th: float) -> float:
    return np.mod(np.mod(th, TWO_PI) + TWO_PI, TWO_PI)


def _wrap_to_pi(th: float) -> float:
    th = th % TWO_PI
    th = (th + TWO_PI) % TWO_PI
    if th > np.pi:
        th -= TWO_PI
    return th


def _signed_angle(v1, v2=None):
    if v2 is None:
        return np.arctan2(v1[1], v1[0])
    return _wrap_to_pi(np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0]))


# ---------------------------------------------------------------------------
# SCSPath - Straight-Curve-Straight path generation
# (ported from hal.utilities.path_planning)
# ---------------------------------------------------------------------------
def SCSPath(startPose, endPose, radius, stepSize=0.01):
    """Calculate the path between two poses using at most one turn.

    Args:
        startPose: 3x1 numpy array [x; y; th]
        endPose: 3x1 numpy array [x; y; th]
        radius: turn radius (0 for straight line)
        stepSize: distance between points (meters)

    Returns:
        (path, path_length) where path is 2xN array, or (None, None) on failure
    """
    if radius < 1e-6:
        # Straight line between the two poses
        p1 = startPose[:2, :].flatten()
        p2 = endPose[:2, :].flatten()
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            return np.empty((2, 0)), 0.0
        n_pts = max(int(dist / stepSize), 2)
        t_vals = np.linspace(0, 1, n_pts, endpoint=False)[1:]
        path = np.empty((2, len(t_vals)))
        for i, t in enumerate(t_vals):
            path[:, i] = p1 + t * (p2 - p1)
        return path, dist

    p1 = startPose[:2, :]
    th1 = startPose[2, 0]
    p2 = endPose[:2, :]
    th2 = endPose[2, 0]

    t1 = np.array([[np.cos(th1)], [np.sin(th1)]])
    t2 = np.array([[np.cos(th2)], [np.sin(th2)]])

    sa = _signed_angle(t1, p2 - p1)
    # When heading points directly at dest (sa≈0), the direction is ambiguous.
    # Resolve by using the actual heading change (end - start) instead.
    if abs(sa) < 0.05:
        sa = _wrap_to_pi(th2 - th1)
    direction = 1 if sa > 0 else -1

    n1 = radius * np.array([[-t1[1, 0]], [t1[0, 0]]]) * direction
    n2 = radius * np.array([[-t2[1, 0]], [t2[0, 0]]]) * direction

    tol = 0.01

    if np.abs(_wrap_to_pi(th2 - th1)) < tol:
        v = p2 - p1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return np.empty((2, 0)), 0.0
        v_uv = v / v_norm
        if 1 - np.abs(np.dot(t1.squeeze(), v_uv.squeeze())) < tol:
            c = p2 + n1
        else:
            return None, None
    elif np.abs(_wrap_to_pi(th2 - th1 + np.pi)) < tol:
        v = (p2 + 2 * n2) - p1
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-9:
            return np.empty((2, 0)), 0.0
        v_uv = v / v_norm
        if 1 - np.abs(np.dot(t1.squeeze(), v_uv.squeeze())) < tol:
            s = np.dot(t1.squeeze(), v.squeeze())
            if s < tol:
                c = p1 + n1
            else:
                c = p2 + n2
        else:
            return None, None
    else:
        d1 = p1 + n1
        d2 = p2 + n2
        A = np.hstack((t1, -t2))
        b = d2 - d1
        try:
            alpha, beta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            return None, None
        if alpha >= -tol and beta <= tol:
            c = d1 + alpha * t1
        else:
            return None, None

    b1 = c - n1
    b2 = c - n2

    # Discretize line-segment p1 -> b1
    line1 = np.empty((2, 0))
    line1_length = np.linalg.norm(b1 - p1)
    if line1_length > stepSize:
        ds = (1.0 / line1_length) * stepSize
        s = ds
        while s < 1:
            p = p1 + s * (b1 - p1)
            line1 = np.hstack((line1, p))
            s += ds

    # Discretize arc b1 -> b2
    arc = np.empty((2, 0))
    ang_dist = _wrap_to_2pi(direction * _signed_angle(b1 - c, b2 - c))
    arc_length = np.abs(ang_dist * radius)
    if arc_length > stepSize:
        start_angle = np.arctan2(b1[1] - c[1], b1[0] - c[0])
        dth = (2 * np.pi / (np.pi * 2 * radius)) * stepSize
        s = dth
        while s < ang_dist:
            th = start_angle + s * direction
            p = c + np.array([np.cos(th), np.sin(th)]) * radius
            arc = np.hstack((arc, p))
            s += dth

    # Discretize line-segment b2 -> p2
    line2 = np.empty((2, 0))
    line2_length = np.linalg.norm(b2 - p2)
    if line2_length > stepSize:
        ds = (1.0 / line2_length) * stepSize
        s = ds
        while s < 1:
            p = b2 + s * (p2 - b2)
            line2 = np.hstack((line2, p))
            s += ds

    path = np.hstack((line1, arc, line2))
    path_length = line1_length + arc_length + line2_length
    return path, path_length


# ---------------------------------------------------------------------------
# RoadMap graph classes (ported from hal.utilities.path_planning)
# ---------------------------------------------------------------------------
class RoadMapNode:
    def __init__(self, pose):
        assert len(pose) == 3, "Pose must be [x, y, th]"
        self.pose = np.array(pose).reshape(3, 1)
        self.inEdges = []
        self.outEdges = []

    def __lt__(self, other):
        # Needed for heapq tie-breaking
        return id(self) < id(other)


class RoadMapEdge:
    def __init__(self, fromNode, toNode):
        self.fromNode = fromNode
        self.toNode = toNode
        self.waypoints = None
        self.length = None


class RoadMap:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, pose):
        self.nodes.append(RoadMapNode(pose))

    def add_edge(self, fromNode, toNode, radius):
        if isinstance(fromNode, int):
            fromNode = self.nodes[fromNode]
        if isinstance(toNode, int):
            toNode = self.nodes[toNode]

        edge = RoadMapEdge(fromNode, toNode)
        self.edges.append(edge)
        fromNode.outEdges.append(edge)
        toNode.inEdges.append(edge)
        self._calculate_trajectory(edge, radius)

    def _calculate_trajectory(self, edge, radius):
        points, length = SCSPath(
            startPose=edge.fromNode.pose,
            endPose=edge.toNode.pose,
            radius=radius,
            stepSize=0.01
        )
        edge.waypoints = points
        edge.length = length

    def find_shortest_path(self, startNode, goalNode):
        """A* shortest path. Returns 2xN path array or None."""
        if isinstance(startNode, int):
            startNode = self.nodes[startNode]
        if isinstance(goalNode, int):
            goalNode = self.nodes[goalNode]

        if startNode == goalNode:
            return None

        openSet = []
        closedSet = set()
        h = np.linalg.norm(goalNode.pose[:2, :] - startNode.pose[:2, :])
        heapq.heappush(openSet, (h, startNode))

        gScore = {node: float('inf') for node in self.nodes}
        gScore[startNode] = 0
        cameFrom = {node: None for node in self.nodes}

        while openSet:
            currentNode = heapq.heappop(openSet)[1]

            if currentNode == goalNode:
                path = goalNode.pose[:2, :]
                node = goalNode
                while True:
                    (node, edge) = cameFrom[node]
                    if edge.waypoints is not None and edge.waypoints.shape[1] > 0:
                        path = np.hstack((node.pose[:2, :], edge.waypoints, path))
                    else:
                        path = np.hstack((node.pose[:2, :], path))
                    if cameFrom[node] is None:
                        break
                return path

            closedSet.add(currentNode)

            for edge in currentNode.outEdges:
                neighborNode = edge.toNode
                if neighborNode in closedSet:
                    continue

                if edge.length is None:
                    tentative_g = float('inf')
                else:
                    tentative_g = gScore[currentNode] + edge.length

                if tentative_g < gScore[neighborNode]:
                    cameFrom[neighborNode] = (currentNode, edge)
                    gScore[neighborNode] = tentative_g
                    hScore = np.linalg.norm(
                        goalNode.pose[:2, :] - neighborNode.pose[:2, :])
                    heapq.heappush(
                        openSet,
                        (gScore[neighborNode] + hScore, neighborNode))

        return None

    def generate_path(self, nodeSequence, spacing=0.001, scale_factor=None):
        """Generate shortest path through a sequence of node indices.

        Matches reference 2025: B-spline resample + optional scale factor.

        Args:
            nodeSequence: list of node indices
            spacing: resample spacing in meters (default 0.001 = 1mm, matches ref)
            scale_factor: [sx, sy] scale factors (default [1.01, 1.0], matches ref)

        Returns:
            2xN numpy array of waypoints, or None if no path exists.
        """
        if scale_factor is None:
            scale_factor = [1.01, 1.0]

        path = np.empty((2, 0))
        for i in range(len(nodeSequence) - 1):
            segment = self.find_shortest_path(
                nodeSequence[i], nodeSequence[i + 1])
            if segment is None:
                return None
            path = np.hstack((path, segment[:, :-1]))
        # Add final node
        final_node = self.nodes[nodeSequence[-1]]
        path = np.hstack((path, final_node.pose[:2, :]))

        # B-spline resample to uniform spacing (matches reference splprep/splev)
        if path.shape[1] >= 4:
            path = self._resample_waypoints(path, spacing=spacing)

        # Apply scale factor (reference uses [1.01, 1.0] for lane centering)
        path[0, :] *= scale_factor[0]
        path[1, :] *= scale_factor[1]

        return path

    @staticmethod
    def _resample_waypoints(path, spacing=0.001):
        """Resample path to uniform spacing using B-spline (matches reference)."""
        total_len = np.sum(np.sqrt(np.sum(np.diff(path, axis=1)**2, axis=0)))
        n_out = max(int(total_len / spacing), 10)
        tck, u = splprep([path[0], path[1]], s=0)
        u_new = np.linspace(0, 1, n_out)
        out = splev(u_new, tck)
        return np.vstack(out)


# ---------------------------------------------------------------------------
# SDCSRoadMap - the actual competition track road network
# (ported from hal.products.mats)
# ---------------------------------------------------------------------------
class SDCSRoadMap(RoadMap):
    """Road network for Quanser's Self-Driving Car Studio (SDCS)."""

    def __init__(self, leftHandTraffic=False, useSmallMap=False):
        super().__init__()

        scale = 0.002035
        xOffset = 1134
        yOffset = 2363

        innerLaneRadius = 305.5 * scale    # 0.622
        outerLaneRadius = 438 * scale       # 0.891
        trafficCircleRadius = 333 * scale   # 0.678
        oneWayStreetRadius = 350 * scale    # 0.712
        kinkStreetRadius = 375 * scale      # 0.763

        pi = np.pi
        halfPi = pi / 2

        def scale_then_add_nodes(nodePoses):
            for pose in nodePoses:
                pose[0] = scale * (pose[0] - xOffset)
                pose[1] = scale * (yOffset - pose[1])
                self.add_node(pose)

        # Right-hand traffic (default for competition)
        nodePoses = [
            [1134, 2299, -halfPi],          # 0
            [1266, 2323, halfPi],            # 1
            [1688, 2896, 0],                 # 2
            [1688, 2763, pi],                # 3
            [2242, 2323, halfPi],            # 4
            [2109, 2323, -halfPi],           # 5
            [1632, 1822, pi],                # 6
            [1741, 1955, 0],                 # 7
            [766, 1822, pi],                 # 8
            [766, 1955, 0],                  # 9
            [504, 2589, -42 * pi / 180],     # 10
        ]
        if not useSmallMap:
            nodePoses += [
                [1134, 1300, -halfPi],       # 11
                [1134, 1454, -halfPi],        # 12
                [1266, 1454, halfPi],         # 13
                [2242, 905, halfPi],          # 14
                [2109, 1454, -halfPi],        # 15
                [1580, 540, -80.6 * pi / 180],    # 16
                [1854.4, 814.5, -9.4 * pi / 180], # 17
                [1440, 856, -138 * pi / 180],      # 18
                [1523, 958, 42 * pi / 180],        # 19
                [1134, 153, pi],              # 20
                [1134, 286, 0],               # 21
                [159, 905, -halfPi],          # 22
                [291, 905, halfPi],           # 23
            ]

        edgeConfigs = [
            [0, 2, outerLaneRadius],
            [1, 7, innerLaneRadius],
            [1, 8, outerLaneRadius],
            [2, 4, outerLaneRadius],
            [3, 1, innerLaneRadius],
            [4, 6, outerLaneRadius],
            [5, 3, innerLaneRadius],
            [6, 0, outerLaneRadius],
            [6, 8, 0],
            [7, 5, innerLaneRadius],
            [8, 10, oneWayStreetRadius],
            [9, 0, innerLaneRadius],
            [9, 7, 0],
            [10, 1, innerLaneRadius],
            [10, 2, innerLaneRadius],
        ]
        if not useSmallMap:
            edgeConfigs += [
                [1, 13, 0],
                [4, 14, 0],
                [6, 13, innerLaneRadius],
                [7, 14, outerLaneRadius],
                [8, 23, innerLaneRadius],
                [9, 13, outerLaneRadius],
                [11, 12, 0],
                [12, 0, 0],
                [12, 7, outerLaneRadius],
                [12, 8, innerLaneRadius],
                [13, 19, innerLaneRadius],
                [14, 16, trafficCircleRadius],
                [14, 20, trafficCircleRadius],
                [15, 5, outerLaneRadius],
                [15, 6, innerLaneRadius],
                [16, 17, trafficCircleRadius],
                [16, 18, innerLaneRadius],
                [17, 15, innerLaneRadius],
                [17, 16, trafficCircleRadius],
                [17, 20, trafficCircleRadius],
                [18, 11, kinkStreetRadius],
                [19, 17, innerLaneRadius],
                [20, 22, outerLaneRadius],
                [21, 16, innerLaneRadius],
                [22, 9, outerLaneRadius],
                [22, 10, outerLaneRadius],
                [23, 21, innerLaneRadius],
            ]

        scale_then_add_nodes(nodePoses)
        for edgeConfig in edgeConfigs:
            self.add_edge(*edgeConfig)

        # Reference 2025 uses D* with +20 penalty on traffic-controlled edges.
        # This steers routing away from intersections with traffic lights/crosswalks.
        # Apply the same penalties to our A* — verified to produce identical routes.
        # Penalties added AFTER edge creation so SCS waypoints are unaffected.
        penalized_edges = {1, 2, 7, 8, 11, 12, 15, 17, 20, 22, 23, 24}
        for idx in penalized_edges:
            if idx < len(self.edges):
                self.edges[idx].length += 20.0

        # Spawn node (node 24): the vehicle's starting position at the hub.
        # Reference 2025 uses: -44.7 % (2*pi) where -44.7 is in radians.
        # This gives 5.5655 rad. Edge radii from reference are tuned for this heading.
        hub_heading = (-44.7) % (2 * np.pi)  # = 5.5655 rad, matches reference
        self.add_node([-1.205, -0.83, hub_heading])  # node 24
        self.add_edge(24, 2, radius=0.0)        # straight to node 2
        # Note: reference uses radius=1.48202 for 10→24 but the SCS geometry is
        # infeasible at this heading (beta > tol). Use straight line since nodes
        # are only 0.38m apart. The reference also silently fails on this edge.
        self.add_edge(10, 24, radius=0.0)       # straight from node 10 (0.38m)
        self.add_edge(24, 1, radius=0.866326)   # curved to node 1 (reference radius)


# ---------------------------------------------------------------------------
# Mission locations (QLabs world coordinates)
# ---------------------------------------------------------------------------
HUB = (-1.205, -0.83)
PICKUP = (0.125, 4.395)
DROPOFF = (-0.905, 0.800)


# ---------------------------------------------------------------------------
# RoadGraph - high-level interface used by mission_manager
# ---------------------------------------------------------------------------
class RoadGraph:
    """
    Pre-defined road network for the SDCS track.

    Uses the reference repo's SDCSRoadMap with SCSPath geometry to generate
    proper lane-following waypoints. Pre-computes three mission leg routes
    from a single loop path, matching C++ road_graph.cpp.
    """

    # Reference 2025 approach: single loop path through [24, 20, 9, 10],
    # then slice at pickup and dropoff indices to produce three mission legs.
    # The loop visits: Hub -> (node 2,4,14,20=Pickup area) -> (node 22,9=Dropoff area)
    # -> (node 13,19,17,20,22,10) -> Hub.
    _LOOP_NODE_SEQUENCE = [24, 20, 9, 10]

    def __init__(self, ds: float = 0.001):
        """
        Args:
            ds: Waypoint spacing in meters (default 0.001 = 1mm, matches C++ and reference).
        """
        self.ds = ds
        self._roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
        self._routes = {}

        # Generate single loop path (matching C++ road_graph.cpp)
        loop_2xN = self._roadmap.generate_path(
            self._LOOP_NODE_SEQUENCE, spacing=self.ds, scale_factor=[1.01, 1.0])
        if loop_2xN is None:
            return
        loop_path = loop_2xN.T  # Nx2

        # Find waypoint indices for slicing. The loop passes near Pickup TWICE
        # (outbound via node 20, then returning via inner track). Use first-pass
        # search for correct ordering: Hub -> Pickup -> Dropoff -> Hub.
        pickup_pt = np.array(PICKUP)
        dropoff_pt = np.array(DROPOFF)

        dists_pickup = np.linalg.norm(loop_path - pickup_pt, axis=1)
        dists_dropoff = np.linalg.norm(loop_path - dropoff_pt, axis=1)

        pickup_idx = self._find_first_local_min(dists_pickup, threshold=0.5)
        dropoff_idx = pickup_idx + self._find_first_local_min(
            dists_dropoff[pickup_idx:], threshold=0.5)

        # Slice into three legs (same as C++ road_graph.cpp)
        self._routes['hub_to_pickup'] = loop_path[:pickup_idx + 1]
        if pickup_idx < dropoff_idx:
            self._routes['pickup_to_dropoff'] = loop_path[pickup_idx:dropoff_idx + 1]
        self._routes['dropoff_to_hub'] = loop_path[dropoff_idx:]

        self._loop_path = loop_path
        self._pickup_idx = pickup_idx
        self._dropoff_idx = dropoff_idx

    @staticmethod
    def _find_first_local_min(dists: np.ndarray, threshold: float = 0.5) -> int:
        """Find index of the first local minimum below threshold.

        Scans forward until entering a region within threshold, then returns
        the argmin within that contiguous region. Falls back to global argmin.
        """
        in_region = False
        region_start = 0
        best_idx = int(np.argmin(dists))  # fallback

        for i in range(len(dists)):
            if dists[i] < threshold:
                if not in_region:
                    in_region = True
                    region_start = i
            else:
                if in_region:
                    best_idx = region_start + int(
                        np.argmin(dists[region_start:i]))
                    return best_idx
        if in_region:
            best_idx = region_start + int(
                np.argmin(dists[region_start:]))
        return best_idx

    def get_route(self, route_name: str) -> Optional[np.ndarray]:
        """Get pre-computed route waypoints. Returns Mx2 array or None."""
        return self._routes.get(route_name)

    def get_route_names(self) -> List[str]:
        return list(self._routes.keys())


def qlabs_path_to_map_path(
    qlabs_waypoints: np.ndarray,
    origin_x: float = -1.205,
    origin_y: float = -0.83,
    origin_heading_deg: float = -44.7,
    origin_heading_rad: float = 0.7177,
) -> np.ndarray:
    """Transform Mx2 waypoints from QLabs frame to Cartographer map frame.

    The transform angle 0.7177 rad is derived from the reference heading:
    2*pi - (-44.7 % 2*pi) = 0.7177 rad. This is empirically calibrated
    and matches the reference 2025 MPC_node.py.
    """
    if origin_heading_rad is not None:
        theta = origin_heading_rad
    else:
        theta = math.radians(-origin_heading_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)

    translated = qlabs_waypoints.copy()
    translated[:, 0] -= origin_x
    translated[:, 1] -= origin_y

    result = np.zeros_like(translated)
    # R(+θ): rotate translated coords into map frame
    result[:, 0] = translated[:, 0] * cos_t - translated[:, 1] * sin_t
    result[:, 1] = translated[:, 0] * sin_t + translated[:, 1] * cos_t
    return result
