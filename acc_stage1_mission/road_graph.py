"""
Road-network-based path planner for SDCS competition track.

Ported from the reference repo's SDCSRoadMap (mats.py), SCSPath and
RoadMap (path_planning.py) classes. Uses Straight-Curve-Straight (SCS)
path generation with A* graph search for proper road-following geometry.

All coordinates are in QLabs world frame (meters).
"""

import math
import heapq
import numpy as np
from typing import List, Tuple, Optional

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

    direction = 1 if _signed_angle(t1, p2 - p1) > 0 else -1

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

    def generate_path(self, nodeSequence):
        """Generate shortest path through a sequence of node indices."""
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
        return path


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

        # Spawn node (node 24): the vehicle's starting position at the hub.
        # Ported from reference repo (MPC_node.py:127-131).
        # This eliminates the gap between hub and the road graph.
        self.add_node([-1.205, -0.83, -44.7 % (2 * np.pi)])  # node 24
        self.add_edge(24, 2, radius=0.0)       # spawn -> node 2
        self.add_edge(10, 24, radius=0.0)      # node 10 -> spawn (straight - nodes are close)
        self.add_edge(24, 1, radius=0.866326)  # spawn -> node 1


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

    Uses the reference repo's SDCSRoadMap with SCSPath geometry and A* search
    to generate proper lane-following waypoints.
    """

    # Node sequences for each mission leg (A* finds shortest path between
    # consecutive nodes in each sequence)
    # Node 24 is the spawn node at the hub position (-1.205, -0.83).
    # Pickup is near node 20. Dropoff is near node 8.
    #
    # hub_to_pickup:  24 -> 1 -> 13 -> 19 -> 17 -> 20
    # pickup_to_dropoff: 21 -> 16 -> 18 -> 11 -> 12 -> 8
    # dropoff_to_hub: 8 -> 10 -> 24

    _ROUTE_NODE_SEQUENCES = {
        'hub_to_pickup': [24, 1, 13, 19, 17, 20],
        'pickup_to_dropoff': [21, 16, 18, 11, 12, 8],
        'dropoff_to_hub': [8, 10, 24],
    }

    def __init__(self, ds: float = 0.01):
        self.ds = ds
        self._roadmap = SDCSRoadMap(leftHandTraffic=False, useSmallMap=False)
        self._routes = {}

        for name, node_seq in self._ROUTE_NODE_SEQUENCES.items():
            path_2xN = self._roadmap.generate_path(node_seq)
            if path_2xN is not None:
                # Convert from 2xN to Nx2 for consistency with rest of codebase
                route = path_2xN.T
                # Attach mission endpoints where the route doesn't already
                # start/end at the exact location (spawn node handles hub).
                route = self._attach_endpoints(name, route)
                self._routes[name] = route

    @staticmethod
    def _interpolate_gap(p1: np.ndarray, p2: np.ndarray,
                         ds: float = 0.01) -> np.ndarray:
        """Create evenly-spaced points between p1 and p2 (exclusive)."""
        dist = np.linalg.norm(p2 - p1)
        if dist < ds * 1.5:
            return np.empty((0, 2))
        n = max(int(dist / ds), 2)
        t = np.linspace(0, 1, n + 1)[1:-1]  # exclude endpoints
        pts = np.outer(1 - t, p1) + np.outer(t, p2)
        return pts

    def _attach_endpoints(self, route_name: str,
                          route: np.ndarray) -> np.ndarray:
        """Prepend start and append goal mission location to route.

        With spawn node 24 at the hub, hub_to_pickup already starts at hub
        and dropoff_to_hub already ends at hub. Only non-hub endpoints
        (pickup, dropoff) need attachment.
        """
        if route_name == 'hub_to_pickup':
            # Start is hub (spawn node 24) — already in graph
            # End is pickup — may need attachment
            end = np.array(PICKUP)
            if np.linalg.norm(route[-1] - end) > 0.02:
                gap = self._interpolate_gap(route[-1], end, self.ds)
                route = np.vstack([route, gap, end.reshape(1, 2)])
        elif route_name == 'pickup_to_dropoff':
            start = np.array(PICKUP)
            end = np.array(DROPOFF)
            if np.linalg.norm(route[0] - start) > 0.02:
                gap = self._interpolate_gap(start, route[0], self.ds)
                route = np.vstack([start.reshape(1, 2), gap, route])
            if np.linalg.norm(route[-1] - end) > 0.02:
                gap = self._interpolate_gap(route[-1], end, self.ds)
                route = np.vstack([route, gap, end.reshape(1, 2)])
        elif route_name == 'dropoff_to_hub':
            # Start is dropoff — may need attachment
            # End is hub (spawn node 24) — already in graph
            start = np.array(DROPOFF)
            if np.linalg.norm(route[0] - start) > 0.02:
                gap = self._interpolate_gap(start, route[0], self.ds)
                route = np.vstack([start.reshape(1, 2), gap, route])
        return route

    def get_route(self, route_name: str) -> Optional[np.ndarray]:
        """Get pre-computed route waypoints. Returns Mx2 array or None."""
        return self._routes.get(route_name)

    def get_route_names(self) -> List[str]:
        return list(self._routes.keys())

    def get_route_for_leg(self, start_label: str,
                          goal_label: str) -> Optional[str]:
        """Determine which route to use based on mission leg labels."""
        sl = start_label.lower() if start_label else ""
        gl = goal_label.lower() if goal_label else ""
        if 'pickup' in gl and ('hub' in sl or not sl):
            return 'hub_to_pickup'
        elif 'dropoff' in gl and 'pickup' in sl:
            return 'pickup_to_dropoff'
        elif 'hub' in gl and 'dropoff' in sl:
            return 'dropoff_to_hub'
        return None

    def plan_path_for_mission_leg(
        self,
        route_name: str,
        current_pos_qlabs: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        Get path for a specific mission leg, starting from current position.

        Returns Mx2 array of waypoints in QLabs frame, or None.
        """
        route = self._routes.get(route_name)
        if route is None:
            return None

        pos = np.array(current_pos_qlabs[:2])
        start_idx = self._find_closest_idx(route, pos)
        route_segment = route[start_idx:]

        if len(route_segment) < 5:
            route_segment = route[max(0, len(route) - 20):]

        if np.linalg.norm(pos - route_segment[0]) > 0.02:
            return np.vstack([pos.reshape(1, 2), route_segment])
        return route_segment.copy()

    def plan_path(self, start_qlabs: Tuple[float, float],
                  goal_qlabs: Tuple[float, float]) -> Optional[np.ndarray]:
        """Find the best route from start to goal."""
        start = np.array(start_qlabs[:2])
        goal = np.array(goal_qlabs[:2])

        best_route = None
        best_score = float('inf')

        for name, waypoints in self._routes.items():
            route_start = waypoints[0]
            route_end = waypoints[-1]
            start_dist = np.linalg.norm(start - route_start)
            end_dist = np.linalg.norm(goal - route_end)
            score = start_dist + end_dist
            if score < best_score:
                best_score = score
                best_route = waypoints

        if best_route is None:
            return None

        start_idx = self._find_closest_idx(best_route, start)
        goal_idx = self._find_closest_idx(best_route, goal)

        if goal_idx <= start_idx:
            return best_route.copy()
        return best_route[start_idx:goal_idx + 1].copy()

    def plan_path_from_pose(
        self,
        current_pos_qlabs: Tuple[float, float],
        goal_qlabs: Tuple[float, float]
    ) -> Optional[np.ndarray]:
        """Plan a path from current position to goal."""
        pos = np.array(current_pos_qlabs[:2])
        goal = np.array(goal_qlabs[:2])

        best_route = None
        best_score = float('inf')

        for name, waypoints in self._routes.items():
            route_end = waypoints[-1]
            end_dist = np.linalg.norm(goal - route_end)
            closest_idx = self._find_closest_idx(waypoints, pos)
            start_dist = np.linalg.norm(pos - waypoints[closest_idx])
            score = start_dist + 2.0 * end_dist
            if score < best_score:
                best_score = score
                best_route = waypoints

        if best_route is None:
            return None

        start_idx = self._find_closest_idx(best_route, pos)
        route_segment = best_route[start_idx:]

        if len(route_segment) < 5:
            route_segment = best_route[max(0, len(best_route) - 20):]

        if np.linalg.norm(pos - route_segment[0]) > 0.02:
            return np.vstack([pos.reshape(1, 2), route_segment])
        return route_segment.copy()

    @staticmethod
    def _find_closest_idx(waypoints: np.ndarray, point: np.ndarray) -> int:
        dists = np.linalg.norm(waypoints - point, axis=1)
        return int(np.argmin(dists))


def qlabs_path_to_map_path(
    qlabs_waypoints: np.ndarray,
    origin_x: float = -1.205,
    origin_y: float = -0.83,
    origin_heading_deg: float = -44.7,
    origin_heading_rad: float = None,
) -> np.ndarray:
    """Transform Mx2 waypoints from QLabs frame to Cartographer map frame."""
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
    result[:, 0] = translated[:, 0] * cos_t + translated[:, 1] * sin_t
    result[:, 1] = -translated[:, 0] * sin_t + translated[:, 1] * cos_t
    return result
