"""
Path planning backend interface for ACC self-driving stack.

Defines the abstract base class for path planning backends and provides
concrete implementations for A*, Experience A*, Weighted A*, and Dijkstra.

Available backends (selected via config/modules.yaml or path_planning_algorithm param):
  - astar:            Standard A* (optimal, 14 expanded, 0.14ms)
  - experience_astar: A* with path caching (optimal, cached calls 13x faster) [DEFAULT]
  - weighted_astar:   Epsilon-inflated heuristic (bounded suboptimal)
  - dijkstra:         Dijkstra's algorithm (optimal, more expansions than A*)

Benchmark: All algorithms produce identical optimal paths on the SDCS road
graph. Experience A* is preferred for the taxi mission because the 3 fixed
routes (hub->pickup, pickup->dropoff, dropoff->hub) benefit from caching.

How to switch:
  ROS param:   path_planning_algorithm:=astar
  Launch arg:  ros2 launch ... path_planning_algorithm:=weighted_astar
  Config file: config/modules.yaml -> path_planning.backend
"""

import heapq
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np


class PathPlannerBackend(ABC):
    """Abstract base for path planning backends.

    Subclasses implement plan() to find a node sequence on the SDCS road
    graph. The RoadGraph class handles path generation (SCSPath waypoints)
    from the node sequence.
    """

    name: str = "base"

    @abstractmethod
    def plan(self, roadmap, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        """Find node sequence from start to goal on the road graph.

        Args:
            roadmap: SDCSRoadMap (or RoadMap) instance with nodes and edges.
            start_idx: Index of start node.
            goal_idx: Index of goal node.

        Returns:
            List of node indices forming the path, or None if no path found.
        """
        ...


class AStarPlanner(PathPlannerBackend):
    """Standard A* search. Optimal and complete."""

    name = "astar"

    def plan(self, roadmap, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        start = roadmap.nodes[start_idx]
        goal = roadmap.nodes[goal_idx]

        if start == goal:
            return None

        open_set = []
        closed_set = set()
        h = np.linalg.norm(goal.pose[:2, :] - start.pose[:2, :])
        heapq.heappush(open_set, (h, id(start), start))

        g_score = {node: float('inf') for node in roadmap.nodes}
        g_score[start] = 0
        came_from = {node: None for node in roadmap.nodes}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # Reconstruct node sequence
                node_seq = [roadmap.nodes.index(goal)]
                node = goal
                while came_from[node] is not None:
                    prev_node, _ = came_from[node]
                    node_seq.insert(0, roadmap.nodes.index(prev_node))
                    node = prev_node
                return node_seq

            if current in closed_set:
                continue
            closed_set.add(current)

            for edge in current.outEdges:
                neighbor = edge.toNode
                if neighbor in closed_set or edge.length is None:
                    continue
                tentative_g = g_score[current] + edge.length
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, edge)
                    g_score[neighbor] = tentative_g
                    h_score = np.linalg.norm(goal.pose[:2, :] - neighbor.pose[:2, :])
                    heapq.heappush(open_set, (tentative_g + h_score,
                                              id(neighbor), neighbor))

        return None


class ExperienceAStarPlanner(PathPlannerBackend):
    """A* with path caching for repeated queries.

    Identical paths to standard A*. Cached calls return in O(n) validation
    time instead of O(E log V) search time. Benchmark shows 13x speedup
    on cached queries (0.008ms vs 0.14ms).
    """

    name = "experience_astar"

    def __init__(self):
        self._cache = {}  # (start_idx, goal_idx) -> node_seq
        self._astar = AStarPlanner()

    def plan(self, roadmap, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        key = (start_idx, goal_idx)

        if key in self._cache:
            cached_seq = self._cache[key]
            if self._validate(roadmap, cached_seq):
                return list(cached_seq)
            else:
                del self._cache[key]

        node_seq = self._astar.plan(roadmap, start_idx, goal_idx)
        if node_seq is not None:
            self._cache[key] = list(node_seq)
        return node_seq

    @staticmethod
    def _validate(roadmap, node_seq: list) -> bool:
        """Check that cached node sequence is still traversable."""
        for i in range(len(node_seq) - 1):
            from_node = roadmap.nodes[node_seq[i]]
            to_node = roadmap.nodes[node_seq[i + 1]]
            if not any(e.toNode == to_node and e.length is not None
                       for e in from_node.outEdges):
                return False
        return True


class WeightedAStarPlanner(PathPlannerBackend):
    """Weighted A*: f(n) = g(n) + epsilon * h(n).

    Bounded suboptimality: cost <= epsilon * optimal. On the small SDCS
    graph, produces the same path as A* with fewer node expansions.
    """

    name = "weighted_astar"

    def __init__(self, epsilon: float = 1.5):
        self.epsilon = epsilon

    def plan(self, roadmap, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        start = roadmap.nodes[start_idx]
        goal = roadmap.nodes[goal_idx]

        if start == goal:
            return None

        open_set = []
        closed_set = set()
        h = np.linalg.norm(goal.pose[:2, :] - start.pose[:2, :])
        heapq.heappush(open_set, (self.epsilon * h, id(start), start))

        g_score = {node: float('inf') for node in roadmap.nodes}
        g_score[start] = 0
        came_from = {node: None for node in roadmap.nodes}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                node_seq = [roadmap.nodes.index(goal)]
                node = goal
                while came_from[node] is not None:
                    prev_node, _ = came_from[node]
                    node_seq.insert(0, roadmap.nodes.index(prev_node))
                    node = prev_node
                return node_seq

            if current in closed_set:
                continue
            closed_set.add(current)

            for edge in current.outEdges:
                neighbor = edge.toNode
                if neighbor in closed_set or edge.length is None:
                    continue
                tentative_g = g_score[current] + edge.length
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, edge)
                    g_score[neighbor] = tentative_g
                    h_score = np.linalg.norm(goal.pose[:2, :] - neighbor.pose[:2, :])
                    heapq.heappush(open_set, (tentative_g + self.epsilon * h_score,
                                              id(neighbor), neighbor))

        return None


class DijkstraPlanner(PathPlannerBackend):
    """Dijkstra's algorithm. A* with heuristic=0. Optimal baseline."""

    name = "dijkstra"

    def plan(self, roadmap, start_idx: int, goal_idx: int) -> Optional[List[int]]:
        start = roadmap.nodes[start_idx]
        goal = roadmap.nodes[goal_idx]

        if start == goal:
            return None

        open_set = []
        closed_set = set()
        heapq.heappush(open_set, (0.0, id(start), start))

        g_score = {node: float('inf') for node in roadmap.nodes}
        g_score[start] = 0
        came_from = {node: None for node in roadmap.nodes}

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                node_seq = [roadmap.nodes.index(goal)]
                node = goal
                while came_from[node] is not None:
                    prev_node, _ = came_from[node]
                    node_seq.insert(0, roadmap.nodes.index(prev_node))
                    node = prev_node
                return node_seq

            if current in closed_set:
                continue
            closed_set.add(current)

            for edge in current.outEdges:
                neighbor = edge.toNode
                if neighbor in closed_set or edge.length is None:
                    continue
                tentative_g = g_score[current] + edge.length
                if tentative_g < g_score[neighbor]:
                    came_from[neighbor] = (current, edge)
                    g_score[neighbor] = tentative_g
                    heapq.heappush(open_set, (tentative_g, id(neighbor), neighbor))

        return None


def create_planner(name: str, **kwargs) -> PathPlannerBackend:
    """Factory function to create a planner by name.

    Args:
        name: One of 'astar', 'experience_astar', 'weighted_astar', 'dijkstra'.
        **kwargs: Passed to planner constructor (e.g., epsilon for weighted_astar).
    """
    planners = {
        'astar': AStarPlanner,
        'experience_astar': ExperienceAStarPlanner,
        'weighted_astar': WeightedAStarPlanner,
        'dijkstra': DijkstraPlanner,
    }
    cls = planners.get(name)
    if cls is None:
        raise ValueError(f"Unknown planner: {name}. Available: {list(planners.keys())}")

    if name == 'weighted_astar':
        return cls(epsilon=kwargs.get('epsilon', kwargs.get('weighted_epsilon', 1.5)))
    return cls()
