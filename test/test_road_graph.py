"""Tests for road_graph.py - SDCSRoadMap-based path planner."""
import math
import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from acc_stage1_mission.road_graph import (
    RoadGraph, SDCSRoadMap, SCSPath, qlabs_path_to_map_path,
    _wrap_to_pi, _wrap_to_2pi, _signed_angle,
)

# Mission locations in QLabs frame
HUB = (-1.205, -0.83)
PICKUP = (0.125, 4.395)
DROPOFF = (-0.905, 0.800)

# Reference node positions for validation
NODE_1 = (0.269, 0.081)    # Northbound lower main road
NODE_8 = (-0.749, 1.101)   # Westbound E-W road
NODE_10 = (-1.282, -0.460) # Hub exit
NODE_13 = (0.269, 1.850)   # Northbound upper intersection
NODE_20 = (0.000, 4.497)   # Near pickup

TRANSFORM = dict(origin_x=-1.205, origin_y=-0.83, origin_heading_deg=-44.7,
                 origin_heading_rad=0.7177)


class TestMathUtilities:

    def test_wrap_to_pi(self):
        assert abs(_wrap_to_pi(0.0)) < 1e-10
        assert abs(_wrap_to_pi(np.pi) - np.pi) < 1e-10 or \
               abs(_wrap_to_pi(np.pi) + np.pi) < 1e-10
        assert abs(_wrap_to_pi(3 * np.pi) - np.pi) < 1e-10 or \
               abs(_wrap_to_pi(3 * np.pi) + np.pi) < 1e-10
        assert abs(_wrap_to_pi(-np.pi / 2) + np.pi / 2) < 1e-10

    def test_wrap_to_2pi(self):
        assert abs(_wrap_to_2pi(0.0)) < 1e-10
        assert abs(_wrap_to_2pi(-np.pi) - np.pi) < 1e-10
        assert abs(_wrap_to_2pi(3 * np.pi) - np.pi) < 1e-10

    def test_signed_angle_single_vector(self):
        v = np.array([1.0, 0.0])
        assert abs(_signed_angle(v)) < 1e-10
        v2 = np.array([0.0, 1.0])
        assert abs(_signed_angle(v2) - np.pi / 2) < 1e-10

    def test_signed_angle_two_vectors(self):
        v1 = np.array([[1.0], [0.0]])
        v2 = np.array([[0.0], [1.0]])
        angle = _signed_angle(v1, v2)
        assert abs(angle - np.pi / 2) < 1e-6


class TestSCSPath:

    def test_straight_line(self):
        """SCSPath with radius=0 should produce a straight line."""
        start = np.array([[0.0], [0.0], [np.pi / 2]])
        end = np.array([[0.0], [1.0], [np.pi / 2]])
        path, length = SCSPath(start, end, radius=0, stepSize=0.01)
        assert path is not None
        assert length is not None
        assert abs(length - 1.0) < 0.01
        # All x values should be ~0
        assert np.allclose(path[0, :], 0.0, atol=0.01)

    def test_curved_path(self):
        """SCSPath with nonzero radius should produce a curved path."""
        start = np.array([[0.0], [0.0], [np.pi / 2]])
        end = np.array([[1.0], [1.0], [0.0]])
        path, length = SCSPath(start, end, radius=0.5, stepSize=0.01)
        assert path is not None
        assert length > 0

    def test_returns_none_for_impossible_path(self):
        """Some configurations are geometrically impossible for SCS."""
        start = np.array([[0.0], [0.0], [0.0]])
        end = np.array([[-1.0], [0.0], [0.0]])  # Behind and same heading
        path, length = SCSPath(start, end, radius=0.5, stepSize=0.01)
        # May or may not be None depending on geometry; just check no crash
        if path is not None:
            assert length >= 0


class TestSDCSRoadMap:

    @pytest.fixture
    def roadmap(self):
        return SDCSRoadMap()

    def test_node_count(self, roadmap):
        assert len(roadmap.nodes) == 25  # 24 original + spawn node 24

    def test_edge_count(self, roadmap):
        assert len(roadmap.edges) == 45  # 42 original + 3 spawn edges

    def test_all_edges_have_length(self, roadmap):
        for edge in roadmap.edges:
            fi = roadmap.nodes.index(edge.fromNode)
            ti = roadmap.nodes.index(edge.toNode)
            # Waypoints may be an empty array for very short straight-line edges
            assert edge.waypoints is not None, \
                f"Edge {fi}->{ti} has no waypoints"
            assert edge.length is not None, \
                f"Edge {fi}->{ti} has no length"

    def test_node_positions_match_reference(self, roadmap):
        """Verify key node positions match reference repo values."""
        n1 = roadmap.nodes[1].pose[:, 0]
        assert abs(n1[0] - 0.269) < 0.01
        assert abs(n1[1] - 0.081) < 0.01

        n10 = roadmap.nodes[10].pose[:, 0]
        assert abs(n10[0] - (-1.282)) < 0.01
        assert abs(n10[1] - (-0.460)) < 0.01

        n20 = roadmap.nodes[20].pose[:, 0]
        assert abs(n20[0] - 0.000) < 0.01
        assert abs(n20[1] - 4.497) < 0.01

    def test_astar_finds_path_10_to_1(self, roadmap):
        path = roadmap.find_shortest_path(10, 1)
        assert path is not None
        assert path.shape[0] == 2
        assert path.shape[1] > 10

    def test_astar_finds_path_10_to_20(self, roadmap):
        path = roadmap.find_shortest_path(10, 20)
        assert path is not None
        # This should go through multiple nodes

    def test_generate_path_hub_to_pickup(self, roadmap):
        path = roadmap.generate_path([24, 1, 13, 19, 17, 20])
        assert path is not None
        assert path.shape[0] == 2
        assert path.shape[1] > 100

    def test_spawn_node_position(self, roadmap):
        """Spawn node 24 should be at hub position."""
        n24 = roadmap.nodes[24].pose[:, 0]
        assert abs(n24[0] - (-1.205)) < 0.01
        assert abs(n24[1] - (-0.83)) < 0.01

    def test_spawn_node_edges(self, roadmap):
        """Spawn node 24 should have edges to nodes 1, 2, and from node 10."""
        n24 = roadmap.nodes[24]
        out_targets = {roadmap.nodes.index(e.toNode) for e in n24.outEdges}
        in_sources = {roadmap.nodes.index(e.fromNode) for e in n24.inEdges}
        assert 1 in out_targets, "Missing edge 24->1"
        assert 2 in out_targets, "Missing edge 24->2"
        assert 10 in in_sources, "Missing edge 10->24"


class TestRoadGraph:

    @pytest.fixture
    def graph(self):
        return RoadGraph(ds=0.01)

    def test_has_three_routes(self, graph):
        names = graph.get_route_names()
        assert 'hub_to_pickup' in names
        assert 'pickup_to_dropoff' in names
        assert 'dropoff_to_hub' in names

    def test_hub_to_pickup_endpoints(self, graph):
        route = graph.get_route('hub_to_pickup')
        assert route is not None
        assert len(route) > 100
        # Start should be at/near hub (spawn node 24)
        assert np.linalg.norm(route[0] - np.array(HUB)) < 0.05
        # End near pickup (sliced from loop — closest pass, not exact point)
        assert np.linalg.norm(route[-1] - np.array(PICKUP)) < 0.15

    def test_pickup_to_dropoff_endpoints(self, graph):
        route = graph.get_route('pickup_to_dropoff')
        assert route is not None
        assert len(route) > 100
        # Sliced from loop at closest pass to pickup/dropoff
        assert np.linalg.norm(route[0] - np.array(PICKUP)) < 0.15
        assert np.linalg.norm(route[-1] - np.array(DROPOFF)) < 0.10

    def test_dropoff_to_hub_endpoints(self, graph):
        route = graph.get_route('dropoff_to_hub')
        assert route is not None
        assert len(route) > 50
        assert np.linalg.norm(route[0] - np.array(DROPOFF)) < 0.10
        # End is at node 10 (-1.295, -0.460), 0.38m from hub spawn point
        assert np.linalg.norm(route[-1] - np.array(HUB)) < 0.45

    def test_waypoint_spacing(self, graph):
        for name in graph.get_route_names():
            route = graph.get_route(name)
            diffs = np.diff(route, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            assert np.mean(seg_lengths) < 0.02, f"{name}: mean spacing too large"
            # B-spline resample at 0.001m produces ~1mm mean spacing
            assert np.mean(seg_lengths) > 0.0005, f"{name}: mean spacing too small"

    def test_no_sharp_jumps(self, graph):
        for name in graph.get_route_names():
            route = graph.get_route(name)
            diffs = np.diff(route, axis=0)
            seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
            assert np.max(seg_lengths) < 0.05, \
                f"{name}: max segment {np.max(seg_lengths):.4f}m too long"

    def test_hub_to_pickup_passes_through_main_road(self, graph):
        """The hub->pickup route must pass through the northbound lane (x~0.27)."""
        route = graph.get_route('hub_to_pickup')
        mask = (route[:, 1] > 0.1) & (route[:, 1] < 1.8)
        main_road_section = route[mask]
        assert len(main_road_section) > 10, "Route doesn't traverse main road"
        mean_x = np.mean(main_road_section[:, 0])
        assert mean_x > 0.0, \
            f"Main road section at mean x={mean_x:.3f}, should be near 0.27"

    def test_pickup_to_dropoff_goes_through_west_side(self, graph):
        """Pickup->dropoff route goes via nodes 20->22->9 (western side)."""
        route = graph.get_route('pickup_to_dropoff')
        # Route should pass through the western side (x < -1.5)
        west_mask = route[:, 0] < -1.5
        assert np.any(west_mask), "Route doesn't pass through western side"


class TestCoordinateTransform:

    def test_hub_maps_to_origin(self):
        """Hub point (-1.205, -0.83) should map to (0, 0) with theta=0.7177."""
        hub_pts = np.array([[HUB[0], HUB[1]]])
        map_pts = qlabs_path_to_map_path(hub_pts, **TRANSFORM)
        assert abs(map_pts[0, 0]) < 0.01
        assert abs(map_pts[0, 1]) < 0.01

    def test_transform_uses_calibrated_angle(self):
        """Verify the calibrated angle is used with R(+θ) convention."""
        # The transform is: translate to origin, then R(+θ) where θ = radians(44.7) = 0.7803
        # A point at (1, 0) relative to hub maps to (cos(θ), sin(θ))
        theta = TRANSFORM.get('origin_heading_rad', math.radians(-TRANSFORM.get('origin_heading_deg', -44.7)))
        test_pt = np.array([[HUB[0] + 1.0, HUB[1]]])
        map_pts = qlabs_path_to_map_path(test_pt, **TRANSFORM)
        expected_x = math.cos(theta)
        expected_y = math.sin(theta)
        assert abs(map_pts[0, 0] - expected_x) < 0.01
        assert abs(map_pts[0, 1] - expected_y) < 0.01

    def test_batch_transform_preserves_count(self):
        graph = RoadGraph(ds=0.05)
        route = graph.get_route('hub_to_pickup')
        map_route = qlabs_path_to_map_path(route, **TRANSFORM)
        assert len(map_route) == len(route)
        assert np.linalg.norm(map_route[0]) < 0.1

    def test_roundtrip_consistency(self):
        # Forward: translate then R(+θ). Inverse: R(-θ) then translate back.
        theta = TRANSFORM.get('origin_heading_rad', math.radians(-TRANSFORM.get('origin_heading_deg', -44.7)))

        test_pts = np.array([[0.0, 2.0], [-0.5, 1.0], [0.125, 4.395]])
        map_pts = qlabs_path_to_map_path(test_pts, **TRANSFORM)

        # Inverse rotation: R(-θ) = R^T(θ)
        cos_inv = math.cos(-theta)
        sin_inv = math.sin(-theta)
        rotated = np.zeros_like(map_pts)
        rotated[:, 0] = map_pts[:, 0] * cos_inv - map_pts[:, 1] * sin_inv
        rotated[:, 1] = map_pts[:, 0] * sin_inv + map_pts[:, 1] * cos_inv
        recovered = rotated + np.array([TRANSFORM['origin_x'], TRANSFORM['origin_y']])
        assert np.allclose(recovered, test_pts, atol=1e-10)


class TestPathOnRoad:

    @pytest.fixture
    def graph(self):
        return RoadGraph(ds=0.01)

    def test_hub_to_pickup_stays_on_road(self, graph):
        route = graph.get_route('hub_to_pickup')
        assert np.all(route[:, 0] >= -1.8), "Path goes too far west"
        assert np.all(route[:, 0] <= 2.5), "Path goes too far east"
        assert route[0, 1] < -0.5
        assert route[-1, 1] > 4.0

    def test_pickup_to_dropoff_stays_on_road(self, graph):
        route = graph.get_route('pickup_to_dropoff')
        # Route goes via western side (nodes 20->22->9)
        assert np.all(route[:, 0] >= -2.3), "Path goes too far west"
        assert np.all(route[:, 0] <= 1.5), "Path goes too far east"

    def test_dropoff_to_hub_stays_on_road(self, graph):
        route = graph.get_route('dropoff_to_hub')
        assert np.all(route[:, 0] >= -2.2), "Path goes too far west"

    def test_total_path_length_reasonable(self, graph):
        for name in graph.get_route_names():
            route = graph.get_route(name)
            diffs = np.diff(route, axis=0)
            total_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
            assert total_length > 1.0, f"{name}: path too short ({total_length:.2f}m)"
            assert total_length < 16.0, f"{name}: path too long ({total_length:.2f}m)"
            print(f"  {name}: {len(route)} waypoints, {total_length:.2f}m")

    def test_hub_to_pickup_is_longer_than_direct(self, graph):
        route = graph.get_route('hub_to_pickup')
        diffs = np.diff(route, axis=0)
        total_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        direct_dist = np.linalg.norm(np.array(PICKUP) - np.array(HUB))
        assert total_length > direct_dist, "Route should be longer than straight line"
        assert total_length > 6.0, f"Hub->pickup route too short ({total_length:.2f}m)"

    def test_no_route_goes_backward(self, graph):
        for name in ['hub_to_pickup']:
            route = graph.get_route(name)
            y_start = route[0, 1]
            y_end = route[-1, 1]
            assert y_end > y_start + 2.0, \
                f"{name}: y from {y_start:.2f} to {y_end:.2f}, should increase"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
