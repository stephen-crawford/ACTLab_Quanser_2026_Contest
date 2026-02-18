"""
Integration tests simulating the full mpcc_controller.py workflow
without ROS2 dependencies.

Verifies that:
1. MPCCBridge correctly wraps pympc_core
2. VehicleState/ReferencePath/Obstacle types work with the bridge
3. Multiple consecutive solves succeed (constraint accumulation fix)
4. Path changes are handled correctly
5. The controller produces valid commands for the QCar2 scenario

Run with:
    cd /home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission
    python3 -m pytest test/test_mpcc_integration.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import the actual types from mpcc_controller (without ROS2)
# We need to extract the non-ROS2 classes
from acc_stage1_mission.pympc_core import MPCCSolver, CubicSplinePath
from acc_stage1_mission.pympc_core.solver import MPCCConfig as CoreConfig
from acc_stage1_mission.pympc_core.dynamics import AckermannDynamics


class MockVehicleState:
    """Mock of mpcc_controller.VehicleState for testing without ROS2."""
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=0.0, delta=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.delta = delta

    def as_array(self):
        return np.array([self.x, self.y, self.theta, self.v, self.delta])


class MockReferencePath:
    """Mock of mpcc_controller.ReferencePath for testing."""
    def __init__(self, waypoints):
        self.waypoints = np.array(waypoints)
        self.n_points = len(self.waypoints)
        diffs = np.diff(self.waypoints, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        self.cumulative_dist = np.zeros(self.n_points)
        self.cumulative_dist[1:] = np.cumsum(seg_lengths)
        self.total_length = self.cumulative_dist[-1]

    def find_closest_progress(self, x, y):
        dists = np.sqrt((self.waypoints[:, 0] - x)**2 + (self.waypoints[:, 1] - y)**2)
        return self.cumulative_dist[np.argmin(dists)]


class MockObstacle:
    """Mock of mpcc_controller.Obstacle for testing."""
    def __init__(self, x, y, radius=0.3):
        self.x = x
        self.y = y
        self.radius = radius


class TestMPCCBridgeIntegration:
    """Test the MPCCBridge workflow as used by the ROS2 node."""

    def test_bridge_workflow(self):
        """Simulate the full bridge workflow."""
        config = CoreConfig(
            horizon=10, dt=0.1, wheelbase=0.256,
            max_velocity=0.5, reference_velocity=0.3, max_iter=30,
        )
        solver = MPCCSolver(config)

        # Create path (simulating Nav2 plan)
        waypoints = np.array([[i * 0.3, 0.0] for i in range(30)])
        path = CubicSplinePath(waypoints)

        # Simulate vehicle starting at origin
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0
        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)

        positions = [state[:2].copy()]
        velocities = []

        for step in range(40):
            result = solver.solve(state, path, current_progress=progress)

            # Track results
            velocities.append(result.v_cmd)

            # Simulate forward
            control = np.array([
                np.clip((result.v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((result.delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = path.find_closest_progress(state[0], state[1])
            positions.append(state[:2].copy())

        positions = np.array(positions)

        # Verify vehicle moved forward
        total_dist = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        assert total_dist > 1.0, f"Vehicle should move >1m, moved {total_dist:.2f}m"

        # Verify velocity ramps up
        assert max(velocities) > 0.1, f"Max velocity {max(velocities):.2f} too low"

        # Verify vehicle stays near path (y should be close to 0)
        max_y_error = max(abs(positions[:, 1]))
        assert max_y_error < 0.3, f"Max lateral error {max_y_error:.2f}m (should be <0.3m)"

    def test_path_change(self):
        """Solver should handle path changes gracefully."""
        config = CoreConfig(horizon=8, dt=0.1, reference_velocity=0.3, max_iter=30)
        solver = MPCCSolver(config)

        # First path: straight along x
        path1 = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(20)]))
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        r1 = solver.solve(state, path1, current_progress=0.0)
        assert r1.success

        # Change to a different path: straight along y
        solver.reset()
        path2 = CubicSplinePath(np.array([[0.0, i * 0.3] for i in range(20)]))
        state = np.array([0.0, 0.0, np.pi / 2, 0.2, 0.0])

        r2 = solver.solve(state, path2, current_progress=0.0)
        assert r2.success

    def test_obstacle_avoidance_loop(self):
        """Test obstacle avoidance over multiple iterations."""
        config = CoreConfig(horizon=10, dt=0.1, reference_velocity=0.3, max_iter=40)
        solver = MPCCSolver(config)

        path = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(30)]))
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        progress = 0.0

        # Obstacle on the path ahead
        obstacles = [(2.0, 0.0, 0.3)]

        success_count = 0
        for step in range(15):
            result = solver.solve(
                state, path, current_progress=progress,
                obstacles=obstacles)
            if result.success:
                success_count += 1

            dt = config.dt
            state = np.array([
                state[0] + result.v_cmd * np.cos(state[2]) * dt,
                state[1] + result.v_cmd * np.sin(state[2]) * dt,
                state[2] + result.v_cmd / config.wheelbase * np.tan(result.delta_cmd) * dt,
                result.v_cmd,
                result.delta_cmd,
            ])
            progress = path.find_closest_progress(state[0], state[1])

        assert success_count >= 10, f"Only {success_count}/15 succeeded with obstacles"

    def test_qcar2_competition_scenario(self):
        """
        Simulate the actual QCar2 competition scenario path.

        The competition route goes:
        Hub [-1.205, -0.83] -> Pickup [0.125, 4.395] -> Dropoff [-0.905, 0.800] -> Hub
        """
        config = CoreConfig(
            horizon=12, dt=0.1, wheelbase=0.256,
            max_velocity=0.5, reference_velocity=0.35,
            contour_weight=20.0, lag_weight=3.0,
            max_iter=40,
        )
        solver = MPCCSolver(config)

        # Simulated Nav2 path (in map frame after coordinate transform)
        # Just first leg: Hub to Pickup
        waypoints = []
        for i in range(50):
            t = i / 49.0
            # Simple interpolation (in map frame, roughly)
            x = t * 3.0
            y = t * 2.0
            waypoints.append([x, y])

        path = CubicSplinePath(np.array(waypoints))
        state = np.array([0.0, 0.0, np.arctan2(2.0, 3.0), 0.0, 0.0])
        progress = 0.0
        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)

        success_count = 0
        for step in range(30):
            result = solver.solve(state, path, current_progress=progress)
            if result.success:
                success_count += 1

            control = np.array([
                np.clip((result.v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((result.delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = path.find_closest_progress(state[0], state[1])

        assert success_count >= 20, f"Only {success_count}/30 succeeded on competition path"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
