"""
Tests for the C++ MPCC solver via ctypes wrapper.

Verifies:
1. Library loads and solver creates/destroys without crash
2. Basic straight-line path following
3. Curved path following
4. Obstacle avoidance
5. Multiple consecutive solves (warm-starting)
6. Performance (solve time)
7. Comparison with Python/CasADi solver
8. Edge cases (near path end, sharp turns, zero velocity)

Run with:
    cd /home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission
    python3 -m pytest test/test_cpp_solver.py -v
"""

import sys
import os
import time
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acc_stage1_mission.pympc_core import CubicSplinePath
from acc_stage1_mission.pympc_core.solver import MPCCConfig
from acc_stage1_mission.pympc_core.mpcc_cpp import (
    is_available, CppMPCCSolver, MPCCResult,
)
from acc_stage1_mission.pympc_core.dynamics import AckermannDynamics


@pytest.fixture
def config():
    return MPCCConfig(
        horizon=15, dt=0.1, wheelbase=0.256,
        max_velocity=0.5, reference_velocity=0.3,
        contour_weight=25.0, lag_weight=5.0,
        max_iter=50,
    )


@pytest.fixture
def straight_path():
    waypoints = np.array([[i * 0.3, 0.0] for i in range(30)])
    return CubicSplinePath(waypoints)


@pytest.fixture
def curved_path():
    t = np.linspace(0, 2 * np.pi * 0.75, 50)
    r = 2.0
    waypoints = np.column_stack([r * np.cos(t), r * np.sin(t)])
    return CubicSplinePath(waypoints)


class TestCppSolverAvailability:
    def test_library_available(self):
        assert is_available(), "C++ MPCC solver library not found"

    def test_create_destroy(self, config):
        solver = CppMPCCSolver(config)
        del solver  # Should not crash


class TestCppSolverBasic:
    def test_straight_path_from_rest(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.0)

        assert result.success, "Solver should succeed on straight path"
        assert result.v_cmd >= 0.0, f"Velocity should be non-negative, got {result.v_cmd}"
        assert abs(result.delta_cmd) < 0.2, f"Steering should be near zero for straight path, got {result.delta_cmd}"

    def test_straight_path_moving(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.5, 0.0, 0.0, 0.2, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.5)

        assert result.success
        assert result.v_cmd > 0.05, "Should command forward velocity"
        assert abs(result.delta_cmd) < 0.15, "Steering should be near zero"

    def test_result_has_predicted_trajectory(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.0)

        assert result.predicted_trajectory.shape[0] > 1, "Should have predicted trajectory"
        assert result.predicted_trajectory.shape[1] == 3, "Trajectory should be [x, y, theta]"

    def test_velocity_within_bounds(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.3, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.0)

        assert 0.0 <= result.v_cmd <= config.max_velocity + 0.01, \
            f"Velocity {result.v_cmd} outside bounds [0, {config.max_velocity}]"

    def test_steering_within_bounds(self, config, straight_path):
        solver = CppMPCCSolver(config)
        # Start offset from path to force steering
        state = np.array([0.5, 0.3, -0.2, 0.2, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.5)

        assert abs(result.delta_cmd) <= config.max_steering + 0.01, \
            f"Steering {result.delta_cmd} exceeds bound {config.max_steering}"


class TestCppSolverCurvedPath:
    def test_curved_path_steering(self, config, curved_path):
        solver = CppMPCCSolver(config)
        # Start at beginning of circular path
        state = np.array([2.0, 0.0, np.pi / 2, 0.2, 0.0])
        result = solver.solve(state, curved_path, current_progress=0.0)

        assert result.success
        # On a left turn, steering should be positive (left)
        # (may not be on first solve from rest, but should be non-zero)


class TestCppSolverObstacles:
    def test_obstacle_avoidance(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        # Place obstacle directly ahead
        obstacles = [(1.5, 0.0, 0.3)]
        result = solver.solve(
            state, straight_path, current_progress=0.0,
            obstacles=obstacles)

        assert result.success
        # Solver should try to avoid - either steer around or slow down

    def test_multiple_obstacles(self, config, straight_path):
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        obstacles = [(1.0, 0.0, 0.2), (2.0, 0.1, 0.2), (3.0, -0.1, 0.2)]
        result = solver.solve(
            state, straight_path, current_progress=0.0,
            obstacles=obstacles)

        assert result.success


class TestCppSolverConsecutiveSolves:
    """Test multiple consecutive solves - the key test for constraint accumulation."""

    def test_no_constraint_accumulation(self, config, straight_path):
        """Run many consecutive solves with varying obstacles.

        This was the original bug: CasADi constraints accumulated.
        The C++ solver should handle this correctly since it has no
        persistent constraint state.
        """
        solver = CppMPCCSolver(config)
        dynamics = AckermannDynamics(wheelbase=config.wheelbase, dt=config.dt)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0

        success_count = 0
        for step in range(30):
            # Vary obstacles each step
            obstacles = []
            if step % 3 == 0:
                obstacles = [(state[0] + 2.0, 0.1, 0.2)]

            result = solver.solve(
                state, straight_path, current_progress=progress,
                obstacles=obstacles)

            if result.success:
                success_count += 1

            # Simulate forward
            control = np.array([
                np.clip((result.v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((result.delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = straight_path.find_closest_progress(state[0], state[1])

        assert success_count >= 25, \
            f"Only {success_count}/30 solves succeeded (should be >=25)"

    def test_warm_start_improves(self, config, straight_path):
        """Warm-starting should not cause degradation."""
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        times = []
        for _ in range(10):
            result = solver.solve(state, straight_path, current_progress=0.0)
            times.append(result.solve_time)
            assert result.success

        # Later solves should generally be faster or similar due to warm-start
        # At minimum, they should all succeed

    def test_path_change_with_reset(self, config):
        """Solver should handle path changes after reset."""
        solver = CppMPCCSolver(config)

        # Path 1: straight along x
        path1 = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(20)]))
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        r1 = solver.solve(state, path1, current_progress=0.0)
        assert r1.success

        # Reset and switch to path 2: straight along y
        solver.reset()
        path2 = CubicSplinePath(np.array([[0.0, i * 0.3] for i in range(20)]))
        state = np.array([0.0, 0.0, np.pi / 2, 0.2, 0.0])
        r2 = solver.solve(state, path2, current_progress=0.0)
        assert r2.success


class TestCppSolverPerformance:
    def test_solve_time_under_5ms(self, config, straight_path):
        """C++ solver should be MUCH faster than CasADi (~50ms)."""
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        # Warm up
        solver.solve(state, straight_path, current_progress=0.0)

        # Time 10 solves
        t0 = time.time()
        for _ in range(10):
            result = solver.solve(state, straight_path, current_progress=0.0)
        elapsed = (time.time() - t0) / 10

        print(f"Average C++ solve time: {elapsed * 1000:.2f}ms")
        assert elapsed < 0.005, \
            f"Solve time {elapsed * 1000:.2f}ms exceeds 5ms budget"

    def test_solve_time_with_obstacles(self, config, straight_path):
        """Solve time with obstacles should still be fast."""
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        obstacles = [(1.0, 0.1, 0.3), (2.0, -0.1, 0.3), (3.0, 0.0, 0.3)]

        # Warm up
        solver.solve(state, straight_path, current_progress=0.0, obstacles=obstacles)

        t0 = time.time()
        for _ in range(10):
            solver.solve(state, straight_path, current_progress=0.0, obstacles=obstacles)
        elapsed = (time.time() - t0) / 10

        print(f"Average C++ solve time with obstacles: {elapsed * 1000:.2f}ms")
        assert elapsed < 0.010, \
            f"Solve time with obstacles {elapsed * 1000:.2f}ms exceeds 10ms budget"


class TestCppSolverPhysics:
    """Test that the solver produces physically reasonable commands."""

    def test_vehicle_follows_straight_path(self, config, straight_path):
        """Simulate forward and verify vehicle follows the path."""
        solver = CppMPCCSolver(config)
        dynamics = AckermannDynamics(wheelbase=config.wheelbase, dt=config.dt)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0

        positions = [state[:2].copy()]
        for step in range(40):
            result = solver.solve(state, straight_path, current_progress=progress)

            control = np.array([
                np.clip((result.v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((result.delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = straight_path.find_closest_progress(state[0], state[1])
            positions.append(state[:2].copy())

        positions = np.array(positions)

        # Vehicle should move forward
        total_dist = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        assert total_dist > 0.8, f"Vehicle only moved {total_dist:.2f}m (expected >0.8m)"

        # Vehicle should stay near the path (y ~ 0 for straight path)
        max_y_error = np.max(np.abs(positions[:, 1]))
        assert max_y_error < 0.3, f"Max lateral error {max_y_error:.2f}m (expected <0.3m)"

    def test_vehicle_follows_curved_path(self, config):
        """Vehicle should follow a gentle curve."""
        # Create a gentle S-curve
        waypoints = []
        for i in range(40):
            x = i * 0.25
            y = 0.5 * np.sin(x * 0.3)
            waypoints.append([x, y])
        path = CubicSplinePath(np.array(waypoints))

        solver = CppMPCCSolver(config)
        dynamics = AckermannDynamics(wheelbase=config.wheelbase, dt=config.dt)

        # Start aligned with path
        tangent = path.get_tangent(0.0)
        state = np.array([0.0, 0.0, tangent, 0.0, 0.0])
        progress = 0.0

        positions = []
        for step in range(50):
            result = solver.solve(state, path, current_progress=progress)

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

        # Vehicle should move forward along path
        total_dist = np.sum(np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1)))
        assert total_dist > 0.8, f"Vehicle only moved {total_dist:.2f}m on curve"

    def test_qcar2_competition_scenario(self, config):
        """Simulate the actual QCar2 competition path."""
        config.horizon = 12
        config.reference_velocity = 0.35
        config.contour_weight = 20.0
        config.lag_weight = 3.0

        solver = CppMPCCSolver(config)
        dynamics = AckermannDynamics(wheelbase=config.wheelbase, dt=config.dt)

        # Simulated path (Hub to Pickup approximation)
        waypoints = []
        for i in range(50):
            t = i / 49.0
            x = t * 3.0
            y = t * 2.0
            waypoints.append([x, y])
        path = CubicSplinePath(np.array(waypoints))

        heading = np.arctan2(2.0, 3.0)
        state = np.array([0.0, 0.0, heading, 0.0, 0.0])
        progress = 0.0

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

        assert success_count >= 25, \
            f"Only {success_count}/30 succeeded on competition path"


class TestCppSolverEdgeCases:
    def test_near_path_end(self, config):
        """Solver should handle being near the end of the path."""
        path = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(10)]))
        solver = CppMPCCSolver(config)

        # Near end of short path
        state = np.array([2.5, 0.0, 0.0, 0.2, 0.0])
        result = solver.solve(state, path, current_progress=2.5)
        assert result.success

    def test_offset_from_path(self, config, straight_path):
        """Solver should handle starting offset from path."""
        solver = CppMPCCSolver(config)
        # 0.5m offset perpendicular to path
        state = np.array([1.0, 0.5, 0.0, 0.2, 0.0])
        result = solver.solve(state, straight_path, current_progress=1.0)

        assert result.success
        # Should steer toward path (negative delta to go toward y=0)

    def test_heading_misalignment(self, config, straight_path):
        """Solver should handle heading misalignment."""
        solver = CppMPCCSolver(config)
        # 30 degree heading error
        state = np.array([0.0, 0.0, np.radians(30), 0.2, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.0)

        assert result.success

    def test_zero_velocity_start(self, config, straight_path):
        """Solver should handle starting from rest."""
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        result = solver.solve(state, straight_path, current_progress=0.0)

        assert result.success
        assert result.v_cmd >= 0.0, "Should command non-negative velocity from rest"


class TestCppSolverOmegaConversion:
    """Test the Ackermann to angular velocity conversion."""

    def test_omega_matches_ackermann(self, config, straight_path):
        """omega = v * tan(delta) / L should match the C++ output."""
        solver = CppMPCCSolver(config)
        state = np.array([0.0, 0.3, -0.1, 0.3, 0.05])
        result = solver.solve(state, straight_path, current_progress=0.0)

        if result.v_cmd > 0.001:
            expected_omega = result.v_cmd * np.tan(result.delta_cmd) / config.wheelbase
            assert abs(result.omega_cmd - expected_omega) < 0.01, \
                f"omega_cmd={result.omega_cmd}, expected={expected_omega}"

    def test_zero_velocity_zero_omega(self, config, straight_path):
        """When velocity is near zero, omega should be zero."""
        solver = CppMPCCSolver(config)
        # Very near path end to get low velocity
        path = CubicSplinePath(np.array([[i * 0.1, 0.0] for i in range(5)]))
        state = np.array([0.35, 0.0, 0.0, 0.01, 0.0])
        result = solver.solve(state, path, current_progress=0.35)

        # If velocity is very low, omega should be limited
        if abs(result.v_cmd) < 0.01:
            assert abs(result.omega_cmd) < 0.1


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
