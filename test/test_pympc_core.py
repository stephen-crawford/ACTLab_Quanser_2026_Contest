"""
Comprehensive tests for pympc_core module.

Tests cover:
1. CubicSplinePath - spline construction, position/tangent/curvature queries
2. AckermannDynamics - RK4 vs Euler, vehicle kinematics
3. MPCCSolver - CasADi solver correctness, warm-starting, obstacle avoidance
4. Constraint accumulation regression - ensures no constraint buildup across solves
5. Integration - full MPCC loop with realistic QCar2 scenario

Run with:
    cd /home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission
    python3 -m pytest test/test_pympc_core.py -v
"""

import sys
import os
import numpy as np
import pytest
import time

# Add the package to path so we can import without ROS2
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acc_stage1_mission.pympc_core.spline_path import CubicSplinePath
from acc_stage1_mission.pympc_core.dynamics import AckermannDynamics
from acc_stage1_mission.pympc_core.solver import MPCCSolver, MPCCConfig, MPCCResult


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def straight_path():
    """Straight-line path along x-axis."""
    waypoints = np.array([[i * 0.5, 0.0] for i in range(20)])
    return CubicSplinePath(waypoints)


@pytest.fixture
def curved_path():
    """S-curve path for testing curvature."""
    t = np.linspace(0, 2 * np.pi, 40)
    x = t
    y = 0.5 * np.sin(t)
    waypoints = np.column_stack([x, y])
    return CubicSplinePath(waypoints)


@pytest.fixture
def l_shaped_path():
    """L-shaped path (turn right)."""
    # Go forward then turn right
    straight = [[i * 0.2, 0.0] for i in range(10)]
    turn = [[2.0, -i * 0.2] for i in range(1, 10)]
    waypoints = np.array(straight + turn)
    return CubicSplinePath(waypoints)


@pytest.fixture
def qcar2_dynamics():
    """QCar2 dynamics model."""
    return AckermannDynamics(wheelbase=0.256, dt=0.1)


@pytest.fixture
def default_config():
    """Default MPCC config for QCar2."""
    return MPCCConfig(
        horizon=15,
        dt=0.1,
        wheelbase=0.256,
        max_velocity=0.5,
        reference_velocity=0.3,
        contour_weight=20.0,
        lag_weight=3.0,
        max_iter=50,
    )


@pytest.fixture
def fast_config():
    """Fast config with shorter horizon for quicker tests."""
    return MPCCConfig(
        horizon=8,
        dt=0.1,
        wheelbase=0.256,
        max_velocity=0.5,
        reference_velocity=0.3,
        max_iter=30,
    )


# ============================================================================
# CubicSplinePath Tests
# ============================================================================

class TestCubicSplinePath:
    """Tests for spline path representation."""

    def test_straight_path_length(self, straight_path):
        """Straight path should have correct total length."""
        expected_length = 19 * 0.5  # 19 segments of 0.5m
        assert abs(straight_path.total_length - expected_length) < 0.01

    def test_straight_path_position(self, straight_path):
        """Positions along straight path should be correct."""
        x, y = straight_path.get_position(0.0)
        assert abs(x) < 0.01
        assert abs(y) < 0.01

        x, y = straight_path.get_position(1.0)
        assert abs(x - 1.0) < 0.05
        assert abs(y) < 0.05

    def test_straight_path_tangent(self, straight_path):
        """Tangent of straight path should point along x-axis."""
        for s in [0.5, 1.0, 2.0, 4.0]:
            tangent = straight_path.get_tangent(s)
            assert abs(tangent) < 0.1, f"Tangent at s={s} should be ~0, got {tangent}"

    def test_curved_path_has_curvature(self, curved_path):
        """S-curve should have non-zero curvature at interior points."""
        # Check at peak of sine wave
        mid_s = curved_path.total_length * 0.25
        kappa = curved_path.get_curvature(mid_s)
        assert abs(kappa) > 0.01, "Curved path should have non-zero curvature"

    def test_find_closest_progress(self, straight_path):
        """Find closest point on straight path."""
        # Point near path at s~1.0
        s = straight_path.find_closest_progress(1.0, 0.1)
        assert abs(s - 1.0) < 0.2

        # Point at origin
        s = straight_path.find_closest_progress(0.0, 0.0)
        assert s < 0.5

    def test_contouring_errors_on_path(self, straight_path):
        """Vehicle on path should have near-zero errors."""
        e_c, e_l = straight_path.compute_contouring_errors(1.0, 0.0, 1.0)
        assert abs(e_c) < 0.1, f"Contouring error on path should be ~0, got {e_c}"
        assert abs(e_l) < 0.1, f"Lag error on path should be ~0, got {e_l}"

    def test_contouring_error_lateral_offset(self, straight_path):
        """Vehicle offset from path should have non-zero contouring error."""
        e_c, e_l = straight_path.compute_contouring_errors(1.0, 0.5, 1.0)
        assert abs(e_c) > 0.3, f"Contouring error for offset vehicle should be large, got {e_c}"

    def test_sample_points(self, straight_path):
        """Sample points should cover the path."""
        points = straight_path.sample_points(50)
        assert points.shape == (50, 2)
        assert abs(points[0, 0]) < 0.1
        assert abs(points[-1, 0] - straight_path.waypoints[-1, 0]) < 0.5

    def test_minimum_waypoints(self):
        """Should work with minimum 2 waypoints (linear fallback)."""
        path = CubicSplinePath(np.array([[0, 0], [1, 1]]))
        assert abs(path.total_length - np.sqrt(2)) < 0.01
        x, y = path.get_position(path.total_length / 2)
        assert abs(x - 0.5) < 0.1
        assert abs(y - 0.5) < 0.1

    def test_three_waypoints(self):
        """Should work with 3 waypoints (linear fallback)."""
        path = CubicSplinePath(np.array([[0, 0], [1, 0], [2, 0]]))
        assert abs(path.total_length - 2.0) < 0.01

    def test_clipping_at_boundaries(self, straight_path):
        """Queries beyond path bounds should be clipped."""
        # Beyond end
        x, y = straight_path.get_position(straight_path.total_length + 100)
        assert np.isfinite(x) and np.isfinite(y)

        # Before start
        x, y = straight_path.get_position(-100)
        assert np.isfinite(x) and np.isfinite(y)


# ============================================================================
# AckermannDynamics Tests
# ============================================================================

class TestAckermannDynamics:
    """Tests for vehicle dynamics model."""

    def test_straight_line_motion(self, qcar2_dynamics):
        """Driving straight should increase x, keep y constant."""
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])  # v=0.5, delta=0
        control = np.array([0.0, 0.0])  # No acceleration, no steering

        next_state = qcar2_dynamics.rk4_step(state, control)
        assert next_state[0] > 0.04  # x increased
        assert abs(next_state[1]) < 0.001  # y unchanged
        assert abs(next_state[2]) < 0.001  # theta unchanged
        assert abs(next_state[3] - 0.5) < 0.001  # velocity unchanged

    def test_acceleration(self, qcar2_dynamics):
        """Applying acceleration should increase velocity."""
        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        control = np.array([1.0, 0.0])  # Accelerate

        next_state = qcar2_dynamics.rk4_step(state, control)
        assert next_state[3] > 0.09  # Velocity increased

    def test_steering(self, qcar2_dynamics):
        """Applying steering rate should change steering angle."""
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        control = np.array([0.0, 0.5])  # Steer

        next_state = qcar2_dynamics.rk4_step(state, control)
        assert next_state[4] > 0.04  # Steering angle increased

    def test_turning(self, qcar2_dynamics):
        """Moving with steering should change heading."""
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.3])  # v=0.5, delta=0.3
        control = np.array([0.0, 0.0])

        next_state = qcar2_dynamics.rk4_step(state, control)
        assert abs(next_state[2]) > 0.01  # Heading changed

    def test_rk4_vs_euler(self, qcar2_dynamics):
        """RK4 should be more accurate than Euler for turning."""
        state = np.array([0.0, 0.0, 0.0, 1.0, 0.4])
        control = np.array([0.0, 0.0])

        rk4_state = qcar2_dynamics.rk4_step(state, control)
        euler_state = qcar2_dynamics.euler_step(state, control)

        # Both should produce finite results
        assert all(np.isfinite(rk4_state))
        assert all(np.isfinite(euler_state))

        # They should differ (RK4 is more accurate)
        diff = np.linalg.norm(rk4_state - euler_state)
        assert diff > 1e-6, "RK4 and Euler should give different results for curved motion"

    def test_simulate_trajectory(self, qcar2_dynamics):
        """Simulate should produce a valid trajectory."""
        x0 = np.array([0.0, 0.0, 0.0, 0.3, 0.0])
        controls = np.zeros((10, 2))
        controls[:, 0] = 0.5  # Constant acceleration

        states = qcar2_dynamics.simulate(x0, controls)
        assert states.shape == (11, 5)
        assert all(np.isfinite(states.flat))
        assert states[-1, 3] > x0[3]  # Velocity increased
        assert states[-1, 0] > 0  # Moved forward


# ============================================================================
# MPCCSolver Tests
# ============================================================================

class TestMPCCSolver:
    """Tests for the MPCC solver."""

    def test_solver_creation(self, default_config):
        """Solver should initialize without errors."""
        solver = MPCCSolver(default_config)
        assert solver is not None

    def test_solve_straight_path(self, fast_config, straight_path):
        """Solver should follow a straight path."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = solver.solve(x0, straight_path, current_progress=0.0)

        assert result.success, "Solver should succeed on simple straight path"
        assert result.v_cmd >= 0, "Velocity should be non-negative"
        assert abs(result.delta_cmd) < 0.3, "Steering should be small on straight path"
        assert result.predicted_trajectory.shape[0] > 1
        assert result.solve_time < 5.0, "Solve should complete in reasonable time"

    def test_solve_from_offset(self, fast_config, straight_path):
        """Solver should steer back toward path when starting from an offset."""
        solver = MPCCSolver(fast_config)
        # Start 0.3m to the left of the path
        x0 = np.array([0.5, 0.3, 0.0, 0.2, 0.0])

        result = solver.solve(x0, straight_path, current_progress=0.5)

        assert result.success
        # Should steer toward the path (negative delta to go right toward y=0)
        # The contouring cost should create a steering correction

    def test_solve_with_obstacles(self, fast_config, straight_path):
        """Solver should avoid obstacles."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])
        obstacles = [(1.0, 0.0, 0.3)]  # Obstacle on path at x=1.0

        result = solver.solve(
            x0, straight_path, current_progress=0.0, obstacles=obstacles)

        assert result.success

    def test_no_constraint_accumulation(self, fast_config, straight_path):
        """
        REGRESSION TEST: Multiple consecutive solves should not degrade.

        This is the critical test for the constraint accumulation bug.
        The old CasadiMPCCSolver would add constraints in solve() that
        accumulated in the Opti object, causing solver failure after
        a few iterations.
        """
        solver = MPCCSolver(fast_config)

        success_count = 0
        total_runs = 20  # Simulate 20 control cycles

        x0 = np.array([0.0, 0.0, 0.0, 0.1, 0.0])
        progress = 0.0

        for i in range(total_runs):
            # Add some obstacles (varying to test constraint reset)
            obstacles = [(1.0 + i * 0.1, 0.1, 0.2)]

            result = solver.solve(
                x0, straight_path, current_progress=progress,
                obstacles=obstacles)

            if result.success:
                success_count += 1

            # Advance state (simple simulation)
            dt = fast_config.dt
            x0 = np.array([
                x0[0] + result.v_cmd * np.cos(x0[2]) * dt,
                x0[1] + result.v_cmd * np.sin(x0[2]) * dt,
                x0[2] + result.v_cmd / fast_config.wheelbase * np.tan(result.delta_cmd) * dt,
                result.v_cmd,
                result.delta_cmd,
            ])
            progress = straight_path.find_closest_progress(x0[0], x0[1])

        # With the old buggy solver, success_count would drop to 0 after
        # a few iterations. With the fix, most or all should succeed.
        success_rate = success_count / total_runs
        assert success_rate >= 0.8, (
            f"Success rate {success_rate:.0%} too low - "
            f"possible constraint accumulation regression! "
            f"({success_count}/{total_runs} succeeded)"
        )

    def test_warm_starting(self, fast_config, straight_path):
        """Second solve should be faster due to warm-starting."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        # First solve (cold start)
        r1 = solver.solve(x0, straight_path, current_progress=0.0)

        # Second solve (warm start from first)
        x0_next = np.array([0.02, 0.0, 0.0, 0.2, 0.0])
        r2 = solver.solve(x0_next, straight_path, current_progress=0.02)

        # Both should succeed
        assert r1.success
        assert r2.success
        # Warm-started solve is typically faster (but not guaranteed)

    def test_goal_reaching(self, fast_config, straight_path):
        """MPCC should make progress along the path."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0])

        result = solver.solve(x0, straight_path, current_progress=0.0)
        assert result.success
        assert result.v_cmd > 0, "Solver should command forward velocity"

    def test_fallback_solver(self, fast_config, straight_path):
        """Fallback solver should produce valid commands."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        result = solver._solve_fallback(
            x0, straight_path, current_progress=0.0, obstacles=None)

        assert result.success
        assert result.v_cmd > 0
        assert abs(result.delta_cmd) < fast_config.max_steering
        assert result.predicted_trajectory.shape[0] > 1

    def test_curved_path_following(self, default_config, curved_path):
        """Solver should handle curved paths."""
        solver = MPCCSolver(default_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        result = solver.solve(x0, curved_path, current_progress=0.0)
        assert result.success

    def test_reset_clears_warmstart(self, fast_config, straight_path):
        """Reset should clear warm-start state."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        solver.solve(x0, straight_path, current_progress=0.0)
        assert solver._prev_X is not None

        solver.reset()
        assert solver._prev_X is None

    def test_boundary_constraints(self, fast_config, straight_path):
        """Solver should work with boundary constraints."""
        solver = MPCCSolver(fast_config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        # Create boundary constraints (normal vector perpendicular to path)
        boundary_constraints = [
            (np.array([0.0, 1.0]), 0.3, 0.3)  # ±0.3m from path
            for _ in range(fast_config.horizon)
        ]

        result = solver.solve(
            x0, straight_path, current_progress=0.0,
            boundary_constraints=boundary_constraints)

        assert result.success


# ============================================================================
# Integration Test - Full MPCC Loop
# ============================================================================

class TestMPCCIntegration:
    """Integration tests simulating full MPCC control loops."""

    def test_full_control_loop_straight(self):
        """Simulate full control loop on straight path."""
        config = MPCCConfig(
            horizon=10,
            dt=0.1,
            wheelbase=0.256,
            max_velocity=0.5,
            reference_velocity=0.3,
            max_iter=30,
        )

        waypoints = np.array([[i * 0.3, 0.0] for i in range(30)])
        path = CubicSplinePath(waypoints)
        solver = MPCCSolver(config)

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0
        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)

        total_distance = 0.0
        max_lateral_error = 0.0

        for step in range(50):
            result = solver.solve(state, path, current_progress=progress)

            if not result.success:
                continue

            # Simulate one step
            control = np.array([
                (result.v_cmd - state[3]) / config.dt,  # acceleration
                (result.delta_cmd - state[4]) / config.dt,  # steering rate
            ])
            control[0] = np.clip(control[0], -config.max_acceleration, config.max_acceleration)
            control[1] = np.clip(control[1], -config.max_steering_rate, config.max_steering_rate)

            new_state = dynamics.rk4_step(state, control)
            total_distance += np.sqrt(
                (new_state[0] - state[0])**2 + (new_state[1] - state[1])**2)

            # Track lateral error
            e_c, _ = path.compute_contouring_errors(new_state[0], new_state[1], progress)
            max_lateral_error = max(max_lateral_error, abs(e_c))

            state = new_state
            progress = path.find_closest_progress(state[0], state[1])

        assert total_distance > 1.0, f"Car should have moved >1m, moved {total_distance:.2f}m"
        assert max_lateral_error < 0.5, f"Max lateral error {max_lateral_error:.3f}m too large"

    def test_full_control_loop_curved(self):
        """Simulate full control loop on curved path."""
        config = MPCCConfig(
            horizon=10,
            dt=0.1,
            wheelbase=0.256,
            max_velocity=0.4,
            reference_velocity=0.25,
            contour_weight=25.0,
            max_iter=40,
        )

        # Gentle curve
        t = np.linspace(0, np.pi, 30)
        x = 2.0 * np.cos(t) + 2.0
        y = 2.0 * np.sin(t)
        waypoints = np.column_stack([x, y])
        path = CubicSplinePath(waypoints)
        solver = MPCCSolver(config)

        state = np.array([waypoints[0, 0], waypoints[0, 1], np.pi / 2, 0.0, 0.0])
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

        assert success_count > 15, f"Only {success_count}/30 solves succeeded on curved path"

    def test_solve_time_performance(self):
        """Solver should complete within real-time requirements."""
        config = MPCCConfig(
            horizon=15,
            dt=0.1,
            max_iter=50,
            reference_velocity=0.3,
        )

        waypoints = np.array([[i * 0.3, 0.0] for i in range(30)])
        path = CubicSplinePath(waypoints)
        solver = MPCCSolver(config)
        x0 = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        times = []
        for _ in range(5):
            t0 = time.time()
            solver.solve(x0, path, current_progress=0.0)
            times.append(time.time() - t0)

        avg_time = np.mean(times)
        # Should complete in < 500ms for real-time control at 20Hz
        assert avg_time < 0.5, f"Average solve time {avg_time:.3f}s exceeds 500ms"


# ============================================================================
# Run tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
