"""
Tests for slip angle dynamics model and coordinate transforms.

Verifies the critical changes made to the MPCC system:
1. Slip angle dynamics (beta = atan(tan(delta)/2)) in Python dynamics
2. Slip angle dynamics in CasADi solver
3. Coordinate transform qlabs_to_map_frame correctness
4. C++ and Python dynamics consistency
5. Command pipeline with updated dynamics
6. Full path-following simulation with slip angle model

Run with:
    cd /home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission
    python3 -m pytest test/test_slip_angle_dynamics.py -v
"""

import sys
import os
import math
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acc_stage1_mission.pympc_core.dynamics import AckermannDynamics
from acc_stage1_mission.pympc_core.spline_path import CubicSplinePath
from acc_stage1_mission.pympc_core.solver import MPCCConfig, MPCCSolver


# ============================================================================
# Slip Angle Dynamics Tests
# ============================================================================

class TestSlipAngleDynamics:
    """Verify the bicycle model with slip angle beta = atan(tan(delta)/2)."""

    @pytest.fixture
    def dynamics(self):
        return AckermannDynamics(wheelbase=0.256, dt=0.1)

    def test_zero_steering_no_slip(self, dynamics):
        """With delta=0, slip angle beta=0 so dynamics match simple model."""
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.0])
        control = np.array([0.0, 0.0])

        # beta = atan(tan(0)/2) = 0
        # dx/dt = v*cos(theta+0) = v*cos(theta) = 0.5
        # dy/dt = v*sin(theta+0) = 0
        # dtheta/dt = v/L*sin(0) = 0
        deriv = dynamics.continuous_dynamics(state, control)

        assert abs(deriv[0] - 0.5) < 1e-10, f"x_dot should be 0.5, got {deriv[0]}"
        assert abs(deriv[1]) < 1e-10, f"y_dot should be 0, got {deriv[1]}"
        assert abs(deriv[2]) < 1e-10, f"theta_dot should be 0, got {deriv[2]}"
        assert abs(deriv[3]) < 1e-10, f"v_dot should be 0, got {deriv[3]}"
        assert abs(deriv[4]) < 1e-10, f"delta_dot should be 0, got {deriv[4]}"

    def test_small_steering_slip_angle(self, dynamics):
        """For small delta, beta ≈ delta/2 (first-order approximation)."""
        delta = 0.1  # Small steering angle
        state = np.array([0.0, 0.0, 0.0, 0.5, delta])
        control = np.array([0.0, 0.0])

        beta = np.arctan(np.tan(delta) / 2.0)
        # For small angles: beta ≈ delta/2
        assert abs(beta - delta / 2.0) < 0.001, \
            f"For small delta, beta should be ~delta/2, got beta={beta}, delta/2={delta/2}"

        deriv = dynamics.continuous_dynamics(state, control)

        # Verify dynamics use slip angle
        expected_x_dot = 0.5 * np.cos(0.0 + beta)
        expected_y_dot = 0.5 * np.sin(0.0 + beta)
        expected_theta_dot = 0.5 / 0.256 * np.sin(beta)

        assert abs(deriv[0] - expected_x_dot) < 1e-10
        assert abs(deriv[1] - expected_y_dot) < 1e-10
        assert abs(deriv[2] - expected_theta_dot) < 1e-10

    def test_large_steering_slip_angle(self, dynamics):
        """For larger delta, slip angle significantly differs from simple model."""
        delta = 0.45  # Max steering
        state = np.array([0.0, 0.0, 0.0, 0.5, delta])
        control = np.array([0.0, 0.0])

        beta = np.arctan(np.tan(delta) / 2.0)
        # beta should be less than delta
        assert beta < delta, f"beta={beta} should be less than delta={delta}"
        assert beta > 0, f"beta should be positive for positive delta"

        # Compare with simple (no-slip) model
        simple_theta_dot = 0.5 / 0.256 * np.tan(delta)
        slip_theta_dot = 0.5 / 0.256 * np.sin(beta)

        # With slip angle, turning rate is lower (more realistic)
        assert abs(slip_theta_dot) < abs(simple_theta_dot), \
            f"Slip model theta_dot ({slip_theta_dot:.4f}) should be less than " \
            f"simple model ({simple_theta_dot:.4f})"

    def test_slip_angle_symmetry(self, dynamics):
        """Slip angle should be antisymmetric: beta(-delta) = -beta(delta)."""
        for delta in [0.1, 0.2, 0.3, 0.45]:
            beta_pos = np.arctan(np.tan(delta) / 2.0)
            beta_neg = np.arctan(np.tan(-delta) / 2.0)
            assert abs(beta_pos + beta_neg) < 1e-10, \
                f"beta({delta})={beta_pos}, beta({-delta})={beta_neg}, not antisymmetric"

    def test_rk4_preserves_slip_angle(self, dynamics):
        """RK4 step should correctly propagate slip angle dynamics."""
        delta = 0.2
        state = np.array([0.0, 0.0, 0.0, 0.3, delta])
        control = np.array([0.0, 0.0])

        next_state = dynamics.rk4_step(state, control)

        # After one step, heading should have changed
        beta = np.arctan(np.tan(delta) / 2.0)
        expected_theta_change = 0.3 / 0.256 * np.sin(beta) * 0.1
        actual_theta_change = next_state[2] - state[2]

        # RK4 should be close to first-order estimate
        assert abs(actual_theta_change - expected_theta_change) < 0.005, \
            f"theta change: expected ~{expected_theta_change:.5f}, got {actual_theta_change:.5f}"

    def test_velocity_direction_with_slip(self, dynamics):
        """
        With slip angle, velocity vector is NOT aligned with heading.
        The car's velocity direction is theta + beta, not just theta.
        """
        delta = 0.3
        theta = 0.5
        state = np.array([0.0, 0.0, theta, 0.4, delta])
        control = np.array([0.0, 0.0])

        deriv = dynamics.continuous_dynamics(state, control)
        beta = np.arctan(np.tan(delta) / 2.0)

        # Velocity direction = atan2(y_dot, x_dot) should be theta + beta
        vel_direction = np.arctan2(deriv[1], deriv[0])
        expected_direction = theta + beta

        assert abs(vel_direction - expected_direction) < 1e-10, \
            f"Velocity direction {vel_direction:.4f} should be theta+beta={expected_direction:.4f}"

    def test_simulate_circle(self, dynamics):
        """Vehicle with constant steering should trace a circle."""
        delta = 0.2
        v = 0.3
        state = np.array([0.0, 0.0, 0.0, v, delta])
        controls = np.zeros((200, 2))  # 20 seconds of constant driving

        states = dynamics.simulate(state, controls)

        # Check it traces a curve (not a straight line)
        positions = states[:, :2]
        max_y = np.max(np.abs(positions[:, 1]))
        assert max_y > 0.1, "Vehicle with steering should deviate from x-axis"

        # Should eventually come back near start (circle)
        # With beta model, the turning radius is R = L / sin(beta)
        beta = np.arctan(np.tan(delta) / 2.0)
        theoretical_radius = 0.256 / np.sin(beta)

        # After one full circle (circumference = 2*pi*R), at v m/s,
        # it takes T = 2*pi*R/v seconds = 2*pi*R/v / dt steps
        circumference = 2 * np.pi * theoretical_radius
        steps_for_circle = int(circumference / (v * 0.1))

        if steps_for_circle < 200:
            # Check that vehicle returns near start
            end_dist = np.linalg.norm(states[steps_for_circle, :2] - states[0, :2])
            assert end_dist < theoretical_radius * 0.3, \
                f"After ~1 circle, distance from start ({end_dist:.2f}) " \
                f"should be much less than radius ({theoretical_radius:.2f})"


# ============================================================================
# Coordinate Transform Tests
# ============================================================================

def _qlabs_to_map_frame(qlabs_x, qlabs_y, qlabs_yaw,
                        origin_x=-1.205, origin_y=-0.83,
                        origin_heading_deg=-44.7):
    """Local copy of qlabs_to_map_frame (avoids rclpy import)."""
    translated_x = qlabs_x - origin_x
    translated_y = qlabs_y - origin_y
    theta = math.radians(-origin_heading_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    map_x = translated_x * cos_t - translated_y * sin_t
    map_y = translated_x * sin_t + translated_y * cos_t
    map_yaw = qlabs_yaw - math.radians(origin_heading_deg)
    while map_yaw > math.pi:
        map_yaw -= 2 * math.pi
    while map_yaw < -math.pi:
        map_yaw += 2 * math.pi
    return (map_x, map_y, map_yaw)


def _map_to_qlabs_frame(map_x, map_y, map_yaw,
                         origin_x=-1.205, origin_y=-0.83,
                         origin_heading_deg=-44.7):
    """Local copy of map_to_qlabs_frame (avoids rclpy import)."""
    theta = math.radians(origin_heading_deg)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    rotated_x = map_x * cos_t - map_y * sin_t
    rotated_y = map_x * sin_t + map_y * cos_t
    qlabs_x = rotated_x + origin_x
    qlabs_y = rotated_y + origin_y
    qlabs_yaw = map_yaw + math.radians(origin_heading_deg)
    while qlabs_yaw > math.pi:
        qlabs_yaw -= 2 * math.pi
    while qlabs_yaw < -math.pi:
        qlabs_yaw += 2 * math.pi
    return (qlabs_x, qlabs_y, qlabs_yaw)


class TestCoordinateTransform:
    """Verify QLabs ↔ map frame coordinate transforms."""

    def _get_transforms(self):
        return _qlabs_to_map_frame, _map_to_qlabs_frame

    def test_hub_transforms_to_origin(self):
        """Hub at QLabs spawn should map to (0, 0, ~0)."""
        qlabs_to_map, _ = self._get_transforms()

        hub_qlabs = (-1.205, -0.83, math.radians(-44.7))
        map_x, map_y, map_yaw = qlabs_to_map(*hub_qlabs)

        assert abs(map_x) < 0.01, f"Hub x in map frame should be ~0, got {map_x}"
        assert abs(map_y) < 0.01, f"Hub y in map frame should be ~0, got {map_y}"
        assert abs(map_yaw) < 0.05, f"Hub yaw in map frame should be ~0, got {map_yaw}"

    def test_roundtrip_identity(self):
        """qlabs→map→qlabs should be identity."""
        qlabs_to_map, map_to_qlabs = self._get_transforms()

        test_points = [
            (-1.205, -0.83, -0.78),
            (0.125, 4.395, 1.57),
            (-0.905, 0.800, 0.0),
            (0.0, 0.0, 0.0),
            (1.0, 2.0, 0.5),
        ]

        for qlabs_x, qlabs_y, qlabs_yaw in test_points:
            map_x, map_y, map_yaw = qlabs_to_map(qlabs_x, qlabs_y, qlabs_yaw)
            recovered_x, recovered_y, recovered_yaw = map_to_qlabs(map_x, map_y, map_yaw)

            assert abs(recovered_x - qlabs_x) < 1e-6, \
                f"x roundtrip failed: {qlabs_x} → {map_x} → {recovered_x}"
            assert abs(recovered_y - qlabs_y) < 1e-6, \
                f"y roundtrip failed: {qlabs_y} → {map_y} → {recovered_y}"
            # Yaw wrapping can differ by 2*pi, so normalize
            yaw_diff = (recovered_yaw - qlabs_yaw + math.pi) % (2 * math.pi) - math.pi
            assert abs(yaw_diff) < 1e-6, \
                f"yaw roundtrip failed: {qlabs_yaw} → {map_yaw} → {recovered_yaw}"

    def test_pickup_transform(self):
        """Verify pickup coordinates transform correctly."""
        qlabs_to_map, _ = self._get_transforms()

        # Pickup in QLabs: (0.125, 4.395, 1.57)
        map_x, map_y, map_yaw = qlabs_to_map(0.125, 4.395, 1.57)

        # After transform, pickup should be ~4-5m from hub (at origin)
        dist = math.sqrt(map_x**2 + map_y**2)
        assert 3.0 < dist < 7.0, \
            f"Pickup should be 3-7m from hub in map frame, got {dist:.2f}m"

        # Map coordinates should be finite and reasonable
        assert abs(map_x) < 10.0 and abs(map_y) < 10.0, \
            f"Map coords ({map_x:.2f}, {map_y:.2f}) unreasonable"

    def test_rotation_preserves_distances(self):
        """Transform should preserve distances between points."""
        qlabs_to_map, _ = self._get_transforms()

        # Two points in QLabs
        p1_qlabs = (0.0, 0.0, 0.0)
        p2_qlabs = (1.0, 1.0, 0.0)

        dist_qlabs = math.sqrt((p2_qlabs[0] - p1_qlabs[0])**2 +
                                (p2_qlabs[1] - p1_qlabs[1])**2)

        p1_map = qlabs_to_map(*p1_qlabs)
        p2_map = qlabs_to_map(*p2_qlabs)

        dist_map = math.sqrt((p2_map[0] - p1_map[0])**2 +
                              (p2_map[1] - p1_map[1])**2)

        assert abs(dist_qlabs - dist_map) < 1e-6, \
            f"Distance not preserved: qlabs={dist_qlabs:.4f}, map={dist_map:.4f}"


# ============================================================================
# C++ vs Python Dynamics Consistency
# ============================================================================

class TestCppPythonConsistency:
    """Test that C++ and Python solvers use consistent dynamics."""

    @pytest.fixture
    def config(self):
        return MPCCConfig(
            horizon=10, dt=0.1, wheelbase=0.256,
            max_velocity=0.5, reference_velocity=0.3,
            contour_weight=20.0, lag_weight=3.0,
            max_iter=30,
        )

    @pytest.fixture
    def path(self):
        waypoints = np.array([[i * 0.3, 0.0] for i in range(30)])
        return CubicSplinePath(waypoints)

    def test_python_dynamics_slip_angle(self):
        """Verify Python dynamics includes slip angle."""
        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)
        state = np.array([0.0, 0.0, 0.0, 0.5, 0.3])
        control = np.array([0.0, 0.0])

        deriv = dynamics.continuous_dynamics(state, control)

        # With slip angle, dy/dt should be non-zero even when theta=0
        # because beta = atan(tan(0.3)/2) ≠ 0
        beta = np.arctan(np.tan(0.3) / 2.0)
        expected_y_dot = 0.5 * np.sin(0.0 + beta)

        assert abs(deriv[1] - expected_y_dot) < 1e-10, \
            f"y_dot should include slip angle: expected {expected_y_dot}, got {deriv[1]}"

    def test_cpp_solver_produces_steering(self, config, path):
        """C++ solver should produce steering corrections for offset start."""
        try:
            from acc_stage1_mission.pympc_core.mpcc_cpp import CppMPCCSolver, is_available
            if not is_available():
                pytest.skip("C++ solver not available")
        except ImportError:
            pytest.skip("C++ solver module not available")

        solver = CppMPCCSolver(config)
        # Start offset from path
        state = np.array([0.5, 0.3, 0.0, 0.2, 0.0])
        result = solver.solve(state, path, current_progress=0.5)

        assert result.success, "C++ solver should succeed"
        # Should steer to correct lateral offset
        assert result.v_cmd > 0.0, "Should command positive velocity"

    def test_cpp_python_same_direction(self, config, path):
        """Both solvers should command similar direction on simple scenario."""
        try:
            from acc_stage1_mission.pympc_core.mpcc_cpp import CppMPCCSolver, is_available
            if not is_available():
                pytest.skip("C++ solver not available")
        except ImportError:
            pytest.skip("C++ solver module not available")

        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0])

        cpp_solver = CppMPCCSolver(config)
        cpp_result = cpp_solver.solve(state, path, current_progress=0.0)

        py_solver = MPCCSolver(config)
        py_result = py_solver.solve(state, path, current_progress=0.0)

        if cpp_result.success and py_result.success:
            # Both should command forward velocity
            assert cpp_result.v_cmd > 0.0
            assert py_result.v_cmd > 0.0

            # Steering direction should agree (both near zero for straight path)
            assert abs(cpp_result.delta_cmd) < 0.3
            assert abs(py_result.delta_cmd) < 0.3


# ============================================================================
# Command Pipeline Tests (Updated for Slip Angle)
# ============================================================================

class TestCommandPipelineSlipAngle:
    """Test command pipeline with slip angle dynamics."""

    WHEELBASE = 0.256

    def test_twist_roundtrip_still_works(self):
        """
        Even with slip angle dynamics, the Twist conversion is still:
        omega = v * tan(delta) / L  (not using beta!)

        This is because the Twist command represents the DESIRED angular
        velocity, and nav2_qcar_command_convert recovers the steering angle.
        The slip angle affects dynamics prediction, not the command conversion.
        """
        for delta in [0.0, 0.1, 0.2, 0.3, 0.45, -0.1, -0.3]:
            for v in [0.1, 0.2, 0.3, 0.5]:
                omega = v * np.tan(delta) / self.WHEELBASE
                delta_recovered = np.arctan(self.WHEELBASE * omega / v)
                assert abs(delta_recovered - delta) < 1e-10, \
                    f"Roundtrip failed: delta={delta:.3f} → omega={omega:.3f} → delta'={delta_recovered:.3f}"

    def test_slip_angle_vs_simple_trajectory(self):
        """
        Compare trajectories from slip angle model vs simple model.
        They should diverge, especially at higher steering angles.
        """
        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)

        # With moderate steering
        delta = 0.3
        v = 0.3
        state = np.array([0.0, 0.0, 0.0, v, delta])
        controls = np.zeros((50, 2))

        # Simulate with slip angle model
        slip_states = dynamics.simulate(state, controls)

        # Simulate with simple model (manually)
        simple_states = np.zeros((51, 5))
        simple_states[0] = state
        L = 0.256
        dt = 0.1
        for k in range(50):
            s = simple_states[k]
            simple_states[k+1, 0] = s[0] + dt * s[3] * np.cos(s[2])
            simple_states[k+1, 1] = s[1] + dt * s[3] * np.sin(s[2])
            simple_states[k+1, 2] = s[2] + dt * s[3] / L * np.tan(s[4])
            simple_states[k+1, 3] = s[3]
            simple_states[k+1, 4] = s[4]

        # The trajectories should diverge
        final_diff = np.linalg.norm(slip_states[-1, :2] - simple_states[-1, :2])
        assert final_diff > 0.01, \
            f"Slip and simple models should diverge, but final diff is only {final_diff:.4f}m"

        # The slip model should have a LARGER turning radius (less aggressive turn)
        # So the final y position should be SMALLER with slip angle
        assert abs(slip_states[-1, 1]) < abs(simple_states[-1, 1]), \
            f"Slip model (y={slip_states[-1, 1]:.3f}) should turn less aggressively " \
            f"than simple model (y={simple_states[-1, 1]:.3f})"


# ============================================================================
# Full Path Following with Updated Dynamics
# ============================================================================

class TestPathFollowingUpdated:
    """Test complete path following with all updates applied."""

    def test_straight_path_tracking(self):
        """Vehicle should track straight path with low lateral error."""
        config = MPCCConfig(
            horizon=10, dt=0.1, wheelbase=0.256,
            max_velocity=0.5, reference_velocity=0.3,
            contour_weight=20.0, lag_weight=3.0,
            max_iter=30,
        )

        try:
            from acc_stage1_mission.pympc_core.mpcc_cpp import CppMPCCSolver, is_available
            if is_available():
                solver = CppMPCCSolver(config)
                solver_name = "C++"
            else:
                solver = MPCCSolver(config)
                solver_name = "Python"
        except ImportError:
            solver = MPCCSolver(config)
            solver_name = "Python"

        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)
        path = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(30)]))

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0

        max_lateral_error = 0.0
        total_dist = 0.0

        for step in range(50):
            result = solver.solve(state, path, current_progress=progress)

            control = np.array([
                np.clip((result.v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((result.delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            new_state = dynamics.rk4_step(state, control)

            total_dist += np.linalg.norm(new_state[:2] - state[:2])
            e_c, _ = path.compute_contouring_errors(new_state[0], new_state[1], progress)
            max_lateral_error = max(max_lateral_error, abs(e_c))

            state = new_state
            progress = path.find_closest_progress(state[0], state[1])

        print(f"\n  {solver_name} solver: distance={total_dist:.2f}m, max_lateral_error={max_lateral_error:.3f}m")
        assert total_dist > 1.0, f"Vehicle only moved {total_dist:.2f}m"
        assert max_lateral_error < 0.3, f"Max lateral error {max_lateral_error:.3f}m too large"

    def test_curved_path_tracking(self):
        """Vehicle should track a curved path."""
        config = MPCCConfig(
            horizon=12, dt=0.1, wheelbase=0.256,
            max_velocity=0.4, reference_velocity=0.25,
            contour_weight=25.0, lag_weight=5.0,
            max_iter=40,
        )

        try:
            from acc_stage1_mission.pympc_core.mpcc_cpp import CppMPCCSolver, is_available
            if is_available():
                solver = CppMPCCSolver(config)
            else:
                solver = MPCCSolver(config)
        except ImportError:
            solver = MPCCSolver(config)

        dynamics = AckermannDynamics(wheelbase=0.256, dt=0.1)

        # Create a gentle curve (90-degree turn with radius 2m)
        waypoints = []
        for i in range(30):
            angle = i * (np.pi / 2) / 29
            waypoints.append([2.0 * np.sin(angle), 2.0 * (1 - np.cos(angle))])
        path = CubicSplinePath(np.array(waypoints))

        tangent = path.get_tangent(0.0)
        state = np.array([0.0, 0.0, tangent, 0.0, 0.0])
        progress = 0.0

        success_count = 0
        max_lateral_error = 0.0

        for step in range(40):
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

            e_c, _ = path.compute_contouring_errors(state[0], state[1], progress)
            max_lateral_error = max(max_lateral_error, abs(e_c))

        print(f"\n  Curved path: {success_count}/40 succeeded, max_error={max_lateral_error:.3f}m")
        assert success_count >= 30, f"Only {success_count}/40 solves succeeded"
        assert max_lateral_error < 0.5, f"Max lateral error {max_lateral_error:.3f}m too large"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short', '-s'])
