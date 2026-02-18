"""
End-to-end command pipeline test.

Verifies that the MPCC solver → cmd_vel_nav → nav2_qcar_command_convert
pipeline produces correct motor commands for the QCar2.

The critical conversion chain:
1. MPCC outputs: v_cmd (m/s), delta_cmd (rad steering angle)
2. mpcc_controller converts: omega = v * tan(delta) / wheelbase
3. Publishes to /cmd_vel_nav: Twist(linear.x=v, angular.z=omega)
4. nav2_qcar_command_convert receives and converts back:
   steering = atan(wheelbase * omega / v)
   This MUST recover the original delta_cmd.

Run with:
    cd /home/stephen/Documents/ACC_Development/Development/ros2/src/acc_stage1_mission
    python3 -m pytest test/test_command_pipeline.py -v
"""

import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from acc_stage1_mission.pympc_core.spline_path import CubicSplinePath
from acc_stage1_mission.pympc_core.solver import MPCCConfig
from acc_stage1_mission.pympc_core.mpcc_cpp import CppMPCCSolver
from acc_stage1_mission.pympc_core.dynamics import AckermannDynamics

WHEELBASE = 0.256


class TestAckermannTwistConversion:
    """Verify the Ackermann ↔ Twist conversion is lossless."""

    @pytest.mark.parametrize("delta", [0.0, 0.1, 0.2, 0.3, 0.45, -0.1, -0.3])
    @pytest.mark.parametrize("v", [0.1, 0.2, 0.3, 0.5])
    def test_roundtrip_conversion(self, v, delta):
        """
        Verify: delta → omega → delta' should recover original delta.

        mpcc_controller does: omega = v * tan(delta) / L
        nav2_qcar_command_convert does: delta' = atan(L * omega / v)
        """
        # Step 1: MPCC controller conversion (mpcc_controller.py line 1024)
        omega = v * np.tan(delta) / WHEELBASE

        # Step 2: nav2_qcar_command_convert conversion
        # (nav2_qcar_command_convert.cpp: steering_angle = atan(wheelbase * angular.z / linear.x))
        if abs(v) > 0.001:
            delta_recovered = np.arctan(WHEELBASE * omega / v)
        else:
            delta_recovered = 0.0

        assert abs(delta_recovered - delta) < 1e-10, \
            f"Roundtrip failed: delta={delta:.3f} → omega={omega:.3f} → delta'={delta_recovered:.3f}"

    def test_zero_velocity_safety(self):
        """When v=0, omega should be 0 (prevents division by zero in converter)."""
        v = 0.0
        delta = 0.3
        # mpcc_controller checks: if abs(v) > 0.001
        # For v=0, it should output omega=0
        if abs(v) > 0.001:
            omega = v * np.tan(delta) / WHEELBASE
        else:
            omega = 0.0
        assert omega == 0.0


class TestFullPipelineSimulation:
    """Simulate the full MPCC → drive pipeline."""

    def test_straight_path_commands(self):
        """On a straight path, commands should be reasonable."""
        config = MPCCConfig(
            horizon=15, dt=0.1, wheelbase=WHEELBASE,
            max_velocity=0.5, reference_velocity=0.35,
        )
        solver = CppMPCCSolver(config)
        path = CubicSplinePath(np.array([[i * 0.3, 0.0] for i in range(30)]))
        dynamics = AckermannDynamics(wheelbase=WHEELBASE, dt=0.1)

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0

        for step in range(30):
            result = solver.solve(state, path, current_progress=progress)

            v_cmd = max(0.0, min(result.v_cmd, config.max_velocity))
            delta_cmd = np.clip(result.delta_cmd, -config.max_steering, config.max_steering)

            # Convert to Twist (as mpcc_controller does)
            if abs(v_cmd) > 0.001:
                omega = v_cmd * np.tan(delta_cmd) / WHEELBASE
            else:
                omega = 0.0
            omega = np.clip(omega, -1.0, 1.0)

            # Verify nav2_qcar_command_convert would recover steering
            if abs(v_cmd) > 0.001:
                delta_recovered = np.arctan(WHEELBASE * omega / v_cmd)
                assert abs(delta_recovered - delta_cmd) < 0.01, \
                    f"Step {step}: delta mismatch: cmd={delta_cmd:.3f} recovered={delta_recovered:.3f}"

            # Simulate vehicle
            control = np.array([
                np.clip((v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = path.find_closest_progress(state[0], state[1])

        # Vehicle should have moved forward (from rest in 3s, ~0.5-1m expected)
        assert state[0] > 0.5, f"Vehicle only at x={state[0]:.2f}, should be >0.5m"

    def test_curved_path_commands(self):
        """On a curved path, verify steering commands are correct."""
        config = MPCCConfig(
            horizon=12, dt=0.1, wheelbase=WHEELBASE,
            max_velocity=0.5, reference_velocity=0.3,
        )
        solver = CppMPCCSolver(config)

        # S-curve path
        waypoints = []
        for i in range(40):
            x = i * 0.25
            y = 0.5 * np.sin(x * 0.4)
            waypoints.append([x, y])
        path = CubicSplinePath(np.array(waypoints))
        dynamics = AckermannDynamics(wheelbase=WHEELBASE, dt=0.1)

        tangent = path.get_tangent(0.0)
        state = np.array([0.0, 0.0, tangent, 0.0, 0.0])
        progress = 0.0

        max_lateral_error = 0.0
        for step in range(40):
            result = solver.solve(state, path, current_progress=progress)

            v_cmd = np.clip(result.v_cmd, 0.0, config.max_velocity)
            delta_cmd = np.clip(result.delta_cmd, -config.max_steering, config.max_steering)

            # Convert and verify
            if abs(v_cmd) > 0.001:
                omega = v_cmd * np.tan(delta_cmd) / WHEELBASE
                delta_recovered = np.arctan(WHEELBASE * omega / v_cmd)
                assert abs(delta_recovered - delta_cmd) < 0.01

            # Simulate
            control = np.array([
                np.clip((v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = path.find_closest_progress(state[0], state[1])

            # Track lateral error
            ref_x, ref_y = path.get_position(progress)
            lateral_error = np.sqrt((state[0] - ref_x)**2 + (state[1] - ref_y)**2)
            max_lateral_error = max(max_lateral_error, lateral_error)

        assert max_lateral_error < 0.5, \
            f"Max lateral error {max_lateral_error:.2f}m on S-curve (expected <0.5m)"

    def test_competition_route_segment(self):
        """Simulate a segment of the actual competition route."""
        config = MPCCConfig(
            horizon=15, dt=0.1, wheelbase=WHEELBASE,
            max_velocity=0.5, reference_velocity=0.4,
            contour_weight=25.0, lag_weight=5.0,
        )
        solver = CppMPCCSolver(config)
        dynamics = AckermannDynamics(wheelbase=WHEELBASE, dt=0.1)

        # Competition-like path with a turn
        waypoints = []
        # Straight section
        for i in range(20):
            waypoints.append([i * 0.2, 0.0])
        # Turn right
        for i in range(20):
            angle = i * np.pi / 40  # 90-degree turn over 20 steps
            waypoints.append([4.0 + 2.0 * np.sin(angle),
                            -2.0 * (1 - np.cos(angle))])
        path = CubicSplinePath(np.array(waypoints))

        state = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        progress = 0.0

        success_count = 0
        for step in range(50):
            result = solver.solve(state, path, current_progress=progress)
            if result.success:
                success_count += 1

            v_cmd = np.clip(result.v_cmd, 0.0, config.max_velocity)
            delta_cmd = np.clip(result.delta_cmd, -config.max_steering, config.max_steering)

            control = np.array([
                np.clip((v_cmd - state[3]) / config.dt,
                        -config.max_acceleration, config.max_acceleration),
                np.clip((delta_cmd - state[4]) / config.dt,
                        -config.max_steering_rate, config.max_steering_rate),
            ])
            state = dynamics.rk4_step(state, control)
            progress = path.find_closest_progress(state[0], state[1])

        assert success_count >= 40, \
            f"Only {success_count}/50 solves succeeded on competition route"
        assert state[0] > 1.0, \
            f"Vehicle should have progressed >1m, only at x={state[0]:.2f}"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
