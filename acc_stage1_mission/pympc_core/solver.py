"""
MPCC Solver - Model Predictive Contouring Control using CasADi.

Adapted from PyMPC's scenario_mpc/solver.py and solver/casadi_solver.py.

Critical fix: Creates a FRESH CasADi Opti() on every solve() call.
This prevents the constraint accumulation bug where obstacle constraints
added via opti.subject_to() accumulate across calls (CasADi Opti never
removes constraints, so reusing the same Opti object causes the problem
to grow unbounded and become infeasible).

Key features:
- Fresh Opti per solve (correct constraint handling)
- Ackermann vehicle dynamics with RK4 integration
- Contouring + lag cost for path following
- Parameterized obstacle avoidance (linearized halfspace constraints)
- Road boundary soft constraints
- Warm-starting from previous solution
- Curvature-adaptive velocity reference
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

from .dynamics import AckermannDynamics
from .spline_path import CubicSplinePath


@dataclass
class MPCCConfig:
    """MPCC solver configuration."""
    # Horizon
    horizon: int = 20
    dt: float = 0.1

    # Vehicle
    wheelbase: float = 0.256
    max_velocity: float = 0.60
    min_velocity: float = 0.0
    max_steering: float = 0.45
    max_acceleration: float = 0.6
    max_steering_rate: float = 0.6

    # Cost weights — contour > lag to prevent lane violations
    contour_weight: float = 25.0
    lag_weight: float = 5.0
    velocity_weight: float = 2.0
    steering_weight: float = 3.0
    acceleration_weight: float = 1.5
    steering_rate_weight: float = 4.0
    jerk_weight: float = 0.5

    # Progress reward
    progress_weight: float = 0.5

    # Obstacle avoidance
    max_obstacles: int = 10
    robot_radius: float = 0.13
    safety_margin: float = 0.10

    # Road boundaries (soft constraints)
    boundary_weight: float = 30.0

    # Solver
    max_iter: int = 75
    tolerance: float = 1e-5

    # Reference velocity
    reference_velocity: float = 0.50


@dataclass
class MPCCResult:
    """Result from MPCC solver."""
    v_cmd: float = 0.0
    delta_cmd: float = 0.0
    predicted_trajectory: np.ndarray = field(default_factory=lambda: np.zeros((1, 3)))
    solve_time: float = 0.0
    success: bool = False
    cost: float = float('inf')


class MPCCSolver:
    """
    MPCC solver using CasADi with fresh Opti per solve.

    Following PyMPC's pattern from scenario_mpc/solver.py:
    a new CasADi Opti() is created on each solve() call to avoid
    constraint accumulation.
    """

    def __init__(self, config: MPCCConfig):
        self.config = config
        self.dynamics = AckermannDynamics(
            wheelbase=config.wheelbase, dt=config.dt)

        # Warm-start storage
        self._prev_X: Optional[np.ndarray] = None
        self._prev_U: Optional[np.ndarray] = None

        # Build CasADi dynamics function (reusable across solves)
        if HAS_CASADI:
            self._build_dynamics_function()

    def _build_dynamics_function(self):
        """Build CasADi function for dynamics (RK4). Reused across solves."""
        nx = self.dynamics.nx
        nu = self.dynamics.nu
        dt = self.config.dt

        x_sym = ca.SX.sym('x', nx)
        u_sym = ca.SX.sym('u', nu)

        # RK4 integration
        k1 = self._casadi_dynamics(x_sym, u_sym)
        k2 = self._casadi_dynamics(x_sym + 0.5 * dt * k1, u_sym)
        k3 = self._casadi_dynamics(x_sym + 0.5 * dt * k2, u_sym)
        k4 = self._casadi_dynamics(x_sym + dt * k3, u_sym)
        x_next = x_sym + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        self._f_rk4 = ca.Function('f_rk4', [x_sym, u_sym], [x_next])

    def _casadi_dynamics(self, state, control):
        """CasADi symbolic Ackermann dynamics with slip angle."""
        L = self.config.wheelbase
        beta = ca.atan(ca.tan(state[4]) / 2.0)
        return ca.vertcat(
            state[3] * ca.cos(state[2] + beta),
            state[3] * ca.sin(state[2] + beta),
            state[3] / L * ca.sin(beta),
            control[0],
            control[1],
        )

    def solve(
        self,
        x0: np.ndarray,
        path: CubicSplinePath,
        current_progress: float,
        obstacles: Optional[List[Tuple[float, float, float]]] = None,
        boundary_constraints: Optional[List[Tuple[np.ndarray, float, float]]] = None,
    ) -> MPCCResult:
        """
        Solve the MPCC problem.

        IMPORTANT: Creates a fresh CasADi Opti() each call to prevent
        constraint accumulation (see module docstring).

        Args:
            x0: Initial state [x, y, theta, v, delta]
            path: Reference path (CubicSplinePath)
            current_progress: Current arc-length progress on path
            obstacles: List of (x, y, radius) tuples
            boundary_constraints: List of (normal_vec, b_left, b_right) per horizon step

        Returns:
            MPCCResult with optimal commands and trajectory
        """
        if not HAS_CASADI:
            return self._solve_fallback(x0, path, current_progress, obstacles)

        import time
        t_start = time.time()

        cfg = self.config
        N = cfg.horizon

        if obstacles is None:
            obstacles = []

        try:
            result = self._solve_casadi(
                x0, path, current_progress, obstacles, boundary_constraints)
            result.solve_time = time.time() - t_start
            return result
        except Exception:
            # CasADi solve failed - use fallback
            result = self._solve_fallback(x0, path, current_progress, obstacles)
            result.solve_time = time.time() - t_start
            return result

    def _solve_casadi(
        self,
        x0: np.ndarray,
        path: CubicSplinePath,
        current_progress: float,
        obstacles: List[Tuple[float, float, float]],
        boundary_constraints: Optional[List[Tuple[np.ndarray, float, float]]],
    ) -> MPCCResult:
        """
        Core CasADi solve - creates fresh Opti() each call.

        This follows the pattern from PyMPC's scenario_mpc/solver.py:
        ```python
        opti = cs.Opti()  # Fresh each solve
        X = opti.variable(...)
        U = opti.variable(...)
        opti.subject_to(...)  # All constraints added fresh
        sol = opti.solve()
        ```
        """
        cfg = self.config
        N = cfg.horizon
        nx = self.dynamics.nx  # 5: [x, y, theta, v, delta]
        nu = self.dynamics.nu  # 2: [a, delta_dot]

        # === Fresh Opti() - prevents constraint accumulation ===
        opti = ca.Opti()

        # Decision variables
        X = opti.variable(nx, N + 1)
        U = opti.variable(nu, N)

        # Virtual progress variable (how far along path)
        S = opti.variable(N + 1)

        # === Dynamics constraints (RK4) ===
        opti.subject_to(X[:, 0] == x0)
        opti.subject_to(S[0] == current_progress)

        for k in range(N):
            x_next = self._f_rk4(X[:, k], U[:, k])
            opti.subject_to(X[:, k + 1] == x_next)

        # === State bounds ===
        for k in range(N + 1):
            opti.subject_to(X[3, k] >= cfg.min_velocity)
            opti.subject_to(X[3, k] <= cfg.max_velocity)
            opti.subject_to(X[4, k] >= -cfg.max_steering)
            opti.subject_to(X[4, k] <= cfg.max_steering)
            # Progress must increase
            opti.subject_to(S[k] >= current_progress)
            opti.subject_to(S[k] <= path.total_length)

        # === Control bounds ===
        for k in range(N):
            opti.subject_to(U[0, k] >= -cfg.max_acceleration)
            opti.subject_to(U[0, k] <= cfg.max_acceleration)
            opti.subject_to(U[1, k] >= -cfg.max_steering_rate)
            opti.subject_to(U[1, k] <= cfg.max_steering_rate)

        # Progress dynamics: s increases proportional to velocity
        for k in range(N):
            # s_dot ≈ v (progress rate approximately equals velocity)
            s_dot = X[3, k]  # Forward progress = velocity
            opti.subject_to(S[k + 1] == S[k] + cfg.dt * s_dot)

        # === Cost function ===
        cost = 0.0

        # Pre-compute path references for each horizon step
        path_refs = []
        for k in range(N + 1):
            s_k = current_progress + k * cfg.reference_velocity * cfg.dt
            s_k = min(s_k, path.total_length - 0.01)
            ref_x, ref_y, cos_t, sin_t = path.get_path_reference(s_k)
            path_refs.append((ref_x, ref_y, cos_t, sin_t))

        for k in range(N):
            ref_x, ref_y, cos_t, sin_t = path_refs[k]

            # Contouring and lag errors
            dx = X[0, k] - ref_x
            dy = X[1, k] - ref_y
            e_c = -sin_t * dx + cos_t * dy   # Lateral (contouring)
            e_l = cos_t * dx + sin_t * dy     # Longitudinal (lag)

            # Stage cost
            cost += cfg.contour_weight * e_c**2
            cost += cfg.lag_weight * e_l**2

            # Curvature-adaptive velocity reference
            # Stronger exponential decay (-1.2) for tight turns on 1:10 scale track.
            # Previous -0.5 was too gentle, causing overspeed through curves.
            curvature = abs(path.get_curvature(
                current_progress + k * cfg.reference_velocity * cfg.dt))
            v_ref = cfg.reference_velocity * np.exp(-1.2 * curvature)
            v_ref = np.clip(v_ref, 0.08, cfg.max_velocity)
            cost += cfg.velocity_weight * (X[3, k] - v_ref)**2

            # Control effort
            cost += cfg.acceleration_weight * U[0, k]**2
            cost += cfg.steering_rate_weight * U[1, k]**2
            cost += cfg.steering_weight * X[4, k]**2

        # Terminal contouring cost (3x weight to anchor trajectory within lane)
        ref_x, ref_y, cos_t, sin_t = path_refs[N]
        dx = X[0, N] - ref_x
        dy = X[1, N] - ref_y
        e_c = -sin_t * dx + cos_t * dy
        e_l = cos_t * dx + sin_t * dy
        cost += 3.0 * cfg.contour_weight * e_c**2
        cost += 2.0 * cfg.lag_weight * e_l**2

        # Control smoothness (jerk penalty)
        for k in range(N - 1):
            cost += cfg.jerk_weight * (U[0, k + 1] - U[0, k])**2
            cost += cfg.jerk_weight * (U[1, k + 1] - U[1, k])**2

        # Progress reward
        cost -= cfg.progress_weight * (S[N] - S[0])

        # === Obstacle constraints (linearized, all added fresh) ===
        n_obs = min(len(obstacles), cfg.max_obstacles)
        for k in range(N):
            for i in range(n_obs):
                obs_x, obs_y, obs_r = obstacles[i][0], obstacles[i][1], obstacles[i][2]
                obs_vx = obstacles[i][3] if len(obstacles[i]) > 3 else 0.0
                obs_vy = obstacles[i][4] if len(obstacles[i]) > 4 else 0.0
                # Predict obstacle position at this time step
                t_k = k * cfg.dt
                pred_obs_x = obs_x + obs_vx * t_k
                pred_obs_y = obs_y + obs_vy * t_k
                obs_x, obs_y = pred_obs_x, pred_obs_y
                safe_r = obs_r + cfg.robot_radius + cfg.safety_margin

                # Linearization point
                if self._prev_X is not None:
                    lin_x = self._prev_X[0, min(k, self._prev_X.shape[1] - 1)]
                    lin_y = self._prev_X[1, min(k, self._prev_X.shape[1] - 1)]
                else:
                    lin_x = x0[0]
                    lin_y = x0[1]

                ddx = lin_x - obs_x
                ddy = lin_y - obs_y
                dist = np.sqrt(ddx**2 + ddy**2)

                if dist > 0.01:
                    nx_dir = ddx / dist
                    ny_dir = ddy / dist
                    opti.subject_to(
                        nx_dir * (X[0, k] - obs_x) +
                        ny_dir * (X[1, k] - obs_y) >= safe_r
                    )

        # === Road boundary soft constraints ===
        if boundary_constraints is not None:
            for k in range(min(N, len(boundary_constraints))):
                normal, b_left, b_right = boundary_constraints[k]
                # Left boundary violation
                left_val = normal[0] * X[0, k] + normal[1] * X[1, k] - b_left
                # Right boundary violation
                right_val = -normal[0] * X[0, k] - normal[1] * X[1, k] - b_right
                cost += cfg.boundary_weight * ca.fmax(0, left_val)**2
                cost += cfg.boundary_weight * ca.fmax(0, right_val)**2

        opti.minimize(cost)

        # === Solver options ===
        opts = {
            'ipopt.print_level': 0,
            'print_time': 0,
            'ipopt.max_iter': cfg.max_iter,
            'ipopt.tol': cfg.tolerance,
            'ipopt.warm_start_init_point': 'yes',
            'ipopt.acceptable_tol': cfg.tolerance * 10,
            'ipopt.acceptable_iter': 5,
        }
        opti.solver('ipopt', opts)

        # === Warm start ===
        if self._prev_X is not None:
            # Shift previous solution forward by one step
            for k in range(N + 1):
                src_k = min(k + 1, self._prev_X.shape[1] - 1)
                opti.set_initial(X[:, k], self._prev_X[:, src_k])
                opti.set_initial(S[k], self._prev_S[min(src_k, len(self._prev_S) - 1)])
            for k in range(N):
                src_k = min(k + 1, self._prev_U.shape[1] - 1)
                opti.set_initial(U[:, k], self._prev_U[:, src_k])
        else:
            # Initialize with straight-line prediction
            for k in range(N + 1):
                opti.set_initial(X[0, k], x0[0] + k * cfg.dt * x0[3] * np.cos(x0[2]))
                opti.set_initial(X[1, k], x0[1] + k * cfg.dt * x0[3] * np.sin(x0[2]))
                opti.set_initial(X[2, k], x0[2])
                opti.set_initial(X[3, k], max(x0[3], 0.1))
                opti.set_initial(X[4, k], x0[4] if len(x0) > 4 else 0.0)
                opti.set_initial(S[k], current_progress + k * cfg.dt * cfg.reference_velocity)

        # === Solve ===
        sol = opti.solve()

        # Extract solution
        X_sol = sol.value(X)
        U_sol = sol.value(U)
        S_sol = sol.value(S)

        # Store for warm start
        self._prev_X = X_sol
        self._prev_U = U_sol
        self._prev_S = S_sol

        # Return result
        v_cmd = float(X_sol[3, 1])
        delta_cmd = float(X_sol[4, 1])
        predicted = X_sol[:3, :].T  # [N+1, 3] of [x, y, theta]

        return MPCCResult(
            v_cmd=v_cmd,
            delta_cmd=delta_cmd,
            predicted_trajectory=predicted,
            success=True,
            cost=float(sol.value(opti.f)),
        )

    def _solve_fallback(
        self,
        x0: np.ndarray,
        path: CubicSplinePath,
        current_progress: float,
        obstacles: Optional[List[Tuple[float, float, float]]],
    ) -> MPCCResult:
        """
        Pure Pursuit + Stanley hybrid fallback (no CasADi needed).

        Used when CasADi is unavailable or solver fails.
        """
        cfg = self.config
        N = cfg.horizon
        x, y, theta = x0[0], x0[1], x0[2]
        v = x0[3] if len(x0) > 3 else 0.0

        # --- Stanley cross-track correction blended with Pure Pursuit ---
        # Use a lookahead that scales with speed but stays reasonable
        L_d = np.clip(0.30 + 0.8 * abs(v), 0.30, 0.80)
        la_progress = min(current_progress + L_d, path.total_length - 0.01)
        la_x, la_y = path.get_position(la_progress)

        dx = la_x - x
        dy = la_y - y
        dist_to_la = np.sqrt(dx**2 + dy**2)

        if dist_to_la < 0.01:
            delta_cmd = 0.0
        else:
            angle_to_la = np.arctan2(dy, dx)
            alpha = self._normalize_angle(angle_to_la - theta)
            delta_cmd = np.arctan2(2.0 * cfg.wheelbase * np.sin(alpha), dist_to_la)
            delta_cmd = np.clip(delta_cmd, -cfg.max_steering, cfg.max_steering)

        # --- Velocity ---
        curvature = abs(path.get_curvature(current_progress))
        v_cmd = cfg.reference_velocity * np.exp(-1.2 * curvature)
        v_cmd = np.clip(v_cmd, 0.12, cfg.max_velocity)

        # Heading error reduction - smoother progressive slowdown
        # Use lookahead tangent instead of tangent at current progress,
        # which can be distorted by spline boundary effects
        la_tangent = path.get_tangent(min(current_progress + 0.15, path.total_length - 0.01))
        heading_err = abs(self._normalize_angle(la_tangent - theta))
        if heading_err > np.radians(90):
            # Very large error: creep forward to let Pure Pursuit correct
            v_cmd = min(v_cmd, 0.10)
        elif heading_err > np.radians(45):
            # Large error: proportional slowdown
            scale = 1.0 - (heading_err - np.radians(45)) / np.radians(45)
            v_cmd = min(v_cmd, cfg.reference_velocity * max(scale, 0.25))

        # Obstacle slowdown (velocity-aware for moving obstacles)
        if obstacles:
            for i, obs in enumerate(obstacles):
                obs_x, obs_y, obs_r = obs[0], obs[1], obs[2]
                obs_vx = obs[3] if len(obs) > 3 else 0.0
                obs_vy = obs[4] if len(obs) > 4 else 0.0

                ddx = obs_x - x
                ddy = obs_y - y
                dist = np.sqrt(ddx**2 + ddy**2)
                angle_to_obs = np.arctan2(ddy, ddx)
                if abs(self._normalize_angle(angle_to_obs - theta)) > np.pi / 3:
                    continue

                # Predict obstacle position at time of closest approach
                if obs_vx != 0.0 or obs_vy != 0.0:
                    t_approach = dist / max(abs(v_cmd), 0.01)
                    pred_x = obs_x + obs_vx * t_approach
                    pred_y = obs_y + obs_vy * t_approach
                    ddx = pred_x - x
                    ddy = pred_y - y
                    dist = np.sqrt(ddx**2 + ddy**2)

                eff = dist - obs_r - cfg.robot_radius
                if eff < 0.05:
                    v_cmd = 0.0
                elif eff < 0.8:
                    scale = (eff - 0.05) / 0.75
                    v_cmd = min(v_cmd, cfg.reference_velocity * np.clip(scale, 0.1, 1.0))

        v_cmd = np.clip(v_cmd, 0.0, cfg.max_velocity)

        # Predicted trajectory
        predicted = np.zeros((N + 1, 3))
        px, py, pt = x, y, theta
        for k in range(N + 1):
            predicted[k] = [px, py, pt]
            px += v_cmd * np.cos(pt) * cfg.dt
            py += v_cmd * np.sin(pt) * cfg.dt
            pt += v_cmd / cfg.wheelbase * np.tan(delta_cmd) * cfg.dt

        return MPCCResult(
            v_cmd=v_cmd,
            delta_cmd=delta_cmd,
            predicted_trajectory=predicted,
            success=True,
        )

    @staticmethod
    def _normalize_angle(angle: float) -> float:
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

    def reset(self):
        """Reset warm-start state (call when path changes)."""
        self._prev_X = None
        self._prev_U = None
        self._prev_S = None
