"""
Ackermann vehicle dynamics for QCar2.

Adapted from PyMPC's EgoDynamics (scenario_mpc/dynamics.py) but using
the Ackermann (bicycle) model instead of unicycle for better accuracy
with QCar2's steering geometry.

State: [x, y, theta, v, delta]
  x, y  - position
  theta  - heading
  v      - velocity
  delta  - steering angle

Control: [a, delta_dot]
  a         - acceleration
  delta_dot - steering rate
"""

import numpy as np
from typing import Union, Optional

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False


class AckermannDynamics:
    """
    Ackermann (bicycle) dynamics model for QCar2 with slip angle.

    Uses bicycle model with slip angle beta for CG offset:
        beta = atan(tan(delta) / 2)

    Continuous dynamics:
        dx/dt     = v * cos(theta + beta)
        dy/dt     = v * sin(theta + beta)
        dtheta/dt = v / L * sin(beta)
        dv/dt     = a
        ddelta/dt = delta_dot
    """

    def __init__(self, wheelbase: float = 0.256, dt: float = 0.1):
        self.L = wheelbase
        self.dt = dt
        self.nx = 5  # [x, y, theta, v, delta]
        self.nu = 2  # [a, delta_dot]

    def continuous_dynamics(self, state, control):
        """Compute state derivative for Ackermann model with slip angle."""
        if HAS_CASADI and (isinstance(state, (ca.SX, ca.MX)) or
                           isinstance(control, (ca.SX, ca.MX))):
            beta = ca.atan(ca.tan(state[4]) / 2.0)
            return ca.vertcat(
                state[3] * ca.cos(state[2] + beta),
                state[3] * ca.sin(state[2] + beta),
                state[3] / self.L * ca.sin(beta),
                control[0],
                control[1],
            )
        else:
            beta = np.arctan(np.tan(state[4]) / 2.0)
            return np.array([
                state[3] * np.cos(state[2] + beta),
                state[3] * np.sin(state[2] + beta),
                state[3] / self.L * np.sin(beta),
                control[0],
                control[1],
            ])

    def rk4_step(self, state, control, dt: Optional[float] = None):
        """Integrate one step using RK4 (more accurate than Euler)."""
        h = dt if dt is not None else self.dt
        k1 = self.continuous_dynamics(state, control)
        k2 = self.continuous_dynamics(state + 0.5 * h * k1, control)
        k3 = self.continuous_dynamics(state + 0.5 * h * k2, control)
        k4 = self.continuous_dynamics(state + h * k3, control)
        return state + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    def euler_step(self, state, control, dt: Optional[float] = None):
        """Integrate one step using forward Euler (faster, less accurate)."""
        h = dt if dt is not None else self.dt
        return state + h * self.continuous_dynamics(state, control)

    def simulate(self, x0: np.ndarray, controls: np.ndarray,
                 method: str = 'rk4') -> np.ndarray:
        """
        Simulate trajectory from initial state with control sequence.

        Args:
            x0: Initial state [nx]
            controls: Control sequence [N, nu]
            method: 'rk4' or 'euler'

        Returns:
            States trajectory [N+1, nx]
        """
        N = controls.shape[0]
        states = np.zeros((N + 1, self.nx))
        states[0] = x0
        step_fn = self.rk4_step if method == 'rk4' else self.euler_step
        for k in range(N):
            states[k + 1] = step_fn(states[k], controls[k])
        return states
